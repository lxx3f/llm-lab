"""Reproducible Dense Transformer training, validation, checkpoint, and generation."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from architecture_lab.data import TokenStreamBatcher, load_token_cache
from architecture_lab.models.dense_transformer import DenseTransformer, TransformerConfig
from architecture_lab.models.gqa_transformer import GQAConfig, GQATransformer
from architecture_lab.models.mla_transformer import MLAConfig, MLATransformer
from architecture_lab.tokenization import BPETokenizer
from architecture_lab.training.results import build_training_result


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def causal_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute next-token loss for input windows and explicit shifted targets."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits must be [batch, time, vocab] and targets [batch, time]")
    if logits.size(0) != targets.size(0) or logits.size(1) != targets.size(1):
        raise ValueError("logits and targets must have matching batch/time dimensions")
    if logits.size(1) < 2:
        raise ValueError("at least two positions are required for causal loss")
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    return device


def resolve_dtype(value: str) -> torch.dtype:
    name = value.replace("torch.", "")
    try:
        dtype = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"unknown torch dtype: {value}") from error
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"not a torch dtype: {value}")
    return dtype


def build_model_config(settings: dict[str, Any], vocab_size: int) -> TransformerConfig:
    model = settings["model"]
    configured_vocab = model.get("vocab_size")
    if configured_vocab is not None and int(configured_vocab) != vocab_size:
        raise ValueError(
            f"config vocab_size {configured_vocab} does not match tokenizer vocab_size {vocab_size}"
        )
    architecture = model.get("architecture", "DenseTransformer")
    common_kwargs = dict(
        vocab_size=vocab_size,
        max_seq_len=int(model["max_seq_len"]),
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        n_layers=int(model["n_layers"]),
        d_ff=int(model["d_ff"]),
        dropout=float(model.get("dropout", 0.0)),
        rope_base=float(model.get("rope_base", 10000.0)),
    )
    if architecture == "DenseTransformer":
        return TransformerConfig(**common_kwargs)
    if architecture == "GQATransformer":
        num_kv_heads = int(model.get("num_kv_heads", model["n_heads"]))
        return GQAConfig(**common_kwargs, num_kv_heads=num_kv_heads)
    if architecture == "MLATransformer":
        latent_dim = int(model.get("latent_dim", 64))
        return MLAConfig(**common_kwargs, latent_dim=latent_dim)
    raise ValueError(
        f"unsupported architecture {architecture!r}; "
        "expected DenseTransformer | GQATransformer | MLATransformer"
    )


def build_model(config: TransformerConfig) -> nn.Module:
    """Instantiate the model class matching ``config`` (Dense/GQA/MLA)."""
    if isinstance(config, MLAConfig):
        return MLATransformer(config)
    if isinstance(config, GQAConfig):
        return GQATransformer(config)
    if isinstance(config, TransformerConfig):
        return DenseTransformer(config)
    raise ValueError(f"unsupported config type {type(config).__name__}")


def _metadata_hash(path: Path) -> str:
    return sha256_file(path)


@dataclass(frozen=True)
class TrainingState:
    step: int = 0
    epoch: int = 0


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and total_steps positive")
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps cannot exceed total_steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        if step >= total_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def resolve_amp(training: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype, bool]:
    config = training.get("amp", {}) or {}
    enabled = bool(config.get("enabled", False))
    if enabled and device.type not in {"cuda", "cpu"}:
        raise ValueError("AMP is only supported on CUDA or CPU devices")
    name = str(config.get("dtype", "float16")).replace("torch.", "")
    try:
        amp_dtype = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"unknown AMP dtype: {name}") from error
    if amp_dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("AMP dtype must be float16 or bfloat16")
    scaler_enabled = enabled and device.type == "cuda" and amp_dtype == torch.float16
    return enabled, amp_dtype, scaler_enabled


def _checkpoint_payload(
    *,
    model: DenseTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    state: TrainingState,
    settings: dict[str, Any],
    tokenizer_path: Path,
    train_metadata_path: Path,
    validation_metadata_path: Path,
) -> dict[str, Any]:
    return {
        "checkpoint_type": "llm-lab-dense-checkpoint",
        "checkpoint_version": "1.0",
        "model_config": asdict(model.config),
        "training_state": asdict(state),
        "settings": settings,
        "tokenizer": {
            "path": str(tokenizer_path),
            "artifact_sha256": sha256_file(tokenizer_path),
            "vocab_size": model.config.vocab_size,
        },
        "cache_metadata_sha256": {
            "train": _metadata_hash(train_metadata_path),
            "validation": _metadata_hash(validation_metadata_path),
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: DenseTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    state: TrainingState,
    settings: dict[str, Any],
    tokenizer_path: Path,
    train_metadata_path: Path,
    validation_metadata_path: Path,
) -> None:
    """Atomically save model, optimizer, training state, and data bindings."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        state=state,
        settings=settings,
        tokenizer_path=tokenizer_path,
        train_metadata_path=train_metadata_path,
        validation_metadata_path=validation_metadata_path,
    )
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".tmp", dir=destination.parent, delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: str | Path,
    *,
    model: DenseTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    tokenizer_path: Path,
    train_metadata_path: Path,
    validation_metadata_path: Path,
    map_location: torch.device | str,
) -> TrainingState:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("checkpoint_type") != "llm-lab-dense-checkpoint":
        raise ValueError("unsupported Dense checkpoint type")
    if payload.get("checkpoint_version") != "1.0":
        raise ValueError("unsupported Dense checkpoint version")
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("checkpoint model config does not match current config")
    tokenizer = payload.get("tokenizer", {})
    if tokenizer.get("artifact_sha256") != sha256_file(tokenizer_path):
        raise ValueError("checkpoint tokenizer hash does not match current artifact")
    cache_hashes = payload.get("cache_metadata_sha256", {})
    if cache_hashes.get("train") != _metadata_hash(train_metadata_path):
        raise ValueError("checkpoint train cache metadata hash mismatch")
    if cache_hashes.get("validation") != _metadata_hash(validation_metadata_path):
        raise ValueError("checkpoint validation cache metadata hash mismatch")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload.get("scheduler_state", scheduler.state_dict()))
    scaler_state = payload.get("scaler_state")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    raw_state = payload.get("training_state", {})
    return TrainingState(step=int(raw_state.get("step", 0)), epoch=int(raw_state.get("epoch", 0)))


def evaluate(
    model: DenseTransformer,
    batcher: TokenStreamBatcher,
    *,
    batches: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        with torch.no_grad():
            for index, (inputs, targets) in enumerate(batcher.iter_epoch(device=device)):
                if index >= batches:
                    break
                logits, _ = model(inputs)
                losses.append(float(causal_loss(logits, targets)))
        if not losses:
            raise ValueError("validation produced no batches")
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def compute_curve_summary(
    train_loss_samples: list[dict[str, Any]],
    validation_losses: dict[str, float],
) -> dict[str, Any]:
    """Compute a one-shot summary of a Dense training curve.

    Args:
        train_loss_samples: list of dicts with at least ``{"step", "loss", "lr"}``;
            typically the JSON-shaped samples emitted by ``train()`` at each
            ``log_interval`` boundary. Sampled, not per-step.
        validation_losses: mapping from step (int-like str) to validation loss.

    Returns:
        Dict matching ``schemas/dense_training_result.schema.json#$defs/metrics/properties/curve_summary``.
        ``None`` entries are emitted for first/last/min when the corresponding
        series is empty.
    """
    train_loss_first: float | None = train_loss_samples[0]["loss"] if train_loss_samples else None
    train_loss_last: float | None = train_loss_samples[-1]["loss"] if train_loss_samples else None
    delta_train_loss: float | None = (
        (train_loss_last - train_loss_first) if (train_loss_first is not None and train_loss_last is not None) else None
    )

    parsed_val: list[tuple[int, float]] = []
    for step_key, loss in validation_losses.items():
        try:
            step_int = int(step_key)
        except (TypeError, ValueError):
            continue
        try:
            loss_f = float(loss)
        except (TypeError, ValueError):
            continue
        parsed_val.append((step_int, loss_f))

    if parsed_val:
        val_loss_min = min(loss for _, loss in parsed_val)
        val_loss_min_step = next(step for step, loss in parsed_val if loss == val_loss_min)
        val_loss_last = parsed_val[-1][1]
        delta_val_loss = val_loss_last - val_loss_min
    else:
        val_loss_min = None
        val_loss_min_step = None
        val_loss_last = None
        delta_val_loss = None

    return {
        "train_loss_first": train_loss_first,
        "train_loss_last": train_loss_last,
        "val_loss_min": val_loss_min,
        "val_loss_min_step": val_loss_min_step,
        "val_loss_last": val_loss_last,
        "delta_train_loss": delta_train_loss,
        "delta_val_loss": delta_val_loss,
        "train_loss_sample_count": len(train_loss_samples),
        "val_loss_count": len(parsed_val),
    }


def generate_text(
    model: DenseTransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(generated[0].tolist())


def load_settings(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    if not isinstance(settings, dict):
        raise ValueError("training config must be a YAML object")
    return settings


def train(settings: dict[str, Any], *, resume: str | Path | None = None) -> dict[str, Any]:
    data = settings["data"]
    training = settings["training"]
    tokenizer_path = Path(data["tokenizer"])
    train_token_path = Path(data["train_tokens"])
    train_metadata_path = Path(data["train_metadata"])
    validation_token_path = Path(data["validation_tokens"])
    validation_metadata_path = Path(data["validation_metadata"])
    tokenizer = BPETokenizer.load(tokenizer_path)
    requested_device = str(training.get("device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = resolve_device(requested_device)
    dtype = resolve_dtype(str(training.get("dtype", "float32")))
    set_seed(int(training.get("seed", 42)))
    model_config = build_model_config(settings, tokenizer.vocab_size)
    sequence_length = int(training["sequence_length"])
    if sequence_length > model_config.max_seq_len:
        raise ValueError("training sequence_length exceeds model max_seq_len")
    train_tokens, _ = load_token_cache(
        token_path=train_token_path,
        metadata_path=train_metadata_path,
        source_path=data.get("train_source"),
        tokenizer_path=tokenizer_path,
        split="train",
        max_bytes=data.get("train_max_bytes"),
    )
    validation_tokens, _ = load_token_cache(
        token_path=validation_token_path,
        metadata_path=validation_metadata_path,
        source_path=data.get("validation_source"),
        tokenizer_path=tokenizer_path,
        split="validation",
        max_bytes=data.get("validation_max_bytes"),
    )
    train_batcher = TokenStreamBatcher(
        train_tokens,
        batch_size=int(training["batch_size"]),
        sequence_length=sequence_length,
        seed=int(training.get("seed", 42)),
    )
    validation_batcher = TokenStreamBatcher(
        validation_tokens,
        batch_size=int(training["batch_size"]),
        sequence_length=sequence_length,
        seed=int(training.get("seed", 42)) + 1,
    )
    model = build_model(model_config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    gradient_accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    max_steps = int(training["max_steps"])
    scheduler_config = training.get("scheduler", {}) or {}
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(scheduler_config.get("warmup_steps", 0)),
        total_steps=max_steps,
        min_lr_ratio=float(scheduler_config.get("min_lr_ratio", 0.0)),
    )
    amp_enabled, amp_dtype, scaler_enabled = resolve_amp(training, device)
    scaler = torch.amp.GradScaler(device=device.type, enabled=scaler_enabled)
    amp_config = {
        "enabled": amp_enabled,
        "dtype": str(amp_dtype).replace("torch.", ""),
        "scaler_enabled": scaler_enabled,
    }
    state = TrainingState()
    if resume is not None:
        state = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer_path=tokenizer_path,
            train_metadata_path=train_metadata_path,
            validation_metadata_path=validation_metadata_path,
            map_location=device,
        )
    model.train()
    log_interval = int(training.get("log_interval", 1))
    validation_interval = int(training.get("validation_interval", max_steps))
    checkpoint_path = Path(training["checkpoint"])
    validation_batches = int(training.get("validation_batches", 1))
    losses: list[float] = []
    train_loss_samples: list[dict[str, Any]] = []
    validation_losses: dict[str, float] = {}
    epoch = state.epoch
    batch_iterator = iter(train_batcher.iter_epoch(device=device))
    while state.step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            try:
                inputs, targets = next(batch_iterator)
            except StopIteration:
                epoch += 1
                state = TrainingState(step=state.step, epoch=epoch)
                batch_iterator = iter(train_batcher.iter_epoch(device=device))
                inputs, targets = next(batch_iterator)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits, _ = model(inputs)
                micro_loss = causal_loss(logits, targets)
            accumulated_loss += float(micro_loss.detach())
            scaler.scale(micro_loss / gradient_accumulation_steps).backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)
        clip_value = training.get("gradient_clip_norm")
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_value))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        state = TrainingState(step=state.step + 1, epoch=epoch)
        step_loss = accumulated_loss / gradient_accumulation_steps
        losses.append(step_loss)
        if state.step % log_interval == 0:
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss_samples.append(
                {"step": int(state.step), "loss": step_loss, "lr": current_lr}
            )
            print(f"step={state.step} train_loss={step_loss:.6f} lr={current_lr:.8f}")
        if state.step % validation_interval == 0 or state.step == max_steps:
            validation_loss = evaluate(
                model, validation_batcher, batches=validation_batches, device=device
            )
            validation_losses[str(state.step)] = validation_loss
            print(f"step={state.step} validation_loss={validation_loss:.6f}")
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                state=state,
                settings=settings,
                tokenizer_path=tokenizer_path,
                train_metadata_path=train_metadata_path,
                validation_metadata_path=validation_metadata_path,
            )

    max_new_tokens = int(training.get("max_new_tokens", 0))
    prompt = str(training.get("prompt", ""))
    generated = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    ) if prompt and max_new_tokens > 0 else None
    curve_summary = compute_curve_summary(train_loss_samples, validation_losses)
    return build_training_result(
        settings=settings,
        model=model,
        tokenizer_path=tokenizer_path,
        train_token_path=train_token_path,
        train_metadata_path=train_metadata_path,
        validation_token_path=validation_token_path,
        validation_metadata_path=validation_metadata_path,
        device=device,
        model_dtype=dtype,
        optimizer_steps=state.step,
        epoch=state.epoch,
        last_train_loss=losses[-1] if losses else None,
        validation_losses=validation_losses,
        train_loss_samples=train_loss_samples,
        curve_summary=curve_summary,
        checkpoint_path=checkpoint_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        generated_text=generated,
        gradient_accumulation_steps=gradient_accumulation_steps,
        scheduler_config={
            "name": "warmup_cosine",
            "warmup_steps": int(scheduler_config.get("warmup_steps", 0)),
            "total_steps": max_steps,
            "min_lr_ratio": float(scheduler_config.get("min_lr_ratio", 0.0)),
            "final_learning_rate": optimizer.param_groups[0]["lr"],
        },
        amp_config=amp_config,
    )
