"""Reproducible Top-1 MoE training, validation, checkpoint, and generation."""

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
import yaml
from torch import Tensor

from architecture_lab.data import TokenStreamBatcher, load_token_cache
from architecture_lab.models.dense_transformer import TransformerConfig
from architecture_lab.models.moe_transformer import (
    MoEConfig,
    MoETransformer,
    count_active_parameters,
    count_moe_parameters,
)
from architecture_lab.tokenization import BPETokenizer
from architecture_lab.training.dense_training import (
    build_scheduler,
    causal_loss,
    resolve_amp,
    resolve_device,
    resolve_dtype,
    set_seed,
    sha256_file,
)


@dataclass(frozen=True)
class MoETrainingState:
    step: int = 0
    epoch: int = 0


def _metadata_hash(path: Path) -> str:
    return sha256_file(path)


def build_model_configs(settings: dict[str, Any], vocab_size: int) -> tuple[TransformerConfig, MoEConfig]:
    model = settings["model"]
    configured_vocab = model.get("vocab_size")
    if configured_vocab is not None and int(configured_vocab) != vocab_size:
        raise ValueError("config vocab_size does not match tokenizer vocab_size")
    transformer = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=int(model["max_seq_len"]),
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        n_layers=int(model["n_layers"]),
        d_ff=int(model["d_ff"]),
        dropout=float(model.get("dropout", 0.0)),
        rope_base=float(model.get("rope_base", 10000.0)),
    )
    moe_settings = settings["moe"]
    moe = MoEConfig(
        num_experts=int(moe_settings.get("num_experts", 4)),
        capacity_factor=float(moe_settings.get("capacity_factor", 1.0)),
        aux_loss_weight=float(moe_settings.get("aux_loss_weight", 0.01)),
    )
    return transformer, moe


def _checkpoint_payload(
    *,
    model: MoETransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    state: MoETrainingState,
    settings: dict[str, Any],
    tokenizer_path: Path,
    train_metadata_path: Path,
    validation_metadata_path: Path,
) -> dict[str, Any]:
    return {
        "checkpoint_type": "llm-lab-moe-checkpoint",
        "checkpoint_version": "1.0",
        "model_config": asdict(model.config),
        "moe_config": asdict(model.moe_config),
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


def save_checkpoint(path: str | Path, **kwargs: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".tmp", dir=destination.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(_checkpoint_payload(**kwargs), temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: str | Path,
    *,
    model: MoETransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    tokenizer_path: Path,
    train_metadata_path: Path,
    validation_metadata_path: Path,
    map_location: torch.device,
) -> MoETrainingState:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("checkpoint_type") != "llm-lab-moe-checkpoint":
        raise ValueError("invalid MoE checkpoint type")
    if payload.get("checkpoint_version") != "1.0":
        raise ValueError("unsupported MoE checkpoint version")
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("checkpoint model config mismatch")
    if payload.get("moe_config") != asdict(model.moe_config):
        raise ValueError("checkpoint MoE config mismatch")
    tokenizer = payload.get("tokenizer", {})
    if tokenizer.get("artifact_sha256") != sha256_file(tokenizer_path):
        raise ValueError("checkpoint tokenizer hash mismatch")
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
    state = payload.get("training_state", {})
    return MoETrainingState(step=int(state.get("step", 0)), epoch=int(state.get("epoch", 0)))


def evaluate(model: MoETransformer, batcher: TokenStreamBatcher, *, batches: int, device: torch.device) -> tuple[float, float, float]:
    was_training = model.training
    model.eval()
    lm_losses: list[float] = []
    aux_losses: list[float] = []
    try:
        with torch.no_grad():
            for index, (inputs, targets) in enumerate(batcher.iter_epoch(device=device)):
                if index >= batches:
                    break
                logits, _, aux_loss = model(inputs, collect_stats=False)
                lm_loss = causal_loss(logits, targets)
                lm_losses.append(float(lm_loss))
                aux_losses.append(float(aux_loss))
        if not lm_losses:
            raise ValueError("validation produced no batches")
        lm = sum(lm_losses) / len(lm_losses)
        aux = sum(aux_losses) / len(aux_losses)
        return lm, aux, lm + model.moe_config.aux_loss_weight * aux
    finally:
        model.train(was_training)


def generate_text(model: MoETransformer, tokenizer: BPETokenizer, prompt: str, *, max_new_tokens: int, device: torch.device) -> str:
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        prefill_capacity_factor=1.0,
        decode_capacity_factor=2.0,
        collect_stats=False,
    )
    return tokenizer.decode(generated[0].tolist())


def load_settings(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    if not isinstance(settings, dict):
        raise ValueError("training config must be a YAML object")
    return settings


def build_result(**kwargs: Any) -> dict[str, Any]:
    from architecture_lab.training.moe_results import build_moe_training_result
    return build_moe_training_result(**kwargs)


def train(settings: dict[str, Any], *, resume: str | Path | None = None) -> dict[str, Any]:
    data, training = settings["data"], settings["training"]
    tokenizer_path = Path(data["tokenizer"])
    train_metadata_path = Path(data["train_metadata"])
    validation_metadata_path = Path(data["validation_metadata"])
    tokenizer = BPETokenizer.load(tokenizer_path)
    requested = str(training.get("device", "auto"))
    device = resolve_device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
    dtype = resolve_dtype(str(training.get("dtype", "float32")))
    set_seed(int(training.get("seed", 42)))
    transformer_config, moe_config = build_model_configs(settings, tokenizer.vocab_size)
    sequence_length = int(training["sequence_length"])
    if sequence_length > transformer_config.max_seq_len:
        raise ValueError("training sequence_length exceeds model max_seq_len")
    train_tokens, _ = load_token_cache(token_path=data["train_tokens"], metadata_path=train_metadata_path, source_path=data.get("train_source"), tokenizer_path=tokenizer_path, split="train", max_bytes=data.get("train_max_bytes"))
    validation_tokens, _ = load_token_cache(token_path=data["validation_tokens"], metadata_path=validation_metadata_path, source_path=data.get("validation_source"), tokenizer_path=tokenizer_path, split="validation", max_bytes=data.get("validation_max_bytes"))
    batch_size = int(training["batch_size"])
    train_batcher = TokenStreamBatcher(train_tokens, batch_size=batch_size, sequence_length=sequence_length, seed=int(training.get("seed", 42)))
    validation_batcher = TokenStreamBatcher(validation_tokens, batch_size=batch_size, sequence_length=sequence_length, seed=int(training.get("seed", 42)) + 1)
    model = MoETransformer(transformer_config, moe_config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    max_steps = int(training["max_steps"])
    scheduler_settings = training.get("scheduler", {}) or {}
    scheduler = build_scheduler(optimizer, warmup_steps=int(scheduler_settings.get("warmup_steps", 0)), total_steps=max_steps, min_lr_ratio=float(scheduler_settings.get("min_lr_ratio", 0.0)))
    amp_enabled, amp_dtype, scaler_enabled = resolve_amp(training, device)
    scaler = torch.amp.GradScaler(device=device.type, enabled=scaler_enabled)
    state = MoETrainingState()
    if resume is not None:
        state = load_checkpoint(resume, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, tokenizer_path=tokenizer_path, train_metadata_path=train_metadata_path, validation_metadata_path=validation_metadata_path, map_location=device)
    model.train()
    validation_interval = int(training.get("validation_interval", max_steps))
    validation_batches = int(training.get("validation_batches", 1))
    checkpoint_path = Path(training["checkpoint"])
    train_metrics: list[tuple[float, float, float]] = []
    validation_metrics: dict[str, dict[str, float]] = {}
    epoch = state.epoch
    iterator = iter(train_batcher.iter_epoch(device=device))
    while state.step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        sums = [0.0, 0.0, 0.0]
        for _ in range(accumulation):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                epoch += 1
                iterator = iter(train_batcher.iter_epoch(device=device))
                inputs, targets = next(iterator)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits, _, aux_loss = model(inputs, collect_stats=False)
                lm_loss = causal_loss(logits, targets)
                total_loss = model.total_loss(lm_loss, aux_loss)
            sums[0] += float(lm_loss.detach())
            sums[1] += float(aux_loss.detach())
            sums[2] += float(total_loss.detach())
            scaler.scale(total_loss / accumulation).backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)
        if training.get("gradient_clip_norm") is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        state = MoETrainingState(step=state.step + 1, epoch=epoch)
        metrics = tuple(value / accumulation for value in sums)
        train_metrics.append(metrics)
        if state.step % validation_interval == 0 or state.step == max_steps:
            lm, aux, total = evaluate(model, validation_batcher, batches=validation_batches, device=device)
            validation_metrics[str(state.step)] = {"lm_loss": lm, "aux_loss": aux, "total_loss": total}
            save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, state=state, settings=settings, tokenizer_path=tokenizer_path, train_metadata_path=train_metadata_path, validation_metadata_path=validation_metadata_path)
    prompt = str(training.get("prompt", ""))
    generated = generate_text(model, tokenizer, prompt, max_new_tokens=int(training.get("max_new_tokens", 0)), device=device) if prompt and int(training.get("max_new_tokens", 0)) > 0 else None
    return build_result(settings=settings, model=model, tokenizer_path=tokenizer_path, train_token_path=Path(data["train_tokens"]), train_metadata_path=train_metadata_path, validation_token_path=Path(data["validation_tokens"]), validation_metadata_path=validation_metadata_path, device=device, model_dtype=dtype, optimizer_steps=state.step, epoch=state.epoch, train_metrics=train_metrics, validation_metrics=validation_metrics, checkpoint_path=checkpoint_path, prompt=prompt, max_new_tokens=int(training.get("max_new_tokens", 0)), generated_text=generated, gradient_accumulation_steps=accumulation, scheduler_config={"name": "warmup_cosine", "warmup_steps": int(scheduler_settings.get("warmup_steps", 0)), "total_steps": max_steps, "min_lr_ratio": float(scheduler_settings.get("min_lr_ratio", 0.0)), "final_learning_rate": optimizer.param_groups[0]["lr"]}, amp_config={"enabled": amp_enabled, "dtype": str(amp_dtype).replace("torch.", ""), "scaler_enabled": scaler_enabled})
