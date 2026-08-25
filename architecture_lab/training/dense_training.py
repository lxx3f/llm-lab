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
from architecture_lab.tokenization import BPETokenizer


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
    return TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=int(model["max_seq_len"]),
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        n_layers=int(model["n_layers"]),
        d_ff=int(model["d_ff"]),
        dropout=float(model.get("dropout", 0.0)),
        rope_base=float(model.get("rope_base", 10000.0)),
    )


def _metadata_hash(path: Path) -> str:
    return sha256_file(path)


@dataclass(frozen=True)
class TrainingState:
    step: int = 0
    epoch: int = 0


def _checkpoint_payload(
    *,
    model: DenseTransformer,
    optimizer: torch.optim.Optimizer,
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
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: DenseTransformer,
    optimizer: torch.optim.Optimizer,
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
    model = DenseTransformer(model_config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    state = TrainingState()
    if resume is not None:
        state = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            tokenizer_path=tokenizer_path,
            train_metadata_path=train_metadata_path,
            validation_metadata_path=validation_metadata_path,
            map_location=device,
        )
    model.train()
    max_steps = int(training["max_steps"])
    log_interval = int(training.get("log_interval", 1))
    validation_interval = int(training.get("validation_interval", max_steps))
    checkpoint_path = Path(training["checkpoint"])
    validation_batches = int(training.get("validation_batches", 1))
    losses: list[float] = []
    validation_losses: dict[str, float] = {}
    epoch = state.epoch
    while state.step < max_steps:
        completed_epoch = True
        for inputs, targets in train_batcher.iter_epoch(device=device):
            if state.step >= max_steps:
                completed_epoch = False
                break
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss = causal_loss(logits, targets)
            loss.backward()
            clip_value = training.get("gradient_clip_norm")
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_value))
            optimizer.step()
            state = TrainingState(step=state.step + 1, epoch=epoch)
            losses.append(float(loss.detach()))
            if state.step % log_interval == 0:
                print(f"step={state.step} train_loss={losses[-1]:.6f}")
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
                    state=state,
                    settings=settings,
                    tokenizer_path=tokenizer_path,
                    train_metadata_path=train_metadata_path,
                    validation_metadata_path=validation_metadata_path,
                )
        if completed_epoch:
            epoch += 1
            state = TrainingState(step=state.step, epoch=epoch)

    max_new_tokens = int(training.get("max_new_tokens", 0))
    prompt = str(training.get("prompt", ""))
    generated = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    ) if prompt and max_new_tokens > 0 else None
    return {
        "step": state.step,
        "epoch": state.epoch,
        "last_train_loss": losses[-1] if losses else None,
        "validation_losses": validation_losses,
        "train_metadata_sha256": _metadata_hash(train_metadata_path),
        "validation_metadata_sha256": _metadata_hash(validation_metadata_path),
        "checkpoint": str(checkpoint_path),
        "config": settings,
        "generated_text": generated,
        "tokenizer_sha256": sha256_file(tokenizer_path),
    }
