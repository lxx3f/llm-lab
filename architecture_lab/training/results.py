"""Versioned result records for Dense Transformer training experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from jsonschema import Draft202012Validator, FormatChecker

from architecture_lab.models.dense_transformer import DenseTransformer, count_parameters
from architecture_lab.tokenization import BPETokenizer

DENSE_RESULT_SCHEMA_VERSION = "1.1"


def _sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_record(token_path: Path, metadata_path: Path) -> dict[str, Any]:
    metadata = _cache_metadata(metadata_path)
    cache = metadata["cache"]
    source = metadata["source"]
    return {
        "split": cache["split"],
        "token_count": cache["token_count"],
        "encoded_bytes": source["encoded_bytes"],
        "metadata_sha256": _sha256_file(metadata_path),
        "token_file_sha256": cache["token_file_sha256"],
    }


def build_training_result(
    *,
    settings: dict[str, Any],
    model: DenseTransformer,
    tokenizer_path: Path,
    train_token_path: Path,
    train_metadata_path: Path,
    validation_token_path: Path,
    validation_metadata_path: Path,
    device: torch.device,
    model_dtype: torch.dtype,
    optimizer_steps: int,
    epoch: int,
    last_train_loss: float | None,
    validation_losses: dict[str, float],
    train_loss_samples: list[dict[str, Any]] | None = None,
    curve_summary: dict[str, Any] | None = None,
    checkpoint_path: Path,
    prompt: str,
    max_new_tokens: int,
    generated_text: str | None,
    gradient_accumulation_steps: int,
    scheduler_config: dict[str, Any],
    amp_config: dict[str, Any],
) -> dict[str, Any]:
    data = settings["data"]
    training = settings["training"]
    checkpoint_sha256 = _sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    return {
        "schema_version": DENSE_RESULT_SCHEMA_VERSION,
        "result_type": "dense_training",
        "experiment_id": f"{settings['model']['name']}-dense-training",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "model": {
            "name": settings["model"]["name"],
            "architecture": settings["model"].get("architecture", "DenseTransformer"),
            "config": settings["model"],
            "parameter_count": count_parameters(model),
        },
        "data": {
            "data_version": _cache_metadata(train_metadata_path).get("data_version"),
            "tokenizer": {
                "path": str(tokenizer_path),
                "artifact_sha256": _sha256_file(tokenizer_path),
                "vocab_size": BPETokenizer.load(tokenizer_path).vocab_size,
            },
            "train_cache": _cache_record(train_token_path, train_metadata_path),
            "validation_cache": _cache_record(validation_token_path, validation_metadata_path),
        },
        "training": {
            "seed": int(training.get("seed", 42)),
            "device": str(device),
            "model_dtype": str(model_dtype).replace("torch.", ""),
            "optimizer_steps": optimizer_steps,
            "epoch": epoch,
            "micro_batch_size": int(training["batch_size"]),
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": int(training["batch_size"]) * gradient_accumulation_steps,
            "sequence_length": int(training["sequence_length"]),
            "optimizer": {
                "name": "AdamW",
                "learning_rate": float(training["learning_rate"]),
                "weight_decay": float(training.get("weight_decay", 0.0)),
                "gradient_clip_norm": training.get("gradient_clip_norm"),
            },
            "scheduler": scheduler_config,
            "amp": amp_config,
        },
        "metrics": {
            "last_train_loss": last_train_loss,
            "validation_losses": validation_losses,
            "train_losses": list(train_loss_samples) if train_loss_samples is not None else [],
            "curve_summary": dict(curve_summary) if curve_summary is not None else {
                "train_loss_first": None,
                "train_loss_last": None,
                "val_loss_min": None,
                "val_loss_min_step": None,
                "val_loss_last": None,
                "delta_train_loss": None,
                "delta_val_loss": None,
                "train_loss_sample_count": 0,
                "val_loss_count": 0,
            },
        },
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "generation": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "text": generated_text,
        },
    }


def write_training_result(result: dict[str, Any], path: str | Path) -> None:
    schema_path = Path(__file__).parents[2] / "schemas/dense_training_result.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(result),
        key=lambda error: list(error.path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "$"
        raise ValueError(f"invalid Dense training result at {location}: {errors[0].message}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
