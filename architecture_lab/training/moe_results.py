"""Result records for MoE training experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from jsonschema import Draft202012Validator, FormatChecker

from architecture_lab.models.moe_transformer import MoETransformer, count_active_parameters, count_moe_parameters
from architecture_lab.tokenization import BPETokenizer
from architecture_lab.training.dense_training import sha256_file


def _cache_record(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache = metadata["cache"]
    source = metadata["source"]
    return {
        "split": cache["split"],
        "token_count": cache["token_count"],
        "encoded_bytes": source["encoded_bytes"],
        "metadata_sha256": sha256_file(metadata_path),
        "token_file_sha256": cache["token_file_sha256"],
    }


def _expert_capacity(tokens: int, num_experts: int, capacity_factor: float) -> int:
    import math

    if tokens <= 0 or num_experts <= 0 or capacity_factor <= 0:
        raise ValueError("tokens, num_experts, and capacity_factor must be positive")
    return max(1, math.ceil(capacity_factor * tokens / num_experts))


def build_moe_training_result(*, settings: dict[str, Any], model: MoETransformer, tokenizer_path: Path, train_token_path: Path, train_metadata_path: Path, validation_token_path: Path, validation_metadata_path: Path, device: torch.device, model_dtype: torch.dtype, optimizer_steps: int, epoch: int, train_metrics: list[tuple[float, float, float]], validation_metrics: dict[str, dict[str, float]], checkpoint_path: Path, prompt: str, max_new_tokens: int, generated_text: str | None, gradient_accumulation_steps: int, scheduler_config: dict[str, Any], amp_config: dict[str, Any]) -> dict[str, Any]:
    training = settings["training"]
    moe = settings["moe"]
    last = train_metrics[-1] if train_metrics else (None, None, None)
    return {
        "schema_version": "1.0",
        "result_type": "moe_training",
        "experiment_id": f"{settings['model']['name']}-moe-training",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "model": {
            "name": settings["model"]["name"],
            "architecture": "MoETransformer",
            "config": settings["model"],
            "moe_config": moe,
            "parameter_count_total": count_moe_parameters(model),
            "parameter_count_top1_active": count_active_parameters(model),
            "active_parameter_definition": "shared_parameters_plus_one_expert_per_moe_layer",
        },
        "data": {
            "data_version": json.loads(train_metadata_path.read_text(encoding="utf-8")).get("data_version"),
            "tokenizer": {"path": str(tokenizer_path), "artifact_sha256": sha256_file(tokenizer_path), "vocab_size": BPETokenizer.load(tokenizer_path).vocab_size},
            "train_cache": _cache_record(train_metadata_path),
            "validation_cache": _cache_record(validation_metadata_path),
        },
        "training": {
            "seed": int(training.get("seed", 42)), "device": str(device), "model_dtype": str(model_dtype).replace("torch.", ""), "optimizer_steps": optimizer_steps, "epoch": epoch, "micro_batch_size": int(training["batch_size"]), "gradient_accumulation_steps": gradient_accumulation_steps, "effective_batch_size": int(training["batch_size"]) * gradient_accumulation_steps, "sequence_length": int(training["sequence_length"]), "expert_capacity": _expert_capacity(int(training["batch_size"]) * int(training["sequence_length"]), int(moe["num_experts"]), float(moe["capacity_factor"])), "expert_capacity_definition": "ceil(capacity_factor * batch_size * sequence_length / num_experts)", "optimizer": {"name": "AdamW", "learning_rate": float(training["learning_rate"]), "weight_decay": float(training.get("weight_decay", 0.0)), "gradient_clip_norm": training.get("gradient_clip_norm")}, "scheduler": scheduler_config, "amp": amp_config, "collect_stats": False,
        },
        "metrics": {
            "last_train_lm_loss": last[0], "last_train_aux_loss": last[1], "last_train_total_loss": last[2], "validation_metrics": validation_metrics,
        },
        "artifacts": {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None},
        "generation": {"prompt": prompt, "max_new_tokens": max_new_tokens, "text": generated_text, "collect_stats": False, "prefill_capacity_factor": 1.0, "prefill_token_count": len(BPETokenizer.load(tokenizer_path).encode(prompt)), "prefill_expert_capacity": _expert_capacity(len(BPETokenizer.load(tokenizer_path).encode(prompt)), int(moe["num_experts"]), 1.0), "decode_capacity_factor": 2.0, "decode_token_count": 1, "decode_expert_capacity": _expert_capacity(1, int(moe["num_experts"]), 2.0)},
    }


def write_moe_training_result(result: dict[str, Any], path: str | Path) -> None:
    schema_path = Path(__file__).parents[2] / "schemas/moe_training_result.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(result), key=lambda error: list(error.path))
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "$"
        raise ValueError(f"invalid MoE training result at {location}: {errors[0].message}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
