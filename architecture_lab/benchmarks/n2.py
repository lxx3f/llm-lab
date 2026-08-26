"""Shared utilities for the N2 Dense/MoE fairness benchmark."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from architecture_lab.data.batching import TokenStreamBatcher, load_token_cache
from architecture_lab.models.dense_transformer import (
    DenseTransformer,
    TransformerConfig,
    count_parameters,
)
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
    sha256_file,
)


PREFILL_CAPACITY_FACTOR = 1.0
DECODE_CAPACITY_FACTOR = 2.0


def load_settings(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    if not isinstance(settings, dict):
        raise ValueError("N2 config must contain a mapping")
    return settings


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    return device


def resolve_dtype(value: str) -> torch.dtype:
    name = value.replace("torch.", "")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {value}")
    return dtype


def transformer_config(settings: dict[str, Any], *, d_ff: int, vocab_size: int) -> TransformerConfig:
    model = settings["model"]
    return TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=int(model["max_seq_len"]),
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        n_layers=int(model["n_layers"]),
        d_ff=d_ff,
        dropout=float(model.get("dropout", 0.0)),
        rope_base=float(model.get("rope_base", 10000.0)),
    )


def build_models(settings: dict[str, Any], architecture: str, d_ff: int, device: torch.device, dtype: torch.dtype, *, vocab_size: int | None = None):
    tokenizer = None
    if vocab_size is None:
        tokenizer = BPETokenizer.load(settings["data"]["tokenizer"])
        vocab_size = tokenizer.vocab_size
    config = transformer_config(settings, d_ff=d_ff, vocab_size=vocab_size)
    if architecture == "DenseTransformer":
        model = DenseTransformer(config).to(device=device, dtype=dtype).eval()
    elif architecture == "MoETransformer":
        moe_settings = settings["moe"]
        moe = MoEConfig(
            num_experts=int(moe_settings["num_experts"]),
            capacity_factor=float(moe_settings["capacity_factor"]),
            aux_loss_weight=float(moe_settings["aux_loss_weight"]),
        )
        model = MoETransformer(config, moe).to(device=device, dtype=dtype).eval()
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    return model, tokenizer


def parameter_counts(model: DenseTransformer | MoETransformer) -> tuple[int, int]:
    total = count_parameters(model)
    active = total if isinstance(model, DenseTransformer) else count_active_parameters(model)
    return total, active


def choose_protocol_d_ff(settings: dict[str, Any], protocol: str, vocab_size: int) -> dict[str, int]:
    base = int(settings["model"]["d_ff"])
    experts = int(settings["moe"]["num_experts"])
    device = torch.device("cpu")
    dtype = torch.float32
    dense, _ = build_models(settings, "DenseTransformer", base, device, dtype, vocab_size=vocab_size)
    dense_total, dense_active = parameter_counts(dense)
    if protocol == "A":
        candidates = range(max(1, base // experts - 2), base // experts + 3)
        selected = min(
            candidates,
            key=lambda width: abs(
                parameter_counts(build_models(settings, "MoETransformer", width, device, dtype, vocab_size=vocab_size)[0])[0]
                - dense_total
            ),
        )
        return {"dense_d_ff": base, "moe_d_ff": selected, "target_parameters": dense_total}
    if protocol == "B":
        # Protocol B fixes both Dense and per-expert widths to D_act. The MoE
        # router is shared overhead, so the resulting active delta is recorded
        # explicitly rather than silently changing the requested width.
        return {"dense_d_ff": base, "moe_d_ff": base, "target_parameters": dense_active}
    raise ValueError("protocol must be A or B")


def benchmark_inputs(
    settings: dict[str, Any],
    device: torch.device,
    *,
    split: str = "train",
) -> torch.Tensor:
    data = settings["data"]
    if split == "train":
        token_path, metadata_path = data["train_tokens"], data["train_metadata"]
    elif split == "validation":
        token_path, metadata_path = data["validation_tokens"], data["validation_metadata"]
    else:
        raise ValueError("split must be train or validation")
    tokens, _ = load_token_cache(
        token_path=token_path,
        metadata_path=metadata_path,
        tokenizer_path=data["tokenizer"],
        split=split,
        mmap=True,
    )
    batch = int(settings["benchmark"]["batch_size"])
    seq = int(settings["benchmark"]["sequence_length"])
    needed = batch * seq
    if tokens.numel() < needed + 1:
        raise ValueError(f"{split} cache is too short for N2 benchmark input")
    rows = [tokens[index * seq : (index + 1) * seq] for index in range(batch)]
    return torch.stack(rows).to(device=device, dtype=torch.long)


def train_for_benchmark(
    model: DenseTransformer | MoETransformer,
    settings: dict[str, Any],
    *,
    architecture: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run the configured short optimizer protocol before measuring inference."""
    data = settings["data"]
    benchmark = settings["benchmark"]
    train_tokens, _ = load_token_cache(
        token_path=data["train_tokens"], metadata_path=data["train_metadata"],
        tokenizer_path=data["tokenizer"], split="train", mmap=True,
    )
    validation_tokens, _ = load_token_cache(
        token_path=data["validation_tokens"], metadata_path=data["validation_metadata"],
        tokenizer_path=data["tokenizer"], split="validation", mmap=True,
    )
    batch_size = int(benchmark["batch_size"])
    sequence_length = int(benchmark["sequence_length"])
    accumulation = int(benchmark.get("gradient_accumulation_steps", 1))
    token_budget = int(benchmark["token_budget"])
    tokens_per_update = batch_size * sequence_length * accumulation
    if token_budget <= 0 or token_budget % tokens_per_update:
        raise ValueError("token_budget must be a positive multiple of effective batch tokens")
    optimizer_steps = token_budget // tokens_per_update
    train_batcher = TokenStreamBatcher(train_tokens, batch_size=batch_size, sequence_length=sequence_length, seed=int(benchmark["seed"]))
    validation_batcher = TokenStreamBatcher(validation_tokens, batch_size=batch_size, sequence_length=sequence_length, seed=int(benchmark["seed"]) + 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("optimizer", {}).get("learning_rate", 3e-4)),
        weight_decay=float(settings.get("optimizer", {}).get("weight_decay", 0.0)),
    )
    scheduler_settings = settings.get("scheduler", {}) or {}
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(scheduler_settings.get("warmup_steps", 0)),
        total_steps=optimizer_steps,
        min_lr_ratio=float(scheduler_settings.get("min_lr_ratio", 0.0)),
    )
    amp_enabled, amp_dtype, scaler_enabled = resolve_amp(settings, device)
    scaler = torch.amp.GradScaler(device=device.type, enabled=scaler_enabled)
    iterator = iter(train_batcher.iter_epoch(device=device))
    model.train()
    train_metrics: list[tuple[float, float, float]] = []
    for _ in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        sums = [0.0, 0.0, 0.0]
        for _ in range(accumulation):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_batcher.iter_epoch(device=device))
                inputs, targets = next(iterator)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                if architecture == "DenseTransformer":
                    logits, _ = model(inputs)
                    lm_loss = causal_loss(logits, targets)
                    aux_loss = None
                    total_loss = lm_loss
                else:
                    logits, _, aux_loss = model(inputs, collect_stats=False, capacity_factor=PREFILL_CAPACITY_FACTOR)
                    lm_loss = causal_loss(logits, targets)
                    total_loss = model.total_loss(lm_loss, aux_loss)
            sums[0] += float(lm_loss.detach())
            sums[1] += float(aux_loss.detach()) if aux_loss is not None else 0.0
            sums[2] += float(total_loss.detach())
            scaler.scale(total_loss / accumulation).backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_metrics.append(tuple(value / accumulation for value in sums))

    model.eval()
    validation_lm: list[float] = []
    validation_aux: list[float] = []
    with torch.no_grad():
        validation_batches = int(benchmark.get("validation_batches", 1))
        for index, (inputs, targets) in enumerate(validation_batcher.iter_epoch(device=device)):
            if index >= validation_batches:
                break
            if architecture == "DenseTransformer":
                logits, _ = model(inputs)
                lm_loss = causal_loss(logits, targets)
                aux_loss = None
            else:
                logits, _, aux_loss = model(inputs, collect_stats=False, capacity_factor=PREFILL_CAPACITY_FACTOR)
                lm_loss = causal_loss(logits, targets)
            validation_lm.append(float(lm_loss))
            if aux_loss is not None:
                validation_aux.append(float(aux_loss))
    if not validation_lm:
        raise ValueError("validation cache produced no benchmark batches")
    last_train = train_metrics[-1]
    aux_value = sum(validation_aux) / len(validation_aux) if validation_aux else None
    validation_lm_value = sum(validation_lm) / len(validation_lm)
    if aux_value is not None:
        total_value = validation_lm_value + model.moe_config.aux_loss_weight * aux_value
    else:
        total_value = validation_lm_value
    return {
        "train_lm_loss": last_train[0],
        "validation_lm_loss": sum(validation_lm) / len(validation_lm),
        "aux_loss": aux_value,
        "total_loss": total_value,
        "optimizer_steps": optimizer_steps,
        "effective_tokens": token_budget,
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def measure_model(model, input_ids: torch.Tensor, *, architecture: str, settings: dict[str, Any], device: torch.device) -> dict[str, Any]:
    warmups = int(settings["benchmark"]["warmup_steps"])
    measured = int(settings["benchmark"]["measured_steps"])
    batch, seq = input_ids.shape
    with torch.no_grad():
        for _ in range(warmups):
            if architecture == "DenseTransformer":
                model(input_ids)
            else:
                model(input_ids, collect_stats=False, capacity_factor=PREFILL_CAPACITY_FACTOR)
        synchronize(device)
        _memory_reset(device)
        start = time.perf_counter()
        for _ in range(measured):
            if architecture == "DenseTransformer":
                model(input_ids)
            else:
                model(input_ids, collect_stats=False, capacity_factor=PREFILL_CAPACITY_FACTOR)
        synchronize(device)
        prefill_seconds = (time.perf_counter() - start) / measured
        prefill_memory = _peak_memory_mb(device)

        cache = [dict() for _ in range(model.config.n_layers)]
        if architecture == "DenseTransformer":
            model(input_ids, kv_cache=cache)
        else:
            model(input_ids, kv_cache=cache, collect_stats=False, capacity_factor=PREFILL_CAPACITY_FACTOR)
        next_token = input_ids[:, -1:]
        decode_caches = [
            [dict(k=layer["k"].clone(), v=layer["v"].clone()) for layer in cache]
            for _ in range(measured)
        ]
        synchronize(device)
        _memory_reset(device)
        start = time.perf_counter()
        for decode_cache in decode_caches:
            if architecture == "DenseTransformer":
                model(next_token, start_pos=seq, kv_cache=decode_cache)
            else:
                model(next_token, start_pos=seq, kv_cache=decode_cache, collect_stats=False, capacity_factor=DECODE_CAPACITY_FACTOR)
        synchronize(device)
        decode_seconds = (time.perf_counter() - start) / measured
        decode_memory = _peak_memory_mb(device)

    return {
        "prefill": {
            "batch_size": batch,
            "input_tokens": seq,
            "latency_ms": prefill_seconds * 1000,
            "tokens_per_second": batch * seq / prefill_seconds,
            "peak_memory_mb": prefill_memory,
        },
        "decode": {
            "batch_size": batch,
            "context_tokens": seq,
            "new_tokens_per_step": 1,
            "latency_ms_per_token": decode_seconds * 1000,
            "tokens_per_second": batch / decode_seconds,
            "peak_memory_mb": decode_memory,
        },
    }


def cache_binding(settings: dict[str, Any], tokenizer: BPETokenizer) -> dict[str, Any]:
    data = settings["data"]
    metadata = __import__("json").loads(Path(data["train_metadata"]).read_text(encoding="utf-8"))
    # Validate both cache files at binding construction time so a result cannot
    # claim a shared validation cache that was never readable or hash-bound.
    for split in ("train", "validation"):
        load_token_cache(
            token_path=data[f"{split}_tokens"],
            metadata_path=data[f"{split}_metadata"],
            tokenizer_path=data["tokenizer"],
            split=split,
            mmap=True,
        )
    return {
        "data_version": metadata.get("data_version"),
        "tokenizer_sha256": sha256_file(data["tokenizer"]),
        "train_cache_metadata_sha256": sha256_file(data["train_metadata"]),
        "validation_cache_metadata_sha256": sha256_file(data["validation_metadata"]),
        "train_cache_sha256": sha256_file(data["train_tokens"]),
        "validation_cache_sha256": sha256_file(data["validation_tokens"]),
        "seed": int(settings["benchmark"]["seed"]),
        "batch_size": int(settings["benchmark"]["batch_size"]),
        "sequence_length": int(settings["benchmark"]["sequence_length"]),
        "token_budget": int(settings["benchmark"]["token_budget"]),
        "gradient_accumulation_steps": int(settings["benchmark"].get("gradient_accumulation_steps", 1)),
        "optimizer": settings.get("optimizer", {"name": "AdamW"}),
        "scheduler": settings.get("scheduler", {"name": "not_run"}),
        "amp": settings.get("amp", {"enabled": False}),
    }


def relative_error(value: int, target: int) -> float:
    return abs(value - target) / max(1, target)


def aggregate_routing(stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats:
        raise ValueError("routing stats cannot be empty")
    tokens = int(stats[0]["tokens"])
    assigned = [sum(layer["assigned_tokens_per_expert"][i] for layer in stats) for i in range(stats[0]["num_experts"])]
    kept = [sum(layer["kept_tokens_per_expert"][i] for layer in stats) for i in range(stats[0]["num_experts"])]
    dropped = sum(int(layer["dropped_tokens"]) for layer in stats)
    mean_assigned = sum(assigned) / len(assigned)
    return {
        "num_experts": stats[0]["num_experts"],
        "capacity_factor": stats[0]["capacity_factor"],
        "capacity": stats[0]["capacity"],
        "tokens": tokens,
        "assigned_tokens_per_expert": assigned,
        "kept_tokens_per_expert": kept,
        "dropped_tokens": dropped,
        "dropped_token_ratio": dropped / max(1, tokens),
        "load_imbalance": max(assigned) / max(1e-12, mean_assigned),
        "per_layer": stats,
    }
