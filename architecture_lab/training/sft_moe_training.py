"""SFT training for tool-calling on a Top-1 MoE Transformer.

Mirrors ``sft_training.py`` but uses ``MoETransformer`` instead of
``DenseTransformer``. The data pipeline (``build_sft_examples``,
``SFTBatcher``, ``masked_causal_loss``) is identical and re-imported
from the Dense module — only the model wiring and the aux-loss term
differ.

The MoE forward returns ``(logits, labels_loss, aux_loss)``; SFT adds
aux_loss to the masked cross-entropy weighted by ``moe_aux_loss_coeff``
(default 0.01, matching the existing ``train_moe`` setting).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from architecture_lab.models.moe_transformer import MoEConfig, MoETransformer
from architecture_lab.models.dense_transformer import TransformerConfig
from architecture_lab.tokenization import BPETokenizer
from architecture_lab.training.dense_training import (
    build_model_config,
    build_scheduler,
    resolve_device,
    resolve_dtype,
    set_seed,
)
# Reuse the entire data pipeline from the Dense SFT module — the templates,
# masking, and rendering are model-agnostic.
from architecture_lab.training.sft_training import (
    SftExample,
    SFTBatcher,
    build_sft_examples,
    masked_causal_loss,
)


def build_moe_model(
    model_settings: dict[str, Any], vocab_size: int
) -> MoETransformer:
    """Construct an MoE Transformer from a settings dict.

    ``model_settings`` carries the same keys as a Dense SFT config plus
    ``moe_num_experts`` (default 4) and ``moe_aux_loss_coeff`` (default
    0.01). ``build_model_config`` returns a ``TransformerConfig``; we
    wrap it in an MoE config with the requested expert count.
    """
    transformer_config = build_model_config({"model": model_settings}, vocab_size)
    moe_config = MoEConfig(
        num_experts=int(model_settings.get("moe_num_experts", 4)),
        aux_loss_weight=float(model_settings.get("moe_aux_loss_coeff",
                                                  model_settings.get("moe_aux_loss_weight", 0.01))),
    )
    return MoETransformer(transformer_config, moe_config)


def train_sft_moe(
    *,
    samples: list[dict[str, Any]],
    tokenizer_path: Path,
    model_settings: dict[str, Any],
    training: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    checkpoint_out: Path,
    init_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Train MoETransformer on tool-calling SFT examples."""
    set_seed(int(training.get("seed", 42)))
    tokenizer = BPETokenizer.load(tokenizer_path)
    max_len = int(training.get("sequence_length", 256))
    batch_size = int(training.get("batch_size", 8))
    max_steps = int(training["max_steps"])
    lr = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.0))
    log_interval = int(training.get("log_interval", max_steps // 20 or 1))
    gradient_accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    moe_aux_loss_coeff = float(model_settings.get("moe_aux_loss_coeff",
                                                   model_settings.get("moe_aux_loss_weight", 0.01)))

    examples = build_sft_examples(samples, tokenizer, max_len=max_len)
    rng = random.Random(int(training.get("seed", 42)))
    n_val = max(1, int(len(examples) * 0.05))
    val_examples = examples[-n_val:]
    train_examples = examples[:-n_val]

    model = build_moe_model(model_settings, tokenizer.vocab_size).to(
        device=device, dtype=dtype
    )
    if init_checkpoint is not None:
        if not init_checkpoint.is_file():
            raise ValueError(f"init checkpoint not found: {init_checkpoint}")
        ckpt = torch.load(init_checkpoint, map_location="cpu")
        # MoE checkpoint stores {"model_state": ...}; we accept either that
        # or a raw state_dict.
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[sft-moe] initialized from {init_checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler_config = training.get("scheduler", {}) or {}
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(scheduler_config.get("warmup_steps", 100)),
        total_steps=max_steps,
        min_lr_ratio=float(scheduler_config.get("min_lr_ratio", 0.1)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    train_batcher = SFTBatcher(train_examples, batch_size=batch_size,
                               seed=int(training.get("seed", 42)))
    val_batcher = SFTBatcher(val_examples, batch_size=batch_size,
                             seed=int(training.get("seed", 42)) + 1)

    def run_eval() -> float:
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for input_ids, targets, mask in val_batcher.iter_epoch():
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                logits, _ce, aux = model(input_ids, labels=input_ids)
                losses.append(float(masked_causal_loss(logits, targets, mask)))
        model.train()
        return sum(losses) / len(losses) if losses else float("nan")

    model.train()
    step = 0
    train_losses: list[dict[str, Any]] = []
    validation_losses: dict[str, float] = {}
    optimizer.zero_grad()
    while step < max_steps:
        for input_ids, targets, mask in train_batcher.iter_epoch():
            if step >= max_steps:
                break
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits, _ce, aux = model(input_ids, labels=input_ids)
                ce = masked_causal_loss(logits, targets, mask)
                loss = (ce + moe_aux_loss_coeff * aux) / gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            step += 1
            if step % log_interval == 0 or step == max_steps:
                val_loss = run_eval()
                validation_losses[str(step)] = val_loss
                lr_now = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else lr
                train_losses.append({
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "ce_loss": float(ce.detach().cpu()),
                    "aux_loss": float(aux.detach().cpu()),
                    "lr": float(lr_now),
                })
                print(
                    f"[sft-moe] step={step} train_loss={float(loss):.4f} "
                    f"ce={float(ce):.4f} aux={float(aux):.4f} "
                    f"val_loss={val_loss:.4f} lr={lr_now:.6f}",
                    flush=True,
                )

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    transformer_config = model.config
    moe_config = model.moe_config
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_state": {"step": step, "epoch": 1},
        "model_config": {
            "vocab_size": transformer_config.vocab_size,
            "max_seq_len": transformer_config.max_seq_len,
            "d_model": transformer_config.d_model,
            "n_heads": transformer_config.n_heads,
            "n_layers": transformer_config.n_layers,
            "d_ff": transformer_config.d_ff,
            "dropout": transformer_config.dropout,
            "rope_base": transformer_config.rope_base,
        },
        "moe_config": {
            "num_experts": moe_config.num_experts,
            "aux_loss_weight": moe_config.aux_loss_weight,
        },
        "architecture": "MoETransformer",
    }, checkpoint_out)

    def _summary(losses: list[dict[str, Any]], val_map: dict[str, float]) -> dict[str, Any]:
        first = losses[0]["loss"] if losses else None
        last = losses[-1]["loss"] if losses else None
        val_items = [(int(k), v) for k, v in val_map.items() if v == v]
        val_min = min(val_items, key=lambda kv: kv[1])[1] if val_items else None
        val_min_step = min(val_items, key=lambda kv: kv[1])[0] if val_items else None
        val_last = val_items[-1][1] if val_items else None
        return {
            "train_loss_first": first,
            "train_loss_last": last,
            "val_loss_min": val_min,
            "val_loss_min_step": val_min_step,
            "val_loss_last": val_last,
            "delta_train_loss": (last - first) if first is not None and last is not None else None,
            "delta_val_loss": (val_last - val_min) if val_min is not None and val_last is not None else None,
            "train_loss_sample_count": len(losses),
            "val_loss_count": len(val_items),
        }

    return {
        "metrics": {
            "last_train_loss": train_losses[-1]["loss"] if train_losses else None,
            "validation_losses": validation_losses,
            "train_losses": train_losses,
            "curve_summary": _summary(train_losses, validation_losses),
        },
        "training": {"optimizer_steps": step, "seed": int(training.get("seed", 42))},
    }