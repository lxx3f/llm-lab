"""SFT training for tool-calling (D1/D1.1 datasets).

Turns tool_calling_sample JSON documents into masked-language-model training
sequences and trains a DenseTransformer to emit tool calls + final answers.

Sequence template (masked on the assistant part only):

    <|endoftext|>### User
    {user_turn}
    ### Assistant
    {plan_json}
    ### Result
    {result_summary}
    {final_answer}<|endoftext|>

Loss is computed ONLY over the assistant span (plan + result + answer),
matching how a chat/tool model is actually scored. The user turn and
separator tokens contribute no gradient.

The plan JSON is rendered deterministically from ``expected_tool_calls`` so
the model learns a canonical tool-call serialization.

Usage (via scripts/train_sft.py):
    .venv/python.exe scripts/train_sft.py --config configs/sft.example.yaml
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor

from architecture_lab.models.dense_transformer import DenseTransformer, TransformerConfig
from architecture_lab.tokenization import BPETokenizer
from architecture_lab.training.dense_training import (
    build_model_config,
    build_scheduler,
    causal_loss,
    resolve_device,
    resolve_dtype,
    set_seed,
)

USER_SEP = "### User\n"
ASSISTANT_SEP = "### Assistant\n"
RESULT_SEP = "### Result\n"


@dataclass(frozen=True)
class SftExample:
    input_ids: list[int]
    target_ids: list[int]
    mask: list[int]  # 1 = predict (assistant span), 0 = ignore


def render_plan_json(expected_calls: Sequence[dict[str, Any]]) -> str:
    """Deterministic canonical serialization of the expected tool-call plan.

    Each call is a JSON object starting with ``{"call_id"...``; multiple
    calls are separated by newlines (NO enclosing array). This matches the
    tokenizer's merged ``{"`` token so the assistant span starts with the
    same token the model learns to emit first (the array bracket ``[`` is a
    rare token that destabilized generation).
    """
    plan = []
    for call in expected_calls:
        entry = {
            "call_id": call.get("call_id", ""),
            "name": call.get("name", ""),
            "arguments": call.get("arguments", {}),
        }
        if call.get("depends_on"):
            entry["depends_on"] = call["depends_on"]
        plan.append(json.dumps(entry, ensure_ascii=False, sort_keys=False))
    return "\n".join(plan)


def render_result_summary(expected_calls: Sequence[dict[str, Any]]) -> str:
    """Render the tool results the model should ground its answer on."""
    if not expected_calls:
        return "（无工具调用）"
    parts = []
    for call in expected_calls:
        result = call.get("expected_result")
        parts.append(f"{call.get('name', '')}: {json.dumps(result, ensure_ascii=False)}")
    return "；".join(parts)


def _find_user_turn(messages: Sequence[dict[str, Any]]) -> str:
    """Extract the last user message content.

    Samples may carry a leading system message (D1 template set) whose text
    must NOT be treated as the user turn.
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError("sample has no non-empty user message")


def build_sft_example(
    sample: dict[str, Any], tokenizer: BPETokenizer, *, max_len: int
) -> SftExample:
    """Build one masked SFT example from a tool_calling_sample document."""
    user_turn = _find_user_turn(sample["messages"])
    expected_calls = sample.get("expected_tool_calls", [])
    plan_json = render_plan_json(expected_calls)
    result_summary = render_result_summary(expected_calls)
    final_answer = sample.get("expected_answer") or "（无答案）"

    prefix = USER_SEP + user_turn + "\n" + ASSISTANT_SEP
    assistant_body = plan_json + "\n" + RESULT_SEP + result_summary + "\n" + final_answer

    # Encode and truncate the FULL sequence to max_len. Truncation happens
    # at the right edge (drops the tail of the final answer, never the user
    # intent at the start). The mask still covers whatever assistant span
    # survives.
    prefix_ids = tokenizer.encode(prefix)
    body_ids = tokenizer.encode(assistant_body)
    eot_id = tokenizer.encode("<|endoftext|>")[0]
    ids = prefix_ids + body_ids + [eot_id]
    if len(ids) > max_len:
        ids = ids[:max_len]
    # Mask: assistant span starts at len(prefix_ids); EOS at the end.
    prefix_len = min(len(prefix_ids), len(ids))
    mask = [0] * prefix_len + [1] * (len(ids) - prefix_len)
    # Targets are the next-token ids; the last position predicts EOS.
    target_ids = ids[1:] + [eot_id]
    return SftExample(input_ids=ids, target_ids=target_ids, mask=mask)


def build_sft_examples(
    samples: Sequence[dict[str, Any]], tokenizer: BPETokenizer, *, max_len: int
) -> list[SftExample]:
    return [build_sft_example(s, tokenizer, max_len=max_len) for s in samples]


class SFTBatcher:
    """Yields (input_ids, target_ids, mask) batches from padded SFT examples.

    Each batch is padded to the longest sequence in the batch (dynamic
    padding), and examples are shuffled deterministically per epoch.
    """

    def __init__(self, examples: list[SftExample], *, batch_size: int, seed: int) -> None:
        # Drop degenerate examples whose assistant span was fully truncated
        # (they carry no training signal and would waste a batch slot).
        self.examples = [e for e in examples if sum(e.mask) > 0]
        if not self.examples:
            raise ValueError("no SFT examples with a non-empty assistant span")
        self.batch_size = batch_size
        self.seed = seed

    def __len__(self) -> int:
        return math.ceil(len(self.examples) / self.batch_size)

    def iter_epoch(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        rng = random.Random(self.seed)
        order = list(range(len(self.examples)))
        rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            batch_idx = order[start : start + self.batch_size]
            batch = [self.examples[i] for i in batch_idx]
            max_len = max(len(e.input_ids) for e in batch)
            input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
            target_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
            mask = torch.zeros(len(batch), max_len, dtype=torch.long)
            for row, ex in enumerate(batch):
                seq_len = len(ex.input_ids)
                input_ids[row, :seq_len] = torch.tensor(ex.input_ids, dtype=torch.long)
                target_ids[row, :seq_len] = torch.tensor(ex.target_ids[:seq_len], dtype=torch.long)
                mask[row, :seq_len] = torch.tensor(ex.mask[:seq_len], dtype=torch.long)
            yield input_ids, target_ids, mask


def masked_causal_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Cross-entropy over masked (assistant-only) positions.

    ``mask`` is 1 where the model should predict the next token (assistant
    span). Padded positions and the user/separator span contribute nothing.
    """
    if logits.ndim != 3 or targets.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            "logits must be [batch, time, vocab], targets/mask [batch, time]"
        )
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1).bool()
    if mask_flat.sum() == 0:
        return torch.zeros((), device=logits.device, requires_grad=True)
    ce = torch.nn.functional.cross_entropy(
        logits_flat[mask_flat], targets_flat[mask_flat], reduction="mean"
    )
    return ce


def train_sft(
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
    """Train DenseTransformer on tool-calling SFT examples.

    Returns a metrics dict mirroring the Dense training curve schema:
    ``{"metrics": {"train_losses": [...], "curve_summary": {...}}, ...}``.
    """
    set_seed(int(training.get("seed", 42)))
    tokenizer = BPETokenizer.load(tokenizer_path)
    max_len = int(training.get("sequence_length", 256))
    batch_size = int(training.get("batch_size", 8))
    max_steps = int(training["max_steps"])
    lr = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.0))
    log_interval = int(training.get("log_interval", max_steps // 20 or 1))
    gradient_accumulation_steps = int(training.get("gradient_accumulation_steps", 1))

    examples = build_sft_examples(samples, tokenizer, max_len=max_len)
    if not examples:
        raise ValueError("no SFT examples built from samples")
    rng = random.Random(int(training.get("seed", 42)))
    # Simple train/validation split: last 5% held out.
    n_val = max(1, int(len(examples) * 0.05))
    val_examples = examples[-n_val:]
    train_examples = examples[:-n_val]

    model_config = build_model_config({"model": model_settings}, tokenizer.vocab_size)
    model = DenseTransformer(model_config).to(device=device, dtype=dtype)
    if init_checkpoint is not None:
        if not init_checkpoint.is_file():
            raise ValueError(f"init checkpoint not found: {init_checkpoint}")
        ckpt = torch.load(init_checkpoint, map_location="cpu")
        model_state = ckpt["model_state"]
        # RoPE position embedding is parameter-free (computed from
        # rope_base), so loading a max_seq_len=64 pre-trained state into a
        # max_seq_len=256 model is safe: only token/weight matrices load.
        model.load_state_dict(model_state, strict=False)
        loaded = sum(1 for _ in model_state)
        print(f"[sft] initialized from {init_checkpoint} ({loaded} tensors)",
              flush=True)
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
                logits, _ = model(input_ids)
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
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, _ = model(input_ids)
                loss = masked_causal_loss(logits, targets, mask) / gradient_accumulation_steps
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
                    "lr": float(lr_now),
                })
                print(f"[sft] step={step} train_loss={float(loss):.4f} "
                      f"val_loss={val_loss:.4f} lr={lr_now:.6f}", flush=True)

    # Persist checkpoint.
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_state": {"step": step, "epoch": 1},
    }, checkpoint_out)

    # Build the curve summary (mirror Dense schema).
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
        "training": {
            "optimizer_steps": step,
            "seed": int(training.get("seed", 42)),
        },
    }
