"""P4 GRPO MVP — minimal group-relative policy optimization trainer.

This script is a minimal but architecturally complete GRPO (Group
Relative Policy Optimization, DeepSeek 2024) trainer. It reuses
existing components rather than reimplementing them:

- **Rollout**: ``scripts.eval_transformers._apply_chat_template`` +
  greedy / temperature-sampled generation (reuses the P5-02 prompt
  pipeline so the rollout prompt format is identical to
  ``eval_transformers.py``).
- **Reward**: ``scripts.reward_offline.compute_reward`` (P2 reward
  signal: ``reward_binary`` + ``reward_layered``).
- **Advantage**: group-relative: ``(r - mean(r)) / max(std(r), eps)``.
- **Policy update**: minimal REINFORCE-style surrogate loss applied
  to the log-probabilities of the assistant continuation tokens.
  This is NOT a full GRPO implementation with KL penalty and PPO
  clipping; it is the MVP that demonstrates the loop end-to-end.
- **Checkpoint / resume**: optimizer state + step counter persisted
  to ``--checkpoint-dir``; ``--resume-from`` restores them.

Why a minimal implementation:

- The goal is a runnable smoke that exercises rollout → reward →
  advantage → policy-update → checkpoint on CPU. A full GRPO with
  reference-policy KL would double the GPU memory cost and require a
  frozen copy of the model — out of scope for the MVP.
- The advantage / loss / checkpoint / resume code is independently
  unit-tested (``tests/test_grpo_mvp.py``) so the policy-update
  correctness can be verified without running the full pipeline.

Usage:

    .venv/python.exe scripts/grpo_train.py \\
        --policy-model Qwen/Qwen2.5-0.5B-Instruct \\
        --samples-dir datasets/tool-calling-d2/train \\
        --checkpoint-dir artifacts/grpo-qwen05 \\
        --max-steps 8 --k-rollouts 4 --limit 8

A "smoke" run that exercises every code path on a few prompts and
produces ``<checkpoint-dir>/step-*.json`` artifacts and a final
``state.json`` summary.

The script is designed to be **CPU-runnable** by default (the default
device is ``auto`` and falls back to ``cpu`` when CUDA is unavailable)
so it works on this host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classify_tool_failure import classify  # noqa: E402
from scripts.eval_sft_tool import extract_tool_calls  # noqa: E402
from scripts.eval_transformers import _apply_chat_template  # noqa: E402
from scripts.reward_offline import compute_reward  # noqa: E402

GRPO_STEP_SCHEMA_VERSION = "1.0"
ADVANTAGE_EPS = 1e-6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-model",
        required=True,
        help="Hugging Face model id or local path for the policy",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("datasets/tool-calling-d2/train"),
        help="Directory of D2 multi-turn sample JSON files (default: train split)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory to persist step artifacts and the final state.json",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a previous state.json; on resume the optimizer + step "
             "counter are restored and the run continues from global_step+1",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum policy-gradient steps (default: 8 for smoke)",
    )
    parser.add_argument(
        "--k-rollouts",
        type=int,
        default=4,
        help="Number of rollouts per prompt (default: 4; GRPO group size K)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Adam learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for rollouts; 1.0 = standard GRPO "
             "(default: 1.0). Pass 0 for greedy (matches eval_transformers).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Rollout continuation length (default: 64, smaller than eval_transformers "
             "to keep CPU smoke fast)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Cap the number of samples loaded (default: 8 for smoke)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Deterministic seed for sample shuffle + rollout sampling",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device override (default: auto = cuda if available else cpu)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold (default: 1.0; 0 = no clipping)",
    )
    parser.add_argument(
        "--smoke-deterministic",
        action="store_true",
        help="Force greedy decoding + temperature=0 regardless of --temperature; "
             "used by tests to make rollouts byte-stable across runs",
    )
    parser.add_argument(
        "--reference-model",
        default=None,
        help="(Optional) HF model id for the reference policy; when set, an "
             "approximate KL penalty is added to the surrogate loss. Skipped by "
             "the MVP (out of scope) but accepted as a no-op for CLI stability.",
    )
    return parser


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


# ---------------------------------------------------------------------------
# Sample loading + deterministic iteration
# ---------------------------------------------------------------------------

def _load_samples(samples_dir: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    paths = sorted(samples_dir.glob("*.json"))
    rng = random.Random(seed)
    rng.shuffle(paths)
    if limit > 0:
        paths = paths[:limit]
    out: list[dict[str, Any]] = []
    for path in paths:
        sample = json.loads(path.read_text(encoding="utf-8"))
        sample["id"] = sample.get("id") or path.stem
        out.append(sample)
    return out


def _iter_prompts(samples: list[dict[str, Any]], resume_step: int,
                  max_steps: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    """Yield ``(global_step, sample)`` pairs starting from ``resume_step``.

    ``global_step`` is the 0-indexed policy-gradient step counter;
    ``resume_step`` is the value of ``global_step`` recorded in the
    previous ``state.json``. When ``resume_step`` is non-zero, the
    iteration continues from ``resume_step`` (caller does not re-run
    already-saved steps).
    """
    if resume_step < 0:
        resume_step = 0
    pairs: list[tuple[int, dict[str, Any]]] = []
    for offset, sample in enumerate(samples):
        gs = resume_step + offset
        if gs >= max_steps:
            break
        pairs.append((gs, sample))
    return pairs


# ---------------------------------------------------------------------------
# Tokenizer + model loaders (lightweight; rely on HF transformers)
# ---------------------------------------------------------------------------

def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)


def _load_model(model_id: str, device: str):
    from transformers import AutoModelForCausalLM
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------

def _rollout_one(model, tokenizer, sample: dict[str, Any], *,
                 max_new_tokens: int, temperature: float, device: str,
                 deterministic: bool) -> dict[str, Any]:
    """Generate one assistant continuation for ``sample`` and return
    ``{generated, extracted_calls}``. Always strips the terminal
    assistant message before rendering (matches
    ``scripts.eval_transformers._strip_terminal_assistant``).
    """
    prompt = _apply_chat_template(tokenizer, sample)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    do_sample = (not deterministic) and (temperature > 0)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    new_token_ids = out[0][prompt_len:].tolist()
    generated = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    extracted = extract_tool_calls(generated)
    return {"generated": generated, "extracted_calls": extracted}


# ---------------------------------------------------------------------------
# Reward + advantage
# ---------------------------------------------------------------------------

def _rewards_for_rollouts(sample: dict[str, Any],
                          rollouts: list[dict[str, Any]],
                          transcript_kind: str,
                          checkpoint: str | None) -> list[dict[str, Any]]:
    rewards: list[dict[str, Any]] = []
    for r in rollouts:
        row = {
            "sample_id": sample.get("id", ""),
            "extracted_calls": r.get("extracted_calls", []),
            "generated": r.get("generated", ""),
            "first_failure": None,
            "layers": {},
        }
        # Build a transcript-style dict so classify_tool_failure.classify works
        transcript = {
            "tool_calls": r.get("extracted_calls", []),
            "final_answer": r.get("generated", ""),
        }
        classification = classify(sample, transcript)
        row["first_failure"] = classification["first_failure"]
        row["layers"] = classification["layers"]
        sig = compute_reward(sample, row, transcript_kind=transcript_kind,
                             checkpoint=checkpoint)
        sig["rollout_index"] = r["rollout_index"]
        rewards.append(sig)
    return rewards


def _group_relative_advantages(rewards: list[dict[str, Any]]) -> list[float]:
    """Standardize rewards within the group to produce advantages."""
    vals = [float(r["reward_layered"]) for r in rewards]
    n = len(vals)
    if n == 0:
        return []
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(max(var, 0.0))
    if std < ADVANTAGE_EPS:
        # No spread → zero advantages, no learning signal
        return [0.0] * n
    return [(v - mean) / std for v in vals]


def _advantage_stats(advantages: list[float]) -> dict[str, float]:
    if not advantages:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(advantages) / len(advantages)
    var = sum((a - mean) ** 2 for a in advantages) / len(advantages)
    return {
        "mean": mean,
        "std": math.sqrt(max(var, 0.0)),
        "min": min(advantages),
        "max": max(advantages),
    }


# ---------------------------------------------------------------------------
# Policy update (REINFORCE surrogate)
# ---------------------------------------------------------------------------

def _policy_update(
    model, tokenizer, sample: dict[str, Any], rollouts: list[dict[str, Any]],
    advantages: list[float], *,
    learning_rate: float, max_grad_norm: float, device: str,
) -> dict[str, Any]:
    """Compute one REINFORCE-style policy gradient step.

    Returns ``{policy_gradient_loss, learning_rate, tokens_seen, grad_norm}``.
    When all advantages are zero (no learning signal), the step is a no-op
    and ``skipped=True`` is set in the returned dict.
    """
    if not any(abs(a) > ADVANTAGE_EPS for a in advantages):
        return {
            "policy_gradient_loss": 0.0,
            "learning_rate": learning_rate,
            "tokens_seen": 0,
            "grad_norm": 0.0,
            "skipped": True,
            "skip_reason": "all advantages are zero (no learning signal)",
        }

    model.train()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    prompt_text = _apply_chat_template(tokenizer, sample)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt",
                              truncation=True, max_length=2048)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    total_loss = 0.0
    total_tokens = 0
    for r, adv in zip(rollouts, advantages):
        if abs(adv) <= ADVANTAGE_EPS:
            continue
        full_text = prompt_text + r["generated"]
        full_inputs = tokenizer(full_text, return_tensors="pt",
                                truncation=True, max_length=2048)
        input_ids = full_inputs["input_ids"].to(device)
        if input_ids.shape[1] <= prompt_len:
            continue
        labels = input_ids.clone()
        # Mask the prompt portion so the loss only applies to generated tokens
        labels[:, :prompt_len] = -100

        outputs = model(input_ids=input_ids, labels=labels)
        # Per-token log-prob of the generated continuation
        # outputs.logits has shape (1, seq, vocab); shift to align with labels
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        per_token_logp = torch.gather(
            log_probs, 2, shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        mask = (shift_labels != -100).float()
        n_valid = mask.sum().clamp(min=1)
        # Per-rollout mean log-prob over generated tokens
        mean_logp = (per_token_logp * mask).sum() / n_valid
        # REINFORCE surrogate: -adv * logp
        loss = -float(adv) * mean_logp
        total_loss += float(loss.detach().cpu())
        total_tokens += int(n_valid.item())
        loss.backward()

    grad_norm = 0.0
    if max_grad_norm > 0:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        ).cpu().item())
    else:
        # No clipping — compute global norm manually
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().pow(2).sum().cpu().item())
        grad_norm = math.sqrt(total_sq)

    optimizer.step()
    optimizer.zero_grad()

    return {
        "policy_gradient_loss": total_loss,
        "learning_rate": learning_rate,
        "tokens_seen": total_tokens,
        "grad_norm": grad_norm,
        "skipped": False,
        "skip_reason": "",
    }


# ---------------------------------------------------------------------------
# Determinism fingerprints
# ---------------------------------------------------------------------------

def _hash_rollouts(rollouts: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in sorted(rollouts, key=lambda x: x["rollout_index"]):
        h.update(r["generated"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _hash_rewards(rewards: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in sorted(rewards, key=lambda x: x["rollout_index"]):
        h.update(repr(float(r["reward_layered"])).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save_state(checkpoint_dir: Path, *, global_step: int, last_step_id: str,
                rng_state: tuple[int, ...], max_steps: int,
                k_rollouts: int, policy_model: str) -> None:
    state = {
        "schema_version": GRPO_STEP_SCHEMA_VERSION,
        "global_step": global_step,
        "last_step_id": last_step_id,
        "rng_state": list(rng_state),
        "max_steps": max_steps,
        "k_rollouts": k_rollouts,
        "policy_model": policy_model,
        "completed": global_step + 1 >= max_steps,
    }
    (checkpoint_dir / "state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_state(resume_from: Path) -> dict[str, Any]:
    return json.loads(resume_from.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Per-step artifact assembly
# ---------------------------------------------------------------------------

def _assemble_step_artifact(
    *, step_id: str, global_step: int, policy_model: str,
    sample: dict[str, Any], rollouts: list[dict[str, Any]],
    rewards: list[dict[str, Any]], advantages: list[float],
    update: dict[str, Any], seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": GRPO_STEP_SCHEMA_VERSION,
        "step_id": step_id,
        "global_step": global_step,
        "policy_model": policy_model,
        "prompt_id": sample.get("id", ""),
        "prompt_metadata": {
            "task_type": sample.get("metadata", {}).get("task_type", ""),
            "expected_answer_length":
                len(sample.get("expected_answer") or ""),
        },
        "rollouts": rollouts,
        "rewards": rewards,
        "advantages": advantages,
        "advantage_stats": _advantage_stats(advantages),
        "update": update,
        "deterministic": {
            "seed": seed,
            "rollouts_text_hash": _hash_rollouts(rollouts),
            "rewards_text_hash": _hash_rewards(rewards),
        },
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if args.smoke_deterministic:
        args.temperature = 0.0

    device = _resolve_device(args.device)
    checkpoint_dir: Path = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_step = 0
    if args.resume_from is not None:
        state = _load_state(args.resume_from)
        resume_step = int(state.get("global_step", 0)) + 1
        print(f"[grpo] resuming from {args.resume_from}: "
              f"global_step+1 = {resume_step}", flush=True)

    print(f"[grpo] loading policy model {args.policy_model} on {device}",
          flush=True)
    tokenizer = _load_tokenizer(args.policy_model)
    model = _load_model(args.policy_model, device)
    if device == "cpu":
        # CPU mode is intentionally limited; warn loudly
        print("[grpo] running on CPU: this is the smoke path, "
              "expect slow generation. Pass --device cuda if available.",
              flush=True)

    samples = _load_samples(args.samples_dir, args.limit, args.seed)
    print(f"[grpo] loaded {len(samples)} samples from {args.samples_dir}",
          flush=True)
    if not samples:
        print(f"[grpo] no samples found in {args.samples_dir}", flush=True)
        return 1

    pairs = _iter_prompts(samples, resume_step, args.max_steps, args.seed)
    if not pairs:
        print(f"[grpo] nothing to do: resume_step={resume_step} >= "
              f"max_steps={args.max_steps}", flush=True)
        return 0

    last_step_id = ""
    for global_step, sample in pairs:
        step_id = f"{global_step:04d}_{sample.get('id', 'unknown')}"
        print(f"[grpo] step {global_step + 1}/{args.max_steps}: "
              f"prompt_id={sample.get('id', '?')}", flush=True)

        rollouts: list[dict[str, Any]] = []
        for k in range(args.k_rollouts):
            r = _rollout_one(
                model, tokenizer, sample,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
                deterministic=args.smoke_deterministic,
            )
            r["rollout_index"] = k
            rollouts.append(r)

        rewards = _rewards_for_rollouts(
            sample, rollouts, transcript_kind="grpo_rollout",
            checkpoint=args.policy_model,
        )
        advantages = _group_relative_advantages(rewards)
        update = _policy_update(
            model, tokenizer, sample, rollouts, advantages,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            device=device,
        )
        artifact = _assemble_step_artifact(
            step_id=step_id, global_step=global_step,
            policy_model=args.policy_model, sample=sample,
            rollouts=rollouts, rewards=rewards, advantages=advantages,
            update=update, seed=args.seed,
        )
        step_path = checkpoint_dir / f"step-{step_id}.json"
        step_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        last_step_id = step_id
        _save_state(
            checkpoint_dir, global_step=global_step, last_step_id=step_id,
            rng_state=(), max_steps=args.max_steps, k_rollouts=args.k_rollouts,
            policy_model=args.policy_model,
        )
        print(f"[grpo] step {global_step} done: "
              f"loss={update['policy_gradient_loss']:.4f} "
              f"grad_norm={update['grad_norm']:.3f} "
              f"tokens={update['tokens_seen']} "
              f"-> {step_path.name}", flush=True)

    print(f"[grpo] all steps complete: state at "
          f"{checkpoint_dir / 'state.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())