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
- **Checkpoint / resume**: model state_dict + optimizer state_dict +
  Python/Torch/CUDA RNG state + sample cursor + step counter all
  persisted under ``--checkpoint-dir``; ``--resume-from`` restores
  all of them so a resumed run continues the prior trajectory
  exactly.

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
``state.json`` + ``state.pt`` (binary model + optimizer + RNG state).

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
import pickle  # nosec — RNG state is internal, not user-controlled
import random
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
# state.pt layout: {model_state, optimizer_state, rng_state} where
# rng_state is a dict of {python, torch, torch_cuda} seed/state pairs.
# Versioned so future schema changes can be rejected gracefully.
STATE_PT_VERSION = "1.0"


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
        help="Directory to persist step artifacts and the final state.json + state.pt",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a previous state.json; on resume the model + optimizer + "
             "RNG + sample cursor are restored from <state_dir>/state.pt and "
             "the run continues from samples_consumed onwards",
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
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Persist state.pt every N steps (default: 1 = every step)",
    )
    parser.add_argument(
        "--strict-resume-config",
        action="store_true",
        help="When set, --resume-from refuses to continue if the saved run "
             "config (model_id, k_rollouts, max_steps, learning_rate, "
             "max_grad_norm, samples_dir, seed, device) differs from the "
             "current CLI args. Off by default (debugging-friendly).",
    )
    return parser


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def _seed_all(seed: int, device: str) -> None:
    """Seed Python, Torch (CPU + CUDA) RNGs for reproducible rollouts."""
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state(device: str) -> dict[str, Any]:
    """Capture the current RNG state from Python, Torch, and CUDA."""
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore Python / Torch / CUDA RNGs from a previously captured dict."""
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


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


def _iter_prompts_with_cursor(
    samples: list[dict[str, Any]],
    cursor: int,
    max_steps: int,
    remaining_steps: int,
) -> list[tuple[int, int, dict[str, Any]]]:
    """Resume-aware iteration. Returns ``(global_step, samples_offset, sample)``
    triples starting at ``samples[cursor]``.

    On a fresh run ``cursor=0, remaining_steps=max_steps``; on resume
    ``cursor`` is the number of distinct samples already consumed in
    the previous run, and ``remaining_steps`` is the count of policy-
    gradient steps still to run (``max_steps - next_global_step``).

    When ``cursor >= len(samples)`` and ``remaining_steps > 0``, the
    cursor wraps to 0 so the run continues instead of doing nothing.

    The ``samples_offset`` is recorded in the state so the next
    resume picks up correctly.
    """
    if cursor < 0:
        cursor = 0
    if remaining_steps <= 0:
        return []
    if cursor >= len(samples):
        if len(samples) == 0:
            return []
        cursor = 0
    end = min(cursor + remaining_steps, len(samples))
    scheduled = samples[cursor:end]
    return [(0, cursor + idx, s) for idx, s in enumerate(scheduled)]


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
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
    """Compute one REINFORCE-style policy gradient step.

    When ``optimizer`` is None, a fresh Adam is created (first step
    of a fresh run). When ``optimizer`` is provided, it is reused
    (resume path — caller has loaded ``state_dict`` from
    ``state.pt``). The (possibly new) optimizer is always returned so
    the caller can persist it next.

    Returns ``({policy_gradient_loss, learning_rate, tokens_seen,
    grad_norm}, optimizer)``. When all advantages are zero (no
    learning signal), the step is a no-op and ``skipped=True``.
    """
    if not any(abs(a) > ADVANTAGE_EPS for a in advantages):
        return ({
            "policy_gradient_loss": 0.0,
            "learning_rate": learning_rate,
            "tokens_seen": 0,
            "grad_norm": 0.0,
            "skipped": True,
            "skip_reason": "all advantages are zero (no learning signal)",
        }, optimizer if optimizer is not None else _fresh_optimizer(
            model, learning_rate))

    model.train()
    if optimizer is None:
        optimizer = _fresh_optimizer(model, learning_rate)

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

    return ({
        "policy_gradient_loss": total_loss,
        "learning_rate": learning_rate,
        "tokens_seen": total_tokens,
        "grad_norm": grad_norm,
        "skipped": False,
        "skip_reason": "",
    }, optimizer)


def _fresh_optimizer(model, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )


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
                samples_consumed: int, max_steps: int,
                k_rollouts: int, policy_model: str,
                seed: int, learning_rate: float, max_grad_norm: float,
                samples_dir: str, device: str, save_every: int,
                model, optimizer, rng_state: dict[str, Any],
                next_global_step_to_save: int | None = None) -> None:
    """Persist run-level state to ``state.json`` and binary state to
    ``state.pt``. The binary blob contains ``{model_state_dict,
    optimizer_state_dict, rng_state}``.

    ``samples_consumed`` is the number of distinct prompts already
    consumed; on resume the next run starts at ``samples[cursor]``.
    """
    state = {
        "schema_version": GRPO_STEP_SCHEMA_VERSION,
        "global_step": global_step,
        "last_step_id": last_step_id,
        "samples_consumed": samples_consumed,
        "max_steps": max_steps,
        "k_rollouts": k_rollouts,
        "policy_model": policy_model,
        "seed": seed,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "samples_dir": str(samples_dir),
        "device": device,
        "save_every": save_every,
        "next_global_step_to_save": next_global_step_to_save
            if next_global_step_to_save is not None
            else ((global_step + 1) % save_every == 0
                  and global_step + 1 < max_steps
                  and global_step + 1),
        "completed": global_step + 1 >= max_steps,
    }
    (checkpoint_dir / "state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Persist the binary state only on the cadence determined by the
    # caller; the caller passes ``next_global_step_to_save`` to keep
    # this function side-effect-free (the decision whether to save was
    # made in main()).
    if state["next_global_step_to_save"]:
        blob = {
            "version": STATE_PT_VERSION,
            "model_state": {k: v.detach().cpu()
                            for k, v in model.state_dict().items()},
            "optimizer_state": (optimizer.state_dict()
                                if optimizer is not None else None),
            "rng_state": rng_state,
        }
        torch.save(blob, checkpoint_dir / "state.pt")


def _load_state(resume_from: Path) -> dict[str, Any]:
    return json.loads(resume_from.read_text(encoding="utf-8"))


def _load_binary_state(checkpoint_dir: Path) -> dict[str, Any]:
    blob_path = checkpoint_dir / "state.pt"
    if not blob_path.exists():
        raise FileNotFoundError(
            f"resume requested from {checkpoint_dir} but no state.pt present"
        )
    return torch.load(blob_path, map_location="cpu", weights_only=False)


def _restore_optimizer(optimizer: torch.optim.Optimizer,
                       state_dict: dict[str, Any]) -> None:
    """Restore optimizer state_dict. ``groups`` may need to be re-sized
    if the resumed model has different requires_grad parameters."""
    try:
        optimizer.load_state_dict(state_dict)
    except (ValueError, KeyError):
        # Mismatch (e.g. frozen/unfrozen parameter sets differ). The
        # optimizer is then effectively fresh; we warn loudly so the
        # caller can decide whether to abort.
        import warnings
        warnings.warn(
            "optimizer state_dict could not be fully restored (param group "
            "mismatch); optimizer is effectively fresh. The policy weights "
            "are still restored, so the next policy update will start from "
            "the restored trajectory but with reset Adam moments.",
        )


def _config_matches(saved: dict[str, Any], args, strict: bool) -> list[str]:
    """Return a list of CLI arg names whose values differ from the
    saved run config. Empty list = match."""
    expected = {
        "policy_model": args.policy_model,
        "k_rollouts": args.k_rollouts,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "samples_dir": str(args.samples_dir),
        "seed": args.seed,
        "device": args.device,
    }
    diffs: list[str] = []
    for key, want in expected.items():
        got = saved.get(key)
        # Numeric tolerance for floats
        if isinstance(want, float) and isinstance(got, (int, float)):
            if abs(float(got) - want) > 1e-9:
                diffs.append(f"{key}: saved={got} cli={want}")
        elif got != want:
            diffs.append(f"{key}: saved={got} cli={want}")
    return diffs


# ---------------------------------------------------------------------------
# Per-step artifact assembly
# ---------------------------------------------------------------------------

def _assemble_step_artifact(
    *, step_id: str, global_step: int, policy_model: str,
    sample: dict[str, Any], rollouts: list[dict[str, Any]],
    rewards: list[dict[str, Any]], advantages: list[float],
    update: dict[str, Any], seed: int,
    rng_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    art = {
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
    if rng_state is not None:
        # Record a deterministic fingerprint of the RNG state at the
        # moment the rollouts were generated so resume verification can
        # cross-check. We hash the (non-empty) keys + a few canonical
        # ints so the fingerprint is portable across versions.
        art["deterministic"]["rng_fingerprint"] = _fingerprint_rng(rng_state)
    return art


def _fingerprint_rng(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    # python state: tuple (version, internalstate, gauss_next)
    py = state.get("python")
    if py is not None:
        try:
            version, internal, gauss = py
            # internal[0] is index; internal[1] is the state tuple (big)
            h.update(repr(version).encode())
            h.update(repr(int(internal[0])).encode())
            h.update(repr(len(internal[1])).encode())
        except Exception:
            h.update(b"<python-state-unreadable>")
    torch_state = state.get("torch")
    if torch_state is not None:
        # torch_state is a torch.ByteTensor; use its sum + first/last
        # byte as a portable fingerprint.
        try:
            arr = torch_state.cpu().numpy()
            h.update(repr(int(arr.sum())).encode())
            h.update(repr(int(arr[0])).encode())
            h.update(repr(int(arr[-1])).encode())
        except Exception:
            h.update(b"<torch-state-unreadable>")
    return h.hexdigest()


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

    # Determine seed + resume cursor + load binary state (if any)
    resume_state: dict[str, Any] | None = None
    binary_state: dict[str, Any] | None = None
    seed = args.seed
    samples_consumed = 0
    next_global_step = 0
    optimizer: torch.optim.Optimizer | None = None

    if args.resume_from is not None:
        resume_state = _load_state(args.resume_from)
        # Resume uses the saved seed so the same sample shuffle order
        # is reproduced.
        seed = int(resume_state.get("seed", args.seed))
        samples_consumed = int(resume_state.get("samples_consumed", 0))
        next_global_step = int(resume_state.get("global_step", -1)) + 1

        diffs = _config_matches(resume_state, args, args.strict_resume_config)
        if diffs and args.strict_resume_config:
            print(f"[grpo] strict resume refused: {diffs}", flush=True)
            return 2
        if diffs:
            print(f"[grpo] resume config differs (non-strict): {diffs}",
                  flush=True)

        # Load the binary blob (model + optimizer + RNG) from the same
        # checkpoint directory the state.json lives in.
        binary_path = args.resume_from.parent
        try:
            binary_state = _load_binary_state(binary_path)
        except FileNotFoundError as e:
            print(f"[grpo] {e}", flush=True)
            return 3
        print(f"[grpo] resuming from {args.resume_from}: "
              f"global_step+1 = {next_global_step}, "
              f"samples_consumed = {samples_consumed}", flush=True)

    print(f"[grpo] loading policy model {args.policy_model} on {device}",
          flush=True)
    tokenizer = _load_tokenizer(args.policy_model)
    model = _load_model(args.policy_model, device)
    if device == "cpu":
        # CPU mode is intentionally limited; warn loudly
        print("[grpo] running on CPU: this is the smoke path, "
              "expect slow generation. Pass --device cuda if available.",
              flush=True)

    # Seed AFTER model load so any RNG state mutations during loading
    # are deterministic relative to the seed.
    _seed_all(seed, device)

    # Restore model + optimizer + RNG on resume
    if binary_state is not None:
        try:
            missing, unexpected = model.load_state_dict(
                binary_state["model_state"], strict=False
            )
            if missing or unexpected:
                print(f"[grpo] resume: model state missing={missing} "
                      f"unexpected={unexpected}", flush=True)
        except Exception as e:
            print(f"[grpo] failed to restore model state: {e}", flush=True)
            return 4
        if binary_state.get("optimizer_state") is not None:
            # Build the optimizer on the (restored) model, then load
            # state_dict so Adam moments carry over.
            optimizer = _fresh_optimizer(model, args.learning_rate)
            _restore_optimizer(optimizer, binary_state["optimizer_state"])
        if binary_state.get("rng_state") is not None:
            _restore_rng_state(binary_state["rng_state"])

    samples = _load_samples(args.samples_dir, args.limit, seed)
    print(f"[grpo] loaded {len(samples)} samples from {args.samples_dir}",
          flush=True)
    if not samples:
        print(f"[grpo] no samples found in {args.samples_dir}", flush=True)
        return 1

    # Resume-correct iteration: start at samples[samples_consumed]
    # and yield up to (max_steps - next_global_step) triples.
    remaining_steps = max(0, args.max_steps - next_global_step)
    if samples_consumed >= len(samples):
        # All samples consumed already: wrap around so a resume that
        # has fewer remaining steps than fresh samples still has work.
        samples_consumed = 0
    scheduled_samples = samples[samples_consumed:samples_consumed + remaining_steps]
    if not scheduled_samples:
        print(f"[grpo] nothing to do: global_step={next_global_step} >= "
              f"max_steps={args.max_steps}", flush=True)
        return 0

    last_step_id = ""
    for local_idx, sample in enumerate(scheduled_samples):
        global_step = next_global_step + local_idx
        sample_offset = samples_consumed + local_idx
        step_id = f"{global_step:04d}_{sample.get('id', 'unknown')}"
        print(f"[grpo] step {global_step + 1}/{args.max_steps}: "
              f"prompt_id={sample.get('id', '?')} "
              f"(sample_offset={sample_offset})", flush=True)

        rng_before = _capture_rng_state(device)
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
        update, optimizer = _policy_update(
            model, tokenizer, sample, rollouts, advantages,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            device=device,
            optimizer=optimizer,
        )
        artifact = _assemble_step_artifact(
            step_id=step_id, global_step=global_step,
            policy_model=args.policy_model, sample=sample,
            rollouts=rollouts, rewards=rewards, advantages=advantages,
            update=update, seed=seed, rng_state=rng_before,
        )
        step_path = checkpoint_dir / f"step-{step_id}.json"
        step_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        last_step_id = step_id
        next_save = ((global_step + 1) % args.save_every == 0)
        _save_state(
            checkpoint_dir,
            global_step=global_step,
            last_step_id=step_id,
            samples_consumed=sample_offset + 1,
            max_steps=args.max_steps,
            k_rollouts=args.k_rollouts,
            policy_model=args.policy_model,
            seed=seed,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            samples_dir=str(args.samples_dir),
            device=device,
            save_every=args.save_every,
            model=model, optimizer=optimizer,
            rng_state=_capture_rng_state(device),
            next_global_step_to_save=next_save,
        )
        print(f"[grpo] step {global_step} done: "
              f"loss={update['policy_gradient_loss']:.4f} "
              f"grad_norm={update['grad_norm']:.3f} "
              f"tokens={update['tokens_seen']} "
              f"-> {step_path.name}", flush=True)

    print(f"[grpo] all steps complete: state at "
          f"{checkpoint_dir / 'state.json'} + "
          f"{checkpoint_dir / 'state.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())