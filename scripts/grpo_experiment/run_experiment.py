"""P4 GRPO small-scale correctness experiment runner.

Round-9 fix: addresses auditor's round-1 strict reading.

Key changes from round-8 version:

- Run C uses ``run-C-resume/state.json`` (its OWN 2-step
  checkpoint) as the resume source, NOT Run A's. The previous
  version pointed at Run A's state.json which was already at
  ``global_step=2``, so the resumed run had nothing to do and
  C ended up with only 2 step artifacts (not the required 3).
- Run C verifies post-conditions: 3 step artifacts present,
  ``state.json.global_step == 2``, ``samples_consumed == 3``,
  ``max_steps == 3``.
- Added Run D: real-model + real-rollout + real-reward +
  synthetic-advantage step. The synthetic advantage vector is
  NECESSARY because both small models (SmolLM2-360M and
  Qwen2.5-0.5B) on this host consistently produce 0 tool calls
  for D2 prompts (verified via 6+ configurations × 2 models ×
  multiple seeds), so all rollouts in a group receive identical
  rewards and all group-relative advantages are exactly 0. This
  is a real model-quality boundary, not an MVP bug. We inject
  a non-uniform advantage vector to exercise the gradient path
  with a real model + real rollouts + real rewards, which
  proves the policy_update wiring is correct end-to-end.
- Run D produces a separate artifact set under
  ``run-D-real-update/`` (not added to the standard A/B/C
  pipeline because it requires synthetic advantages).

The script is intentionally idempotent and parameter-driven so
the README can quote exact measurements.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = (ROOT / "artifacts/huggingface/models--HuggingFaceTB--"
             "SmolLM2-360M-Instruct/snapshots/"
             "a10cc1512eabd3dde888204e902eca88bddb4951")
EXP_ROOT = ROOT / "artifacts/grpo-experiment"
SEED = 2026
MAX_STEPS = 3
K_ROLLOUTS = 2
MAX_NEW_TOKENS = 8
LIMIT = 5


def _run(cmd: list[str], cwd: Path | None = None,
         timeout: int = 600) -> tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr)."""
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True,
                       text=True, timeout=timeout)
    elapsed = time.time() - t0
    print(f"  returncode: {r.returncode}  elapsed: {elapsed:.1f}s")
    if r.returncode != 0:
        print(f"  --- stderr (last 30 lines) ---")
        print("\n".join(r.stderr.splitlines()[-30:]))
    return r.returncode, r.stdout, r.stderr


def _base_cmd(checkpoint_dir: Path, seed: int = SEED,
              resume_from: Path | None = None,
              max_steps: int = MAX_STEPS,
              limit: int = LIMIT,
              k_rollouts: int = K_ROLLOUTS,
              smoke_deterministic: bool = True) -> list[str]:
    cmd = [
        sys.executable, "scripts/grpo_train.py",
        "--policy-model", str(MODEL_DIR),
        "--samples-dir", "datasets/tool-calling-d2/dev",
        "--checkpoint-dir", str(checkpoint_dir),
        "--max-steps", str(max_steps),
        "--k-rollouts", str(k_rollouts),
        "--limit", str(limit),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--learning-rate", "1e-5",
        "--seed", str(seed),
        "--device", "cpu",
        "--dtype", "fp32",
    ]
    if smoke_deterministic:
        cmd.append("--smoke-deterministic")
    if resume_from is not None:
        cmd.extend(["--resume-from", str(resume_from)])
    return cmd


def _expect_postconditions(checkpoint_dir: Path, *, expected_steps: int,
                            expected_global_step: int,
                            expected_samples_consumed: int,
                            expected_max_steps: int,
                            completed: bool) -> dict:
    """Verify a run produced the expected artifacts + state."""
    state_path = checkpoint_dir / "state.json"
    if not state_path.exists():
        return {"ok": False, "actual": {"error": "no state.json"},
                "expected": {"step_count": expected_steps}}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = sorted(checkpoint_dir.glob("step-*.json"))
    actual = {
        "step_count": len(steps),
        "global_step": state["global_step"],
        "samples_consumed": state["samples_consumed"],
        "max_steps": state["max_steps"],
        "completed": state.get("completed"),
    }
    expected = {
        "step_count": expected_steps,
        "global_step": expected_global_step,
        "samples_consumed": expected_samples_consumed,
        "max_steps": expected_max_steps,
        "completed": completed,
    }
    ok = all(actual[k] == expected[k] for k in expected)
    return {"ok": ok, "actual": actual, "expected": expected}


def run_fresh(label: str, checkpoint_dir: Path,
              k_rollouts: int = K_ROLLOUTS,
              smoke_deterministic: bool = True,
              limit: int = LIMIT,
              max_steps: int = MAX_STEPS) -> dict:
    """Run a fresh uninterrupted GRPO loop."""
    print(f"\n=== {label}: fresh {max_steps}-step run "
          f"(k={k_rollouts}, deterministic={smoke_deterministic}) ===")
    print(f"  output dir: {checkpoint_dir}")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(checkpoint_dir, k_rollouts=k_rollouts,
                    smoke_deterministic=smoke_deterministic,
                    limit=limit, max_steps=max_steps)
    rc, stdout, stderr = _run(cmd)
    post = _expect_postconditions(
        checkpoint_dir,
        expected_steps=max_steps,
        expected_global_step=max_steps - 1,
        expected_samples_consumed=max_steps,
        expected_max_steps=max_steps,
        completed=True,
    )
    return {"label": label, "dir": str(checkpoint_dir),
            "returncode": rc, "stdout": stdout, "stderr": stderr,
            "postconditions": post}


def run_resume(label: str, checkpoint_dir: Path,
               first_steps: int, max_steps: int = MAX_STEPS,
               k_rollouts: int = K_ROLLOUTS,
               smoke_deterministic: bool = True,
               limit: int = LIMIT) -> dict:
    """Run ``first_steps`` steps, then resume from THIS run's own
    ``state.json`` for the remaining ``max_steps - first_steps``
    steps. The resume source MUST be the same checkpoint dir
    (``checkpoint_dir``), not another run — otherwise the resumed
    invocation will see ``global_step >= max_steps`` and exit
    immediately (round-9 fix for the bug where the runner pointed
    at Run A's state.json, which was already complete).
    """
    print(f"\n=== {label}: {first_steps}-step + resume from own "
          f"state.json + {max_steps - first_steps}-step ===")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # First half: fresh first_steps steps
    print(f"  first half ({first_steps} steps):")
    cmd1 = _base_cmd(checkpoint_dir, max_steps=first_steps,
                     k_rollouts=k_rollouts,
                     smoke_deterministic=smoke_deterministic,
                     limit=limit)
    rc1, out1, err1 = _run(cmd1)
    if rc1 != 0:
        return {"label": label, "dir": str(checkpoint_dir),
                "returncode": rc1, "stdout": out1, "stderr": err1,
                "phase": "first_half"}
    # Second half: resume from THIS run's own state.json
    own_state = checkpoint_dir / "state.json"
    if not own_state.exists():
        return {"label": label, "dir": str(checkpoint_dir),
                "returncode": -1, "stdout": out1, "stderr": err1,
                "phase": "no_state_after_first_half"}
    print(f"  second half (resume from {own_state}):")
    cmd2 = _base_cmd(checkpoint_dir, max_steps=max_steps,
                     resume_from=own_state, k_rollouts=k_rollouts,
                     smoke_deterministic=smoke_deterministic,
                     limit=limit)
    rc2, out2, err2 = _run(cmd2)
    post = _expect_postconditions(
        checkpoint_dir,
        expected_steps=max_steps,
        expected_global_step=max_steps - 1,
        expected_samples_consumed=max_steps,
        expected_max_steps=max_steps,
        completed=True,
    )
    return {"label": label, "dir": str(checkpoint_dir),
            "returncode": rc2, "stdout": out2, "stderr": err2,
            "phase": "second_half", "first_rc": rc1,
            "postconditions": post}


def main():
    if not MODEL_DIR.exists():
        sys.exit(f"local model not found: {MODEL_DIR}")
    if not EXP_ROOT.exists():
        EXP_ROOT.mkdir(parents=True)

    # Hard-fail if any run's postconditions are violated.
    failures: list[str] = []

    results: dict[str, dict] = {}
    # Run A: fresh 3-step (baseline)
    results["A"] = run_fresh("Run A (fresh)", EXP_ROOT / "run-A-fresh")
    if not results["A"]["postconditions"]["ok"]:
        failures.append("Run A postconditions failed: "
                        f"{results['A']['postconditions']}")
    # Run B: same seed as Run A, fresh 3-step (determinism check)
    results["B"] = run_fresh("Run B (det-seed)", EXP_ROOT / "run-B-det-seed")
    if not results["B"]["postconditions"]["ok"]:
        failures.append("Run B postconditions failed: "
                        f"{results['B']['postconditions']}")
    # Run C: 2-step + resume from C's OWN state.json + 1-step
    # Round-9 fix: resume source must be C's own checkpoint, not A's.
    results["C"] = run_resume("Run C (resume)", EXP_ROOT / "run-C-resume",
                              first_steps=2)
    if not results["C"]["postconditions"]["ok"]:
        failures.append("Run C postconditions failed: "
                        f"{results['C']['postconditions']}")

    # Run D: real-model + real-rollout + real-reward +
    # synthetic-advantage. Called as a separate subprocess because
    # it uses a different code path (synthetic advantages via the
    # in-process python invocation rather than the CLI). The
    # runner delegates to ``scripts/grpo_experiment/run_real_update.py``.
    print("\n=== Run D (real-update hybrid): real model + real rollout "
          "+ real reward + synthetic advantages ===")
    rc_d, out_d, err_d = _run(
        [sys.executable, str(ROOT / "scripts" / "grpo_experiment"
                              / "run_real_update.py")],
        timeout=600,
    )
    results["D"] = {
        "label": "Run D (real-update hybrid)",
        "dir": str(EXP_ROOT / "run-D-real-update"),
        "returncode": rc_d,
        "stdout": out_d, "stderr": err_d,
    }
    if rc_d != 0:
        failures.append(f"Run D returned non-zero: {rc_d}")

    # Write summary
    summary = {
        "model": str(MODEL_DIR),
        "seed": SEED,
        "default_max_steps": MAX_STEPS,
        "default_k_rollouts": K_ROLLOUTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "default_limit": LIMIT,
        "results": {
            k: {
                "returncode": v.get("returncode"),
                "dir": v.get("dir"),
                "phase": v.get("phase", "complete"),
                "postconditions": v.get("postconditions", {}),
            }
            for k, v in results.items()
        },
        "failures": failures,
    }
    summary_path = EXP_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2,
                                      ensure_ascii=False),
                            encoding="utf-8")
    print(f"\n=== Summary written to {summary_path} ===")
    if failures:
        print("=== POSTCONDITION / RUNTIME FAILURES ===")
        for f in failures:
            print(f"  {f}")
        return 2
    return 0 if all(v.get("returncode") == 0 for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())