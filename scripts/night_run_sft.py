"""Night-run orchestrator: extend D1.1 dataset, then SFT-retrain the model.

Phase A — Extend ``datasets/tool-calling-d1-llm`` to ``--target`` samples
(1500 by default). The generator uses --skip-existing so successive
attempts fill in only missing files. Providers:

  1. ``minimax`` (MiniMax-M3) — primary, via auth.json::minimax-cn.
  2. ``deepseek`` (deepseek-chat) — fallback on quota/rate-limit errors.

Phase B — Train a new Dense-large SFT checkpoint using the resulting
dataset (deterministic augmentation k=3; 20000 optimizer steps).
Both phases are run sequentially in a single background process; each
phase's stdout/stderr is captured under ``.pi-glla/scratch/night-*.log``.

Usage:
    .venv/python.exe scripts/night_run_sft.py --target 1500 \
        --train-steps 20000

The script exits 0 on success. To resume after a kill, just rerun — the
generator and SFT trainer both support idempotent resumption
(--skip-existing / checkpoint resume).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".pi-glla" / "scratch"
D1LLM_DIR = ROOT / "datasets" / "tool-calling-d1-llm"

# Provider registry. Each entry maps the auth.json key to its API endpoint
# and model identifier.
PROVIDERS: dict[str, dict[str, str]] = {
    "minimax": {
        "auth_key": "minimax-cn",
        "base_url": "https://api.minimaxi.com/anthropic",
        "model": "MiniMax-M3",
        "model_version": "MiniMax-M3@minimax-cn",
    },
    "deepseek": {
        "auth_key": "deepseek",
        "base_url": "https://api.deepseek.com/anthropic",
        "model": "deepseek-chat",
        "model_version": "deepseek-chat@deepseek",
    },
}

QUOTA_PATTERNS = [
    r"\b429\b",
    r"\bquota\b",
    r"\brate.?limit\b",
    r"\bexceeded\b",
    r"\binsufficient.?balance\b",
    r"\btoo.?many.?requests\b",
]


def load_auth(auth_path: Path) -> dict:
    return json.loads(auth_path.read_text(encoding="utf-8"))


def read_api_key(auth: dict, auth_key: str) -> str:
    entry = auth[auth_key]
    if entry.get("type") != "api_key":
        raise ValueError(f"auth entry '{auth_key}' is not api_key type")
    return entry["key"]


def count_samples(out_dir: Path) -> int:
    train_dir = out_dir / "train"
    if not train_dir.is_dir():
        return 0
    return sum(1 for _ in train_dir.glob("d1llm-*.json"))


def log_indicates_quota(log_text: str) -> bool:
    lower = log_text.lower()
    return any(re.search(p, lower) for p in QUOTA_PATTERNS)


def run_phase_a(
    *,
    target: int,
    auth_path: Path,
    providers: list[str],
) -> dict:
    """Extend D1.1 to ``target`` samples, with auto fallback on quota."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    auth = load_auth(auth_path)
    started = count_samples(D1LLM_DIR)
    print(f"[night-a] starting count={started}, target={target}", flush=True)
    tried: list[str] = []
    for provider in providers:
        current = count_samples(D1LLM_DIR)
        if current >= target:
            print(f"[night-a] target reached ({current}>= {target})", flush=True)
            break
        cfg = PROVIDERS[provider]
        api_key = read_api_key(auth, cfg["auth_key"])
        env = os.environ.copy()
        env["D1_LLM_API_KEY"] = api_key
        env["D1_LLM_API_BASE"] = cfg["base_url"]
        env["D1_LLM_MODEL"] = cfg["model"]
        env["PYTHONIOENCODING"] = "utf-8"
        log_path = SCRATCH / f"night-phase-a-{provider}.log"
        cmd = [
            str(ROOT / ".venv" / "python.exe"),
            "-u", "scripts/generate_d1_llm.py",
            "--count", str(target),
            "--split", "train",
            "--out", "datasets/tool-calling-d1-llm",
            "--skip-existing",
            "--model-version", cfg["model_version"],
        ]
        print(f"[night-a] provider={provider}: have {current}, need "
              f"{target - current} more -> {log_path.name}", flush=True)
        with open(log_path, "ab") as logf:
            proc = subprocess.run(cmd, env=env, cwd=str(ROOT),
                                  stdout=logf, stderr=subprocess.STDOUT)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        final = count_samples(D1LLM_DIR)
        print(f"[night-a]   generated={final - current}, total={final}, "
              f"rc={proc.returncode}", flush=True)
        tried.append(provider)
        if final >= target:
            break
        if not log_indicates_quota(log_text):
            print(f"[night-a]   no quota error in log; stopping", flush=True)
            break
        print(f"[night-a]   quota/rate-limit detected; switching provider", flush=True)
    return {
        "phase": "A",
        "providers_tried": tried,
        "started": started,
        "ended": count_samples(D1LLM_DIR),
        "target": target,
    }


def run_phase_b(*, train_steps: int, auth_path: Path | None = None) -> dict:
    """Train large SFT for ``train_steps`` optimizer steps.

    The orchestrator writes a *scratch* config under .pi-glla/scratch/
    rather than mutating the source sft-large.example.yaml on main, so a
    short interactive run (configs/sft-large.example.yaml) and a long
    night run can coexist without overwriting each other.
    """
    import yaml  # local import; pyyaml only needed here.
    src_cfg = ROOT / "configs" / "sft-large.example.yaml"
    cfg = yaml.safe_load(src_cfg.read_text(encoding="utf-8"))
    cfg["training"]["max_steps"] = int(train_steps)
    cfg["training"]["log_interval"] = max(100, train_steps // 40)
    cfg["training"]["checkpoint"] = "artifacts/checkpoints/sft-tool-large-night.pt"
    scratch_cfg = SCRATCH / "sft-large-night.yaml"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    scratch_cfg.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    log_path = SCRATCH / "night-phase-b.log"
    cmd = [
        str(ROOT / ".venv" / "python.exe"),
        "-u", "scripts/train_sft.py",
        "--config", str(scratch_cfg),
        "--init-checkpoint", "artifacts/checkpoints/dense-owt-formal-curve-large.pt",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"[night-b] launching {train_steps}-step SFT (config: {scratch_cfg.name})",
          flush=True)
    with open(log_path, "wb") as logf:
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT),
                              stdout=logf, stderr=subprocess.STDOUT)
    print(f"[night-b] done, rc={proc.returncode}", flush=True)
    return {
        "phase": "B",
        "steps": train_steps,
        "checkpoint": "artifacts/checkpoints/sft-tool-large-night.pt",
        "config": str(scratch_cfg.relative_to(ROOT)),
        "rc": proc.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1500,
                        help="D1.1 total samples to reach in Phase A")
    parser.add_argument("--train-steps", type=int, default=20000,
                        help="SFT optimizer steps in Phase B")
    parser.add_argument("--auth", type=Path,
                        default=Path.home() / ".pi" / "agent" / "auth.json")
    parser.add_argument("--skip-phase-a", action="store_true",
                        help="skip Phase A (assume dataset already extended)")
    parser.add_argument("--skip-phase-b", action="store_true",
                        help="skip Phase B (training only)")
    args = parser.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    summary: dict = {"target": args.target, "train_steps": args.train_steps,
                     "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not args.skip_phase_a:
        summary["phase_a"] = run_phase_a(
            target=args.target, auth_path=args.auth, providers=list(PROVIDERS),
        )
    if not args.skip_phase_b:
        summary["phase_b"] = run_phase_b(train_steps=args.train_steps)
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (SCRATCH / "night-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[night] DONE: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())