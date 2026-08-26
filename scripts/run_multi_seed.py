"""Run the N5 (4 scale) + N6 (3 dropout) training suite across 3 seeds.

Produces 21 result JSON artifacts (one per {scale, seed}) with seed from
{42, 123, 7}, plus a small overview JSON with mean/std of val_loss_min.

Usage:
    .venv/python.exe scripts/run_multi_seed.py
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# (config, artifact-stem) pairs for the 7 (scale, dropout) settings
RUNS = [
    ("configs/dense_training.owt-formal-curve.example.yaml", "dense-owt-formal-curve"),
    ("configs/dense_training.owt-formal-curve-small.example.yaml", "dense-owt-formal-curve-small"),
    ("configs/dense_training.owt-formal-curve-medium.example.yaml", "dense-owt-formal-curve-medium"),
    ("configs/dense_training.owt-formal-curve-large.example.yaml", "dense-owt-formal-curve-large"),
    ("configs/dense_training.owt-formal-curve-medium-dropout01.example.yaml", "dense-owt-formal-curve-medium-dropout01"),
    ("configs/dense_training.owt-formal-curve-medium-dropout02.example.yaml", "dense-owt-formal-curve-medium-dropout02"),
]
SEEDS = [42, 123, 7]
SEED_SUFFIX = {42: "seed42", 123: "seed123", 7: "seed7"}


def main() -> int:
    tmpdir = ROOT / "artifacts" / "multi-seed-configs"
    tmpdir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for config_rel, stem in RUNS:
        config_path = ROOT / config_rel
        base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["training"]["seed"] = seed
            cfg["training"]["checkpoint"] = f"artifacts/checkpoints/{stem}-{SEED_SUFFIX[seed]}.pt"
            out_json = f"artifacts/{stem}-{SEED_SUFFIX[seed]}-result.json"
            tmp_cfg = tmpdir / f"{stem}-{SEED_SUFFIX[seed]}.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"[run_multi_seed] {stem} seed={seed} -> {out_json}", flush=True)
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "train_dense.py"),
                 "--config", str(tmp_cfg), "--output", str(ROOT / out_json)],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            if proc.returncode != 0:
                print(f"[run_multi_seed] FAILED {stem} seed={seed}: {proc.stderr[-500:]}", flush=True)
                continue
            result = json.loads((ROOT / out_json).read_text(encoding="utf-8"))
            summary_rows.append({
                "config": config_rel,
                "seed": seed,
                "val_loss_min": result["metrics"]["curve_summary"]["val_loss_min"],
                "val_loss_min_step": result["metrics"]["curve_summary"]["val_loss_min_step"],
            })

    # Group by config and report mean/std of val_loss_min
    by_config: dict[str, list[float]] = {}
    for row in summary_rows:
        by_config.setdefault(row["config"], []).append(row["val_loss_min"])
    overview: dict = {"runs": summary_rows, "config_summary": {}}
    for cfg, vals in by_config.items():
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        overview["config_summary"][cfg] = {
            "seeds": SEEDS[: len(vals)],
            "val_loss_min_values": vals,
            "val_loss_min_mean": mean,
            "val_loss_min_std": std,
        }
    overview_path = ROOT / "artifacts" / "multi-seed-overview.json"
    overview_path.write_text(json.dumps(overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[run_multi_seed] wrote {overview_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())