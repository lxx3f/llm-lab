"""Run the N5 (4 scale) + N6 (3 dropout) training suite across 3 seeds.

Produces 21 result JSON artifacts (one per {scale, seed}) with seed from
{42, 123, 7}, plus a small overview JSON with mean/std of val_loss_min.

Usage:
    .venv/python.exe scripts/run_multi_seed.py
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# (config, artifact-stem) pairs for all sweep configs.
# N5 scale (4) + N6 dropout (2) + N7 rope (3) + N8 heads (2) + N9 dff (2) = 13 configs.
# Control configs (medium-ropebase-10k, medium-heads-4, medium-dff-512) all map
# to the medium config, so they share artifact filenames across sweeps.
RUNS = [
    ("configs/dense_training.owt-formal-curve.example.yaml", "dense-owt-formal-curve"),
    ("configs/dense_training.owt-formal-curve-small.example.yaml", "dense-owt-formal-curve-small"),
    ("configs/dense_training.owt-formal-curve-medium.example.yaml", "dense-owt-formal-curve-medium"),
    ("configs/dense_training.owt-formal-curve-large.example.yaml", "dense-owt-formal-curve-large"),
    ("configs/dense_training.owt-formal-curve-medium-dropout01.example.yaml", "dense-owt-formal-curve-medium-dropout01"),
    ("configs/dense_training.owt-formal-curve-medium-dropout02.example.yaml", "dense-owt-formal-curve-medium-dropout02"),
    ("configs/dense_training.owt-formal-curve-medium-ropebase-10k.example.yaml", "dense-owt-formal-curve-medium-ropebase-10k"),
    ("configs/dense_training.owt-formal-curve-medium-ropebase-50k.example.yaml", "dense-owt-formal-curve-medium-ropebase-50k"),
    ("configs/dense_training.owt-formal-curve-medium-ropebase-100k.example.yaml", "dense-owt-formal-curve-medium-ropebase-100k"),
    ("configs/dense_training.owt-formal-curve-medium-heads-2.example.yaml", "dense-owt-formal-curve-medium-heads-2"),
    ("configs/dense_training.owt-formal-curve-medium-heads-8.example.yaml", "dense-owt-formal-curve-medium-heads-8"),
    ("configs/dense_training.owt-formal-curve-medium-dff-256.example.yaml", "dense-owt-formal-curve-medium-dff-256"),
    ("configs/dense_training.owt-formal-curve-medium-dff-1024.example.yaml", "dense-owt-formal-curve-medium-dff-1024"),
    # N11 long curve (50000 steps, seed 42 already exists as -result.json;
    # multi-seed variant writes -seed{42,123,7}-result.json).
    ("configs/dense_training.owt-formal-curve-long-baseline.example.yaml", "dense-owt-formal-curve-long-baseline"),
    ("configs/dense_training.owt-formal-curve-long-medium.example.yaml", "dense-owt-formal-curve-long-medium"),
    ("configs/dense_training.owt-formal-curve-long-large.example.yaml", "dense-owt-formal-curve-long-large"),
    # MoE curve (5000 step + 50000 step) — multi-seed validation against
    # the existing single-seed observations (val_min 7.22 @ 4600 / 6.04 @ 50000).
    ("configs/moe_training.owt-formal-curve.example.yaml", "moe-owt-formal-curve"),
    ("configs/moe_training.owt-formal-curve-long.example.yaml", "moe-owt-formal-curve-long"),
]
SEEDS = [42, 123, 7]
SEED_SUFFIX = {42: "seed42", 123: "seed123", 7: "seed7"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="only run config stems containing any of these substrings")
    args = parser.parse_args()
    tmpdir = ROOT / "artifacts" / "multi-seed-configs"
    tmpdir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for config_rel, stem in RUNS:
        if args.only and not any(sub in stem for sub in args.only):
            continue
        config_path = ROOT / config_rel
        base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for seed in SEEDS:
            out_json = f"artifacts/{stem}-{SEED_SUFFIX[seed]}-result.json"
            if (ROOT / out_json).is_file():
                # Reuse an existing artifact instead of re-training.
                result = json.loads((ROOT / out_json).read_text(encoding="utf-8"))
                summary_rows.append({
                    "config": config_rel,
                    "seed": seed,
                    "val_loss_min": result["metrics"]["curve_summary"]["val_loss_min"],
                    "val_loss_min_step": result["metrics"]["curve_summary"]["val_loss_min_step"],
                })
                print(f"[run_multi_seed] SKIP (exists) {stem} seed={seed}", flush=True)
                continue
            cfg = copy.deepcopy(base)
            cfg["training"]["seed"] = seed
            cfg["training"]["checkpoint"] = f"artifacts/checkpoints/{stem}-{SEED_SUFFIX[seed]}.pt"
            out_json = f"artifacts/{stem}-{SEED_SUFFIX[seed]}-result.json"
            tmp_cfg = tmpdir / f"{stem}-{SEED_SUFFIX[seed]}.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"[run_multi_seed] {stem} seed={seed} -> {out_json}", flush=True)
            # Dispatch to MoE or Dense trainer based on config architecture.
            architecture = cfg.get("model", {}).get("architecture", "DenseTransformer")
            if architecture == "MoETransformer":
                train_script = "train_moe.py"
            else:
                train_script = "train_dense.py"
            proc = subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts" / train_script),
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

    # Group by config and report mean/std of val_loss_min. When --only is
    # used, rebuild the overview from ALL existing artifacts so partial runs
    # do not wipe previously collected seeds.
    by_config: dict[str, list[float]] = {}
    for config_rel, stem in RUNS:
        for seed in SEEDS:
            out_json = ROOT / f"artifacts/{stem}-{SEED_SUFFIX[seed]}-result.json"
            if not out_json.is_file():
                continue
            result = json.loads(out_json.read_text(encoding="utf-8"))
            by_config.setdefault(config_rel, []).append(
                result["metrics"]["curve_summary"]["val_loss_min"]
            )
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