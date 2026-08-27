"""Aggregate d256 multi-seed held-out evaluation results.

Reads N eval JSONs (one per seed) and produces a single aggregate JSON
containing per-metric mean and population standard deviation (P1-03
protocol), with a primary focus on tool-calling quality:

- parse_success_rate: fraction of samples where tool_calls JSON parsed
  successfully (regardless of name/argument correctness).
- eight_level_layer_distribution: count of samples that reached each
  P1-05 classifier layer before first failure, summed across seeds.
- no_failure_rate: fraction of samples where all eight layers passed.

Metric definitions are documented in
docs/protocols/p1-05-failure-classification.md.

Usage:
    .venv/python.exe scripts/aggregate_d256_eval.py \\
        --eval-glob "artifacts/sft-d256-*-eval-d1dev.json" \\
        --output artifacts/sft-d256-eval-aggregate.json \\
        --label "d256 20k × 3 seeds (held-out D1 dev)"

Notes:
- The aggregator uses POPULATION standard deviation (÷N), matching
  the P1-03 protocol and dense_curve/eval_sft_tool conventions.
- If parse_success_rate is degenerate (all seeds = 0), the std is
  also 0; this is the truthful result for the d256 20k runs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pop_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-glob", required=True,
                        help="Glob pattern for per-seed eval JSONs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True,
                        help="Human label for this aggregation")
    args = parser.parse_args()

    eval_glob_path = Path(args.eval_glob)
    if eval_glob_path.is_absolute():
        eval_paths = sorted(eval_glob_path.parent.glob(eval_glob_path.name))
    else:
        eval_paths = sorted(ROOT.glob(args.eval_glob))
    if not eval_paths:
        raise SystemExit(f"no eval JSONs matched: {args.eval_glob}")
    seeds: list[dict] = []
    for path in eval_paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        seeds.append({
            "path": str(path.relative_to(ROOT)),
            "checkpoint": data["summary"]["checkpoint"],
            "total": data["summary"]["total"],
            "parse_success_rate": data["summary"]["parse_success_rate"],
            "no_failure": data["summary"]["no_failure"],
            "first_failure_distribution": data["summary"]["first_failure_distribution"],
            "rows": data.get("rows", []),
        })

    parse_rates = [s["parse_success_rate"] for s in seeds]
    no_failure_rates = [s["no_failure"] / s["total"] if s["total"] else 0.0
                        for s in seeds]
    parse_mean = sum(parse_rates) / len(parse_rates)
    parse_std = pop_std(parse_rates)
    no_fail_mean = sum(no_failure_rates) / len(no_failure_rates)
    no_fail_std = pop_std(no_failure_rates)

    agg_layer_dist: Counter[str] = Counter()
    for s in seeds:
        for layer, count in s["first_failure_distribution"].items():
            agg_layer_dist[layer] += count
    total_evaluations = sum(s["total"] for s in seeds)
    layer_pct = {layer: count / total_evaluations
                 for layer, count in agg_layer_dist.items()}

    output = {
        "label": args.label,
        "n_seeds": len(seeds),
        "total_evaluations": total_evaluations,
        "metric_definitions": {
            "parse_success_rate": (
                "fraction of samples whose generation produces valid "
                "tool_calls JSON (regardless of name/argument correctness)"
            ),
            "no_failure_rate": (
                "fraction of samples that pass all 8 P1-05 layers "
                "(parse_success, schema_valid, tool_name_correct, "
                "argument_value_correct, call_plan_matches, "
                "execution_success, result_grounded, final_answer_correct)"
            ),
            "first_failure_layer_distribution": (
                "histogram of first failure layer; counts summed across "
                "all seeds and all samples"
            ),
            "std_convention": "population std (÷N), matching P1-03 protocol",
        },
        "per_seed": seeds,
        "aggregate_metrics": {
            "parse_success_rate": {
                "mean": parse_mean,
                "pop_std": parse_std,
                "per_seed": parse_rates,
            },
            "no_failure_rate": {
                "mean": no_fail_mean,
                "pop_std": no_fail_std,
                "per_seed": no_failure_rates,
            },
            "first_failure_layer_distribution_count": dict(agg_layer_dist),
            "first_failure_layer_distribution_pct": layer_pct,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    try:
        out_rel = args.output.resolve().relative_to(ROOT.resolve())
    except ValueError:
        out_rel = args.output
    print(f"[aggregate] wrote {out_rel}")
    print(f"[aggregate] {len(seeds)} seeds, {total_evaluations} evaluations")
    print(f"[aggregate] parse_success_rate mean={parse_mean:.4f} ± {parse_std:.4f}")
    print(f"[aggregate] no_failure_rate mean={no_fail_mean:.4f} ± {no_fail_std:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())