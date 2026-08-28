"""Offline reward computation for tool-calling transcripts.

P2 deterministic evaluator: given a sample and a transcript (either a
real model generation or a deterministic replay), this module produces a
``reward_signal`` dict per the schema in
``schemas/reward_signal.schema.json``.

The reward is derived from the existing P1-05 eight-layer classifier
(see ``scripts/classify_tool_failure.py``); this module only adds the
reward mapping and the deterministic interface needed for offline
validation / GRPO rollouts.

Two reward signals are exposed:

- ``reward_binary``: 1.0 iff ``first_failure is None`` (all eight
  layers passed); 0.0 otherwise.
- ``reward_layered``: ``layer_pass_count / layer_pass_total``. ``None``
  layers (N/A) are excluded from the denominator so the score reflects
  only applicable checks.

CLI:

    .venv/python.exe scripts/reward_offline.py \
        --samples-dir datasets/tool-calling-d1/dev \
        --transcripts eval/sft-d256-seed42-eval-d1dev.json \
        --output artifacts/reward-d1dev-d256.json

The transcript JSON must be the ``summary`` + ``rows`` produced by
``scripts/eval_sft_tool.py``; ``rows[i]`` carries ``sample_id``,
``extracted_calls``, ``layers``, ``first_failure``, and ``generated``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REWARD_OFFLINE_VERSION = "1.0"
TRANSCRIPT_KEYS_USED = (
    "sample_id", "extracted_calls", "layers", "first_failure",
)
# Eight P1-05 layers; ``task_success`` is a backward-compat alias and is
# excluded from the reward signal so the denominator matches the protocol.
EIGHT_LAYERS: tuple[str, ...] = (
    "parse_success",
    "schema_valid",
    "tool_name_correct",
    "argument_value_correct",
    "call_plan_matches",
    "execution_success",
    "result_grounded",
    "final_answer_correct",
)


def _to_transcript(row: dict[str, Any]) -> dict[str, Any]:
    """Map an eval_sft_tool row to the shape ``classify()`` expects."""
    return {
        "tool_calls": row.get("extracted_calls", []),
        "final_answer": row.get("generated", ""),
    }


def _dominant_reward(layers_full: dict[str, Any],
                    first_failure: str | None) -> str:
    """Map the P1-05 layer outcome to one of the four canonical reward channels.

    The reward_type is the **first failure channel** in the P1-05 chain,
    aligned with its documented semantics:

    - ``parse_success``: ``layers.parse_success is not True``. This is the
      only branch that inspects the layer dict rather than ``first_failure``
      because a malformed transcript has no clean ``first_failure``.
    - ``final_answer_correct``: every earlier layer passes; only the
      final-answer layer fails (``first_failure == "final_answer_correct"``).
    - ``execution_correct``: every applicable layer passes
      (``first_failure is None``).
    - ``argument_correct``: parse + schema + name + argument + plan layers
      pass but execution / grounding / answer did not. This bucket
      covers all of ``first_failure in {"tool_name_correct",
      "argument_value_correct", "call_plan_matches", "execution_success",
      "result_grounded"}``.

    Defensive contract:

    - If ``layers_full["parse_success"] is not True``, the function returns
      ``parse_success`` regardless of ``first_failure``. The classifier
      guarantees ``first_failure == "parse_success"`` in this case, but if
      a malformed caller passes ``first_failure=None`` the contract still
      holds: a transcript that fails parse cannot have executed cleanly.
    - Any ``first_failure`` outside the canonical P1-05 layer names falls
      back to ``argument_correct``; the P1-05 classifier is the source of
      truth for layer names so this branch should not trigger under normal
      flow.

    Note: an earlier version of this function returned ``execution_correct``
    whenever ``first_failure in (None, "execution_success", "result_grounded")``
    — that mapping was wrong because execution_success / result_grounded
    failures were labelled as "execution_correct" even though execution /
    grounding did not pass. The current implementation only emits
    ``execution_correct`` when ``first_failure is None``.
    """
    if layers_full.get("parse_success") is not True:
        return "parse_success"
    if first_failure == "final_answer_correct":
        return "final_answer_correct"
    if first_failure is None:
        return "execution_correct"
    # All remaining canonical ``first_failure`` values are argument-layer
    # failures (or execution / grounding failures that imply the argument
    # chain held but the execution chain fell short): bucket them under
    # ``argument_correct``.
    return "argument_correct"


def compute_reward(
    sample: dict[str, Any],
    transcript_row: dict[str, Any],
    *,
    transcript_kind: str = "model_generated",
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Compute the reward signal for one (sample, transcript) pair.

    ``transcript_row`` must carry at least ``extracted_calls`` and
    ``first_failure``/``layers`` from a P1-05 classification, plus
    optionally ``generated`` and ``sample_id``. We re-import the
    classifier lazily to avoid a circular import.
    """
    classifier_path = ROOT / "scripts" / "classify_tool_failure.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_p105_classifier", classifier_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classify = module.classify

    transcript = _to_transcript(transcript_row)
    classification = classify(sample, transcript)
    layers_full: dict[str, Any] = classification["layers"]
    first_failure: str | None = classification["first_failure"]

    pass_count = sum(1 for k in EIGHT_LAYERS
                      if layers_full.get(k) is True)
    pass_total = sum(1 for k in EIGHT_LAYERS
                      if layers_full.get(k) is not None)
    reward_binary = 1.0 if first_failure is None else 0.0
    reward_layered = (
        float(pass_count) / float(pass_total) if pass_total else 0.0
    )

    return {
        "schema_version": "1.0",
        "sample_id": transcript_row.get("sample_id") or sample.get("id", ""),
        "task_type": sample.get("metadata", {}).get("task_type", ""),
        "transcript_kind": transcript_kind,
        "checkpoint": checkpoint,
        "reward_type": _dominant_reward(layers_full, first_failure),
        "reward_binary": reward_binary,
        "reward_layered": reward_layered,
        "layer_pass_count": pass_count,
        "layer_pass_total": pass_total,
        "first_failure": first_failure,
        "layers": {k: layers_full.get(k) for k in EIGHT_LAYERS},
        "expected_calls_count": len(sample.get("expected_tool_calls", []) or []),
        "predicted_calls_count": len(transcript_row.get("extracted_calls", []) or []),
        "expected_answer": sample.get("expected_answer"),
        "predicted_answer": transcript_row.get("generated"),
        "reward_offline_version": REWARD_OFFLINE_VERSION,
    }


def load_samples(samples_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(samples_dir.glob("*.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        sample_id = sample.get("id") or path.stem
        out[sample_id] = sample
    return out


def load_transcripts(transcripts_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(transcripts_path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    return {row.get("sample_id", ""): row for row in rows if row.get("sample_id")}


def aggregate(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample signals into mean ± pop std and layer histogram."""
    if not signals:
        return {"n": 0}
    bin_vals = [s["reward_binary"] for s in signals]
    lay_vals = [s["reward_layered"] for s in signals]
    layer_pass_total = [s["layer_pass_total"] for s in signals]
    first_failure_dist = Counter(s["first_failure"] for s in signals)
    task_dist: Counter[str] = Counter()
    for s in signals:
        task_dist[s.get("task_type") or "unknown"] += 1
    return {
        "n": len(signals),
        "reward_binary_mean": mean(bin_vals),
        "reward_binary_pop_std": pstdev(bin_vals) if len(bin_vals) > 1 else 0.0,
        "reward_layered_mean": mean(lay_vals),
        "reward_layered_pop_std": pstdev(lay_vals) if len(lay_vals) > 1 else 0.0,
        "first_failure_distribution": dict(first_failure_dist),
        "layer_pass_total_mean": mean(layer_pass_total) if layer_pass_total else 0.0,
        "task_type_distribution": dict(task_dist),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--transcripts", required=True, type=Path,
                        help="JSON produced by scripts/eval_sft_tool.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", default=None,
                        help="Optional model checkpoint identifier")
    parser.add_argument("--transcript-kind", default="model_generated",
                        choices=("model_generated", "mock_transcript",
                                 "deterministic_replay"))
    args = parser.parse_args()

    samples = load_samples(args.samples_dir)
    transcripts = load_transcripts(args.transcripts)

    signals: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample_id, sample in samples.items():
        row = transcripts.get(sample_id)
        if row is None:
            missing.append(sample_id)
            continue
        signals.append(compute_reward(
            sample, row,
            transcript_kind=args.transcript_kind,
            checkpoint=args.checkpoint,
        ))

    try:
        samples_rel = args.samples_dir.resolve().relative_to(ROOT.resolve())
    except ValueError:
        samples_rel = args.samples_dir
    try:
        transcripts_rel = args.transcripts.resolve().relative_to(ROOT.resolve())
    except ValueError:
        transcripts_rel = args.transcripts
    out = {
        "reward_offline_version": REWARD_OFFLINE_VERSION,
        "checkpoint": args.checkpoint,
        "transcript_kind": args.transcript_kind,
        "samples_dir": str(samples_rel),
        "transcripts": str(transcripts_rel),
        "aggregate": aggregate(signals),
        "missing_sample_ids": missing,
        "signals": signals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    try:
        out_rel = args.output.resolve().relative_to(ROOT.resolve())
    except ValueError:
        out_rel = args.output
    print(f"[reward-offline] wrote {out_rel}")
    print(f"[reward-offline] n={out['aggregate']['n']} "
          f"reward_binary_mean={out['aggregate']['reward_binary_mean']:.4f} "
          f"reward_layered_mean={out['aggregate']['reward_layered_mean']:.4f}")
    if missing:
        print(f"[reward-offline] missing {len(missing)} sample ids "
              f"(no transcript row): e.g. {missing[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())