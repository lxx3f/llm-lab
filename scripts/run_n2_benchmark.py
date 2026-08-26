"""Run one Dense or MoE model under an N2 fairness protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.benchmarks.n2 import (  # noqa: E402
    DECODE_CAPACITY_FACTOR,
    PREFILL_CAPACITY_FACTOR,
    benchmark_inputs,
    build_models,
    cache_binding,
    choose_protocol_d_ff,
    load_settings,
    measure_model,
    parameter_counts,
    relative_error,
    resolve_device,
    resolve_dtype,
    set_seed,
    train_for_benchmark,
)
from architecture_lab.experiment_metadata import collect_metadata  # noqa: E402
from architecture_lab.tokenization import BPETokenizer  # noqa: E402
from architecture_lab.benchmarks.results import write_n2_result  # noqa: E402


def benchmark(config_path: Path, protocol: str, architecture: str, output: Path | None = None) -> dict[str, object]:
    settings = load_settings(config_path)
    alignment = choose_protocol_d_ff(settings, protocol, int(settings["model"]["vocab_size"]))
    d_ff = alignment["dense_d_ff"] if architecture == "DenseTransformer" else alignment["moe_d_ff"]
    seed = int(settings["benchmark"]["seed"])
    set_seed(seed)
    device = resolve_device(str(settings["benchmark"].get("device", "auto")))
    dtype = resolve_dtype(str(settings["benchmark"].get("dtype", "float32")))
    tokenizer = BPETokenizer.load(settings["data"]["tokenizer"])
    model, _ = build_models(settings, architecture, d_ff, device, dtype, vocab_size=tokenizer.vocab_size)
    total, active = parameter_counts(model)
    training_metrics = train_for_benchmark(model, settings, architecture=architecture, device=device)
    input_ids = benchmark_inputs(settings, device, split="train")
    measured = measure_model(model, input_ids, architecture=architecture, settings=settings, device=device)
    target = alignment["target_parameters"]
    actual = total if protocol == "A" else active
    result = {
        "schema_version": "1.0",
        "result_type": "architecture_benchmark",
        "experiment_id": f"n2-{protocol.lower()}-{architecture.lower()}",
        "status": "completed",
        "metadata": collect_metadata(
            config_path=config_path,
            tokenizer_artifact_dir=Path(settings["data"]["tokenizer"]).parent,
            train_cache_dir=settings["data"]["train_metadata"],
            seed=seed,
        ),
        "protocol": protocol,
        "model": {
            "name": settings["model"]["name"],
            "architecture": architecture,
            "config": {**settings["model"], "architecture": architecture, "d_ff": d_ff, "vocab_size": tokenizer.vocab_size},
            "moe_config": settings.get("moe") if architecture == "MoETransformer" else None,
            "parameter_count_total": total,
            "parameter_count_active": active,
            "active_parameter_definition": "total_parameters_for_dense; shared_parameters_plus_one_expert_per_moe_layer_for_moe",
        },
        "alignment": {
            "basis": "total_parameters" if protocol == "A" else "active_parameters",
            "target_parameters": target,
            "delta": abs(actual - target),
            "relative_error": relative_error(actual, target),
            "dense_d_ff": alignment["dense_d_ff"],
            "moe_d_ff": alignment["moe_d_ff"],
        },
        "binding": cache_binding(settings, tokenizer),
        "metrics": {
            "train_lm_loss": float(training_metrics["train_lm_loss"]),
            "loss_capacity_factor": PREFILL_CAPACITY_FACTOR if architecture == "MoETransformer" else None,
            "validation_lm_loss": float(training_metrics["validation_lm_loss"]),
            "aux_loss": float(training_metrics["aux_loss"]) if training_metrics["aux_loss"] is not None else None,
            "total_loss": float(training_metrics["total_loss"]),
        },
        "benchmark": {
            "warmup_steps": int(settings["benchmark"]["warmup_steps"]),
            "measured_steps": int(settings["benchmark"]["measured_steps"]),
            "latency_statistic": "mean",
            **measured,
        },
        "routing": {
            "collect_stats": False,
            "prefill_capacity_factor": PREFILL_CAPACITY_FACTOR if architecture == "MoETransformer" else None,
            "decode_capacity_factor": DECODE_CAPACITY_FACTOR if architecture == "MoETransformer" else None,
            "prefill": None,
            "decode": None,
        },
    }
    if output is not None:
        write_n2_result(result, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", choices=("A", "B"), required=True)
    parser.add_argument("--architecture", choices=("DenseTransformer", "MoETransformer"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.config, args.protocol, args.architecture, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
