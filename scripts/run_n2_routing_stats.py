"""Collect routing statistics outside the N2 latency benchmark path."""

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
    aggregate_routing,
    benchmark_inputs,
    build_models,
    cache_binding,
    load_settings,
    resolve_device,
    resolve_dtype,
    set_seed,
)
from architecture_lab.benchmarks.results import write_routing_result  # noqa: E402
from architecture_lab.experiment_metadata import collect_metadata  # noqa: E402
from architecture_lab.tokenization import BPETokenizer  # noqa: E402


def collect(config_path: Path, architecture: str, output: Path | None = None) -> dict[str, object]:
    settings = load_settings(config_path)
    seed = int(settings["benchmark"]["seed"])
    set_seed(seed)
    device = resolve_device(str(settings["benchmark"].get("device", "auto")))
    dtype = resolve_dtype(str(settings["benchmark"].get("dtype", "float32")))
    d_ff = int(settings["model"]["d_ff"])
    tokenizer = BPETokenizer.load(settings["data"]["tokenizer"])
    # Validate both train and validation bindings even though routing metrics use
    # the train batch, keeping this analysis path on the same data contract.
    cache_binding(settings, tokenizer)
    model, _ = build_models(settings, architecture, d_ff, device, dtype, vocab_size=tokenizer.vocab_size)
    input_ids = benchmark_inputs(settings, device)
    with torch.no_grad():
        if architecture == "DenseTransformer":
            stats = None
        else:
            cache = [dict() for _ in range(model.config.n_layers)]
            model(input_ids, kv_cache=cache, collect_stats=True, capacity_factor=PREFILL_CAPACITY_FACTOR)
            prefill_layers = list(model.last_routing_stats or [])
            model(input_ids[:, -1:], start_pos=input_ids.size(1), kv_cache=cache, collect_stats=True, capacity_factor=DECODE_CAPACITY_FACTOR)
            decode_layers = list(model.last_routing_stats or [])
            stats = {
                "prefill": aggregate_routing(prefill_layers),
                "decode": aggregate_routing(decode_layers),
            }
    result = {
        "schema_version": "1.0",
        "result_type": "n2_routing_stats",
        "architecture": architecture,
        "collect_stats": True,
        "prefill_capacity_factor": PREFILL_CAPACITY_FACTOR,
        "decode_capacity_factor": DECODE_CAPACITY_FACTOR,
        "metadata": collect_metadata(
            config_path=config_path,
            tokenizer_artifact_dir=Path(settings["data"]["tokenizer"]).parent,
            train_cache_dir=settings["data"]["train_metadata"],
            seed=seed,
        ),
        "stats": stats,
    }
    if output is not None:
        write_routing_result(result, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--architecture", choices=("DenseTransformer", "MoETransformer"), default="MoETransformer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.config, args.architecture, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
