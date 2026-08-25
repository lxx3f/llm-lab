"""Run a reproducible smoke benchmark for the Dense Transformer baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.models.dense_transformer import (  # noqa: E402
    DenseTransformer,
    TransformerConfig,
    count_parameters,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    model_settings = settings["model"]
    experiment = settings["experiment"]
    seed = int(experiment["seed"])
    set_seed(seed)

    requested_device = experiment.get("device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    dtype = getattr(torch, str(experiment.get("dtype", "float32")))
    model_config = TransformerConfig(
        vocab_size=int(model_settings["vocab_size"]),
        max_seq_len=int(model_settings["max_seq_len"]),
        d_model=int(model_settings["d_model"]),
        n_heads=int(model_settings["n_heads"]),
        n_layers=int(model_settings["n_layers"]),
        d_ff=int(model_settings["d_ff"]),
        dropout=float(model_settings["dropout"]),
        rope_base=float(model_settings.get("rope_base", 10000.0)),
    )
    model = DenseTransformer(model_config).to(device=device, dtype=dtype).eval()
    batch_size = int(experiment["batch_size"])
    sequence_length = int(experiment["sequence_length"])
    warmup_steps = int(experiment["warmup_steps"])
    benchmark_steps = int(experiment["benchmark_steps"])
    input_ids = torch.randint(
        0,
        model_config.vocab_size,
        (batch_size, sequence_length),
        device=device,
    )

    with torch.no_grad():
        for _ in range(warmup_steps):
            model(input_ids)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(benchmark_steps):
            logits, loss = model(input_ids, labels=input_ids)
        synchronize(device)
        forward_seconds = (time.perf_counter() - start) / benchmark_steps

        cache = [dict() for _ in range(model_config.n_layers)]
        model(input_ids, kv_cache=cache)
        next_token = input_ids[:, -1:]
        synchronize(device)
        start = time.perf_counter()
        for step in range(benchmark_steps):
            model(next_token, start_pos=sequence_length + step, kv_cache=cache)
        synchronize(device)
        decode_seconds = (time.perf_counter() - start) / benchmark_steps

    result = {
        "experiment_id": "dense-baseline-smoke",
        "model": model_settings["name"],
        "architecture": "DenseTransformer",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "config": model_settings,
        "parameter_count": count_parameters(model),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "forward_latency_ms": forward_seconds * 1000,
        "decode_one_token_latency_ms": decode_seconds * 1000,
        "forward_tokens_per_second": batch_size * sequence_length / forward_seconds,
        "decode_tokens_per_second": batch_size / decode_seconds,
        "loss": float(loss),
        "notes": "Smoke benchmark for baseline validation; not a training result.",
    }
    if device.type == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/dense_baseline.example.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.config)
    output = args.output or ROOT / "docs/experiments/dense-baseline/smoke-result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
