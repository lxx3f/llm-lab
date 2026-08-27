"""SFT tool-calling training CLI for Top-1 MoE Transformer.

Config (see configs/sft-moe.example.yaml) carries an extra
``moe_num_experts`` and ``moe_aux_loss_coeff`` under ``model:``; the rest
mirrors ``configs/sft-large.example.yaml``.

Usage:
    .venv/python.exe scripts/train_sft_moe.py --config configs/sft-moe.example.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from architecture_lab.experiment_metadata import collect_metadata  # noqa: E402
from architecture_lab.training.dense_training import resolve_device, resolve_dtype  # noqa: E402
from architecture_lab.training.sft_augment import augment_samples  # noqa: E402
from architecture_lab.training.sft_moe_training import train_sft_moe  # noqa: E402


def load_samples(sample_dirs: list[str]) -> list[dict]:
    samples: list[dict] = []
    for rel in sample_dirs:
        d = ROOT / rel
        if not d.is_dir():
            raise ValueError(f"sample dir not found: {d}")
        for split in ("train", "dev", "test"):
            split_dir = d / split
            if not split_dir.is_dir():
                continue
            for path in sorted(split_dir.glob("*.json")):
                samples.append(json.loads(path.read_text(encoding="utf-8")))
    if not samples:
        raise ValueError("no samples loaded")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()

    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = settings["data"]
    training = settings["training"]
    requested_device = str(training.get("device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    device = resolve_device(requested_device)
    dtype = resolve_dtype(str(training.get("dtype", "float32")))

    samples = load_samples(data["sample_dirs"])
    print(f"[sft-moe] loaded {len(samples)} samples from {data['sample_dirs']}",
          flush=True)
    aug_k = int(data.get("augment_k", 0))
    if aug_k > 0:
        samples = augment_samples(samples, k=aug_k,
                                  seed=int(training.get("seed", 42)))
        print(f"[sft-moe] augmented to {len(samples)} samples (k={aug_k})",
              flush=True)

    result = train_sft_moe(
        samples=samples,
        tokenizer_path=Path(data["tokenizer"]),
        model_settings=settings.get("model", {}),
        training=training,
        device=device,
        dtype=dtype,
        checkpoint_out=Path(training["checkpoint"]),
        init_checkpoint=args.init_checkpoint,
    )
    result["metadata"] = collect_metadata(
        config_path=args.config,
        tokenizer_artifact_dir=Path(data["tokenizer"]).parent,
        train_cache_dir=Path(data.get("sample_dirs", [""])[0]),
        seed=int(training.get("seed", 42)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())