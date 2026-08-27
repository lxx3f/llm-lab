"""SFT tool-calling training CLI.

Loads D1 + D1.1 tool-calling samples, builds masked SFT sequences, and
trains a DenseTransformer to emit tool calls + final answers.

Config (see configs/sft.example.yaml):
    data:
      sample_dirs: [datasets/tool-calling-d1, datasets/tool-calling-d1-llm]
      tokenizer: artifacts/tokenizers/owt-bpe/v0.2.0/tokenizer.json
    model: (same keys as dense_training: d_model/n_layers/n_heads/d_ff/...)
    training:
      seed: 42
      batch_size: 8
      sequence_length: 256
      learning_rate: 5e-5
      max_steps: 200
      scheduler: {warmup_steps: 20, min_lr_ratio: 0.1}
      checkpoint: artifacts/checkpoints/sft-tool-v1.pt

Usage:
    .venv/python.exe scripts/train_sft.py --config configs/sft.example.yaml
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

from architecture_lab.training.dense_training import resolve_device, resolve_dtype  # noqa: E402
from architecture_lab.training.sft_training import train_sft  # noqa: E402


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
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path,
                        help="pre-trained Dense checkpoint to initialize weights from "
                             "(e.g. dense-owt-formal-curve-medium.pt)")
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
    print(f"[sft] loaded {len(samples)} samples from {data['sample_dirs']}", flush=True)

    # Optional deterministic augmentation to combat overfitting on the small
    # tool-calling set (252 samples).
    aug_k = int(data.get("augment_k", 0))
    if aug_k > 0:
        from architecture_lab.training.sft_augment import augment_samples  # noqa: PLC0415
        samples = augment_samples(samples, k=aug_k, seed=int(training.get("seed", 42)))
        print(f"[sft] augmented to {len(samples)} samples (k={aug_k})", flush=True)

    result = train_sft(
        samples=samples,
        tokenizer_path=Path(data["tokenizer"]),
        model_settings=settings.get("model", {}),
        training=training,
        device=device,
        dtype=dtype,
        checkpoint_out=Path(training["checkpoint"]),
        init_checkpoint=args.init_checkpoint,
    )
    # Metadata (N3-compatible subset).
    import architecture_lab.experiment_metadata as em  # noqa: PLC0415
    result["metadata"] = em.collect_metadata(
        config_path=args.config,
        tokenizer_artifact_dir=Path(data["tokenizer"]).parent,
        train_cache_dir=Path(data.get("sample_dirs", [""])[0]),
        seed=int(training.get("seed", 42)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
