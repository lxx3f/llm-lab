"""Run the reproducible Dense Transformer training MVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.experiment_metadata import collect_metadata
from architecture_lab.training.dense_training import load_settings, train
from architecture_lab.training.results import write_training_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = load_settings(args.config)
    # Per N3 contract, this CLI invokes collect_metadata() with the actual
    # settings-derived tokenizer / train-cache paths and seed, then writes
    # the returned metadata block into the result dict before persisting.
    result = train(settings, resume=args.resume)
    result["metadata"] = collect_metadata(
        config_path=args.config,
        tokenizer_artifact_dir=Path(settings["data"]["tokenizer"]).parent,
        train_cache_dir=Path(settings["data"]["train_metadata"]),
        seed=int(settings["training"].get("seed", 42)),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        write_training_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())