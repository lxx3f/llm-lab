"""Train a Top-1 MoE Transformer and write a validated result record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_lab.training.moe_training import load_settings, train  # noqa: E402
from architecture_lab.training.moe_results import write_moe_training_result  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = train(load_settings(args.config), resume=args.resume)
    write_moe_training_result(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
