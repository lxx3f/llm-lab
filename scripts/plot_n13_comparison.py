"""Generate 3-way comparison plot for N13 (Dense MHA vs GQA vs MLA).

Reads train_losses from each result JSON and plots them on one figure
with different colors per architecture. Saves PNG to docs/experiments/n13-gqa-vs-mla-vs-mha/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.stdin and False:  # noqa
    pass
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_train_losses(path: Path) -> tuple[list[int], list[float]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    steps = [s["step"] for s in d["metrics"]["train_losses"]]
    losses = [s["loss"] for s in d["metrics"]["train_losses"]]
    return steps, losses


def load_validation(path: Path) -> tuple[list[int], list[float]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for k, v in d["metrics"]["validation_losses"].items():
        try:
            items.append((int(k), float(v)))
        except (TypeError, ValueError):
            continue
    items.sort()
    if not items:
        return [], []
    steps, losses = zip(*items)
    return list(steps), list(losses)


def main() -> int:
    out_dir = ROOT / "docs" / "experiments" / "n13-gqa-vs-mla-vs-mha"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        (
            "Dense MHA (baseline)",
            ROOT / "artifacts" / "dense-owt-formal-curve-medium-result.json",
            "#1f77b4",
        ),
        (
            "GQA (num_kv_heads=1)",
            ROOT / "artifacts" / "gqa-owt-formal-curve-medium-result.json",
            "#ff7f0e",
        ),
        (
            "MLA (latent_dim=64)",
            ROOT / "artifacts" / "mla-owt-formal-curve-medium-result.json",
            "#2ca02c",
        ),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Train loss
    for label, path, color in sources:
        steps, losses = load_train_losses(path)
        ax1.plot(steps, losses, label=label, color=color, alpha=0.7, linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train loss (nats)")
    ax1.set_title("N13: train_loss curves (Dense MHA vs GQA vs MLA, 2.10M, 5000 steps)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    # Validation loss
    for label, path, color in sources:
        steps, losses = load_validation(path)
        ax2.plot(steps, losses, label=label, color=color, alpha=0.8, linewidth=2.0, marker="o", markersize=4)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Validation loss (nats)")
    ax2.set_title("N13: validation_loss curves (every 200 steps)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "dense-vs-gqa-vs-mla-curves.png"
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
