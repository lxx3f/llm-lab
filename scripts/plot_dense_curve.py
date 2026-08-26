"""Plot a Dense Transformer training curve from a JSON result file.

Reads one or two N4-era ``dense_training_result`` JSON files and produces a
matplotlib PNG with:

- left axis: sampled train_loss vs step (blue solid line)
- right axis: validation_loss vs step (orange solid line, marker at min)
- title: experiment_id + summary numbers (last train_loss, min val_loss)

The script is intentionally minimal — it does not import any
architecture_lab/* symbols so it can be exercised without GPU.

Typical usage::

    .venv/python.exe scripts/plot_dense_curve.py \
        --input artifacts/dense-owt-formal-curve-result.json \
        --output artifacts/dense-owt-formal-curve.png

Two-input variant (overlay two runs on a single figure)::

    .venv/python.exe scripts/plot_dense_curve.py \
        --input artifacts/dense-owt-formal-curve-result.json \
        --input artifacts/dense-owt-formal-curve-medium-result.json \
        --output artifacts/dense-owt-formal-curve-overlay.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_curve(path: Path) -> dict[str, Any]:
    """Read a JSON result file and extract the curve series."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in payload:
        raise ValueError(f"{path}: missing 'metrics' block (not a Dense training result?)")
    metrics = payload["metrics"]
    train_losses = metrics.get("train_losses") or []
    val_losses = metrics.get("validation_losses") or {}
    summary = metrics.get("curve_summary") or {}
    return {
        "experiment_id": payload.get("experiment_id", path.stem),
        "parameter_count": (payload.get("model") or {}).get("parameter_count"),
        "train_losses": train_losses,
        "validation_losses": val_losses,
        "curve_summary": summary,
    }


def _draw_single(curve: dict[str, Any], ax, title_suffix: str = "") -> None:
    """Draw a single curve onto the given matplotlib Axes."""
    try:
        import matplotlib.pyplot as _plt  # noqa: F401  (validate import)
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plot_dense_curve.py; "
            "install it via `pip install matplotlib`."
        ) from exc

    train_pts = curve["train_losses"]
    val_pts = sorted(
        (int(step), float(loss))
        for step, loss in curve["validation_losses"].items()
    )
    summary = curve["curve_summary"]
    last_train = summary.get("train_loss_last")
    min_val = summary.get("val_loss_min")

    title_parts = [curve["experiment_id"]]
    if title_suffix:
        title_parts.append(title_suffix)
    if curve["parameter_count"]:
        title_parts.append(f"params={curve['parameter_count']:,}")
    if last_train is not None:
        title_parts.append(f"last_train={last_train:.4f}")
    if min_val is not None:
        title_parts.append(f"min_val={min_val:.4f}")
    ax.set_title(" | ".join(title_parts), fontsize=9)

    if train_pts:
        steps = [int(s["step"]) for s in train_pts]
        losses = [float(s["loss"]) for s in train_pts]
        ax.plot(steps, losses, color="tab:blue", linewidth=1.5, label="train_loss")
    if val_pts:
        vsteps = [s for s, _ in val_pts]
        vlosses = [l for _, l in val_pts]
        ax.plot(
            vsteps,
            vlosses,
            color="tab:orange",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="val_loss",
        )
        if summary.get("val_loss_min_step") is not None:
            ax.axvline(
                int(summary["val_loss_min_step"]),
                color="tab:orange",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                label=f"min_val @ step {int(summary['val_loss_min_step'])}",
            )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def _plot(inputs: list[Path], output: Path, overlay: bool) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plot_dense_curve.py; "
            "install it via `pip install matplotlib`."
        ) from exc

    curves = [_load_curve(p) for p in inputs]

    if overlay:
        if len(curves) < 2:
            raise SystemExit("--overlay requires at least two --input files")
        fig, axes = plt.subplots(1, len(curves), figsize=(8 * len(curves), 5), dpi=120)
        if len(curves) == 1:
            axes = [axes]
        for curve, ax in zip(curves, axes):
            _draw_single(curve, ax)
        fig.suptitle(
            f"Dense training curves: {len(curves)} runs (overlay)",
            fontsize=11,
        )
    else:
        if len(curves) > 1:
            print(
                f"[plot_dense_curve] got {len(curves)} inputs but --overlay not set; "
                "plotting the first one only. Re-run with --overlay to combine.",
                file=sys.stderr,
            )
        curve = curves[0]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        _draw_single(curve, ax)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_dense_curve] wrote {output}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Path to a Dense training result JSON. May be repeated for overlay.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the PNG figure to.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="When ≥2 inputs are given, render side-by-side subplots.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    _plot(list(args.input), args.output, overlay=args.overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())