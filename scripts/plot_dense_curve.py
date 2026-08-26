"""Plot a Dense Transformer training curve from a JSON result file.

Reads one or two N4-era ``dense_training_result`` JSON files and produces a
matplotlib PNG with:

- left y-axis: sampled train_loss vs step (blue solid line)
- right y-axis: validation_loss vs step (orange solid line, marker at min)

The script is intentionally minimal — it does not import any
architecture_lab/* symbols so it can be exercised without GPU.

Typical usage::

    .venv/python.exe scripts/plot_dense_curve.py \\
        --input artifacts/dense-owt-formal-curve-result.json \\
        --output artifacts/dense-owt-formal-curve.png

Two-input variant (``--overlay`` puts both curves on the **same** axes,
sharing the dual y-axis layout for a true baseline-vs-medium comparison)::

    .venv/python.exe scripts/plot_dense_curve.py \\
        --input artifacts/dense-owt-formal-curve-result.json \\
        --input artifacts/dense-owt-formal-curve-medium-result.json \\
        --output artifacts/dense-owt-formal-curve-overlay.png \\
        --overlay
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


def _title_for(curve: dict[str, Any], title_suffix: str = "") -> str:
    title_parts = [curve["experiment_id"]]
    if title_suffix:
        title_parts.append(title_suffix)
    if curve["parameter_count"]:
        title_parts.append(f"params={curve['parameter_count']:,}")
    last_train = (curve["curve_summary"] or {}).get("train_loss_last")
    min_val = (curve["curve_summary"] or {}).get("val_loss_min")
    if last_train is not None:
        title_parts.append(f"last_train={last_train:.4f}")
    if min_val is not None:
        title_parts.append(f"min_val={min_val:.4f}")
    return " | ".join(title_parts)


def _draw_dual_axis(ax, curve: dict[str, Any], color_train: str = "tab:blue", color_val: str = "tab:orange") -> Any:
    """Draw a single curve's train_loss (left axis) and val_loss (right axis).

    Returns the right-hand ``Axes`` so callers (overlay mode) can keep adding
    validation series onto the same twin axes.
    """
    train_pts = curve["train_losses"]
    val_pts = sorted(
        (int(step), float(loss))
        for step, loss in curve["validation_losses"].items()
    )
    summary = curve["curve_summary"] or {}

    ax.set_xlabel("step")
    ax.set_ylabel("train_loss", color=color_train)
    ax.tick_params(axis="y", labelcolor=color_train)

    if train_pts:
        steps = [int(s["step"]) for s in train_pts]
        losses = [float(s["loss"]) for s in train_pts]
        ax.plot(steps, losses, color=color_train, linewidth=1.5, label="train_loss (left axis)")

    if val_pts:
        ax2 = ax.twinx()
        ax2.set_ylabel("val_loss", color=color_val)
        ax2.tick_params(axis="y", labelcolor=color_val)
        vsteps = [s for s, _ in val_pts]
        vlosses = [l for _, l in val_pts]
        ax2.plot(
            vsteps,
            vlosses,
            color=color_val,
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="val_loss (right axis)",
        )
        if summary.get("val_loss_min_step") is not None:
            ax2.axvline(
                int(summary["val_loss_min_step"]),
                color=color_val,
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                label=f"min_val @ step {int(summary['val_loss_min_step'])}",
            )
        ax2.grid(False)  # avoid double-grid
        return ax2
    return None


def _plot_single(curve: dict[str, Any], output: Path, title_suffix: str = "") -> None:
    """Render a single Dense training curve to its own PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    _draw_dual_axis(ax, curve)
    ax.set_title(_title_for(curve, title_suffix), fontsize=9)
    ax.grid(True, alpha=0.3)

    # Combined legend (both axes' artists)
    handles_train, labels_train = ax.get_legend_handles_labels()
    ax2_artist = None
    for other in fig.axes[1:]:
        h2, l2 = other.get_legend_handles_labels()
        if any("val" in lab for lab in l2):
            handles_train.extend(h2)
            labels_train.extend(l2)
            ax2_artist = other
            break
    ax.legend(handles_train, labels_train, loc="upper right", fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_dense_curve] wrote {output}")


def _plot_overlay(curves: list[dict[str, Any]], output: Path) -> None:
    """Render multiple curves sharing one axes pair (true overlay).

    Both runs are drawn on the same ``ax`` (left axis: train_loss) and the
    same ``ax.twinx()`` (right axis: val_loss). Each run gets a distinct
    color so the curves can be told apart in the legend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette_train = ["tab:blue", "tab:green", "tab:purple", "tab:brown"]
    palette_val = ["tab:orange", "tab:red", "tab:olive", "tab:pink"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax2 = None
    for index, curve in enumerate(curves):
        color_train = palette_train[index % len(palette_train)]
        color_val = palette_val[index % len(palette_val)]
        # First curve creates the twin axes; subsequent curves reuse it.
        ax2_candidate = _draw_dual_axis(
            ax,
            curve,
            color_train=color_train,
            color_val=color_val,
        )
        if ax2 is None and ax2_candidate is not None:
            ax2 = ax2_candidate
        elif ax2 is not None:
            # Re-draw the val series onto the shared ax2 with this run's color
            val_pts = sorted(
                (int(step), float(loss))
                for step, loss in curve["validation_losses"].items()
            )
            if val_pts:
                vsteps = [s for s, _ in val_pts]
                vlosses = [l for _, l in val_pts]
                ax2.plot(
                    vsteps,
                    vlosses,
                    color=color_val,
                    linewidth=1.5,
                    marker="s" if index == 1 else "o",
                    markersize=4,
                    label=f"val_loss {curve['experiment_id']} (right axis)",
                )
            summary = curve.get("curve_summary") or {}
            if summary.get("val_loss_min_step") is not None:
                ax2.axvline(
                    int(summary["val_loss_min_step"]),
                    color=color_val,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.5,
                )

    ax.set_title(
        "Overlay: " + " vs ".join(c["experiment_id"] for c in curves),
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)
    ax.legend(handles, labels, loc="upper right", fontsize=7)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_dense_curve] wrote {output}")


def _plot(inputs: list[Path], output: Path, overlay: bool) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plot_dense_curve.py; "
            "install it via `pip install matplotlib`."
        ) from exc

    curves = [_load_curve(p) for p in inputs]

    if overlay:
        if len(curves) < 2:
            raise SystemExit("--overlay requires at least two --input files")
        _plot_overlay(curves, output)
    else:
        if len(curves) > 1:
            print(
                f"[plot_dense_curve] got {len(curves)} inputs but --overlay not set; "
                "plotting the first one only. Re-run with --overlay to combine.",
                file=sys.stderr,
            )
        _plot_single(curves[0], output)


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
        help="When ≥2 inputs are given, render both curves on shared axes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    _plot(list(args.input), args.output, overlay=args.overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())