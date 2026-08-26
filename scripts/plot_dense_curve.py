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


def _plot_train(ax, curve: dict[str, Any], color: str, label: str) -> None:
    """Draw a curve's train_loss series onto the given left-axis ``ax``."""
    train_pts = curve["train_losses"]
    if not train_pts:
        return
    steps = [int(s["step"]) for s in train_pts]
    losses = [float(s["loss"]) for s in train_pts]
    ax.plot(steps, losses, color=color, linewidth=1.5, label=label)


def _plot_val(ax2, curve: dict[str, Any], color: str, label: str, marker: str) -> None:
    """Draw a curve's validation_loss series onto the given right-axis ``ax2``."""
    val_pts = sorted(
        (int(step), float(loss))
        for step, loss in curve["validation_losses"].items()
    )
    if not val_pts:
        return
    vsteps = [s for s, _ in val_pts]
    vlosses = [l for _, l in val_pts]
    ax2.plot(
        vsteps,
        vlosses,
        color=color,
        linewidth=1.5,
        marker=marker,
        markersize=4,
        label=label,
    )
    summary = curve.get("curve_summary") or {}
    if summary.get("val_loss_min_step") is not None:
        ax2.axvline(
            int(summary["val_loss_min_step"]),
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )


def _plot_single(curve: dict[str, Any], output: Path, title_suffix: str = "") -> None:
    """Render a single Dense training curve to its own PNG with twin axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.set_xlabel("step")
    ax.set_ylabel("train_loss", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, alpha=0.3)
    _plot_train(ax, curve, "tab:blue", "train_loss (left axis)")

    has_val = bool(curve.get("validation_losses"))
    if has_val:
        ax2 = ax.twinx()
        ax2.set_ylabel("val_loss", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.grid(False)
        _plot_val(ax2, curve, "tab:orange", "val_loss (right axis)", marker="o")
        # min marker legend entry
        summary = curve.get("curve_summary") or {}
        if summary.get("val_loss_min_step") is not None:
            ax2.plot(
                [],
                [],
                color="tab:orange",
                linestyle="--",
                linewidth=0.8,
                label=f"min_val @ step {int(summary['val_loss_min_step'])}",
            )

    ax.set_title(_title_for(curve, title_suffix), fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    if has_val and len(fig.axes) > 1:
        h2, l2 = fig.axes[1].get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)
    ax.legend(handles, labels, loc="upper right", fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_dense_curve] wrote {output}")


def _plot_overlay(curves: list[dict[str, Any]], output: Path) -> None:
    """Render multiple curves sharing one pair of axes (true overlay).

    Allocation rule (audited): the figure must contain EXACTLY two Axes —
    one left ``ax`` for train_loss and one right ``ax2 = ax.twinx()`` for
    val_loss. Each subsequent run reuses both Axes; ``twinx()`` is called
    at most once. This is enforced by the
    ``test_plot_dense_curve_overlay_uses_exactly_two_axes`` behavioral test.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette_train = ["tab:blue", "tab:green", "tab:purple", "tab:brown"]
    palette_val = ["tab:orange", "tab:red", "tab:olive", "tab:pink"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.set_xlabel("step")
    ax.set_ylabel("train_loss")
    ax.grid(True, alpha=0.3)
    ax2 = None  # created lazily on the first curve that has val data
    for index, curve in enumerate(curves):
        color_train = palette_train[index % len(palette_train)]
        color_val = palette_val[index % len(palette_val)]
        marker = markers[index % len(markers)]
        _plot_train(
            ax,
            curve,
            color_train,
            label=f"train_loss {curve['experiment_id']} (left axis)",
        )
        if curve.get("validation_losses"):
            if ax2 is None:
                ax2 = ax.twinx()
                ax2.set_ylabel("val_loss")
                ax2.grid(False)
            _plot_val(
                ax2,
                curve,
                color_val,
                label=f"val_loss {curve['experiment_id']} (right axis)",
                marker=marker,
            )

    if ax2 is None:
        # No validation data in any input — still need a twin for symmetry
        ax2 = ax.twinx()
        ax2.set_ylabel("val_loss")

    ax.set_title(
        "Overlay: " + " vs ".join(c["experiment_id"] for c in curves),
        fontsize=9,
    )

    handles, labels = ax.get_legend_handles_labels()
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