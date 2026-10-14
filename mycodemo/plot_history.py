"""Overlay the per-epoch history CSVs of several runs (metric against epoch).

The CSVs are the ones the Kaggle notebooks save in ``history/`` (see ``modeling/curves.py`` for
the layout): settings columns first, then the Ultralytics ``results.csv`` columns.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Default two panels. Box-only models have no mask column, so the box one is the fallback.
DEFAULT_METRICS = ["metrics/mAP50(M)", "train/box_loss"]
FALLBACK = {"metrics/mAP50(M)": "metrics/mAP50(B)"}


def run_name(df: pd.DataFrame, path) -> str:
    """Legend label: model, run and direction from the file name (date dropped)."""
    stem = Path(path).stem
    head, _, tail = stem.rpartition("_")
    return head if tail.isdigit() else stem


def plot_runs(csv_paths, metric=None, out_png=None) -> Path:
    """One panel per metric, one line per run. ``metric`` is a column name, a list of them, or None.

    A run that lacks the column (e.g. training losses recovered from an old log) is left out of
    that panel. Saves next to the first CSV unless ``out_png`` is given.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric or DEFAULT_METRICS)
    paths = [Path(p) for p in csv_paths]
    runs = [(run_name(df, p), df) for p in paths for df in [pd.read_csv(p)]]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.2), squeeze=False)
    for ax, col in zip(axes[0], metrics):
        for name, df in runs:
            use = col if col in df else FALLBACK.get(col)
            if use in df and df[use].notna().any():
                ax.plot(df["epoch"], df[use], lw=1.8, label=name if use == col else f"{name} (box)")
        ax.set(xlabel="epoch", ylabel=col, title=col)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_png = Path(out_png) if out_png else paths[0].parent / "overlay.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png
