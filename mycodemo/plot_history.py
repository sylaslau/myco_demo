"""Overlay the history CSVs of several runs (score against step), for any model type.

One CSV per run, one row per step. The front columns are the same for every model (see
``modeling/curves.py``); the step column says what the rows are:

    epoch       YOLO / neural nets (Ultralytics ``results.csv`` columns)
    n_trees     random forest / gradient boosting (``train_score``, ``val_score``)
    train_size  learning curve of small models and curve fits (``train_score``, ``val_score``)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mycodemo.modeling.curves import STEP_COLUMNS, save_history_csv

# Default two panels. Box-only models have no mask column, so the box one is the fallback.
DEFAULT_METRICS = ["metrics/mAP50(M)", "train/box_loss"]
STEP_METRICS = ["train_score", "val_score"]  # one panel, both lines, for n_trees / train_size runs
FALLBACK = {"metrics/mAP50(M)": "metrics/mAP50(B)"}


def step_column(df: pd.DataFrame):
    """The step column of a history table (epoch, n_trees or train_size), or None."""
    return next((c for c in STEP_COLUMNS if c in df), None)


def run_name(df: pd.DataFrame, path) -> str:
    """Legend label: number, model, run and direction from the file name (date dropped)."""
    stem = Path(path).stem
    head, _, tail = stem.rpartition("_")
    return head if tail.isdigit() else stem


def _score_label(df: pd.DataFrame) -> str:
    """Name of the score from the ``settings`` column (e.g. "RMSE, lower is better"), else 'score'."""
    try:
        return json.loads(df["settings"].iloc[0]).get("score", "score")
    except (KeyError, TypeError, ValueError, IndexError):
        return "score"


def plot_runs(csv_paths, metric=None, out_png=None) -> Path:
    """One panel per metric, one line per run. ``metric`` is a column name, a list of them, or None.

    With None: runs with an epoch column get the YOLO panels, runs with n_trees or train_size get one
    score panel (train and validation). The x axis is labelled from the step column of the runs in
    the panel. A run that lacks the column is left out of that panel. Saves next to the first CSV
    unless ``out_png`` is given.
    """
    paths = [Path(p) for p in csv_paths]
    runs = [(run_name(df, p), df) for p in paths for df in [pd.read_csv(p)]]
    if metric is None:
        metrics = []
        if any(step_column(d) == "epoch" for _, d in runs):
            metrics += DEFAULT_METRICS
        if any(step_column(d) in ("n_trees", "train_size") for _, d in runs):
            metrics += ["score"]
        metrics = metrics or DEFAULT_METRICS
    else:
        metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.2), squeeze=False)
    for ax, col in zip(axes[0], metrics):
        steps, ylabels = [], []
        for name, df in runs:
            x = step_column(df)
            if x is None:
                continue
            if col == "score":  # train (dashed) and validation (solid) of one run
                if x == "epoch":
                    continue
                for c, ls, tag in [("val_score", "-", "validation"), ("train_score", "--", "train")]:
                    if c in df and df[c].notna().any():
                        ax.plot(df[x], df[c], ls=ls, lw=1.8, marker="o", ms=3, label=f"{name} ({tag})")
                steps.append(STEP_COLUMNS[x])
                ylabels.append(_score_label(df))
                continue
            use = col if col in df else FALLBACK.get(col)
            if use in df and df[use].notna().any():
                ax.plot(df[x], df[use], lw=1.8, label=name if use == col else f"{name} (box)")
                steps.append(STEP_COLUMNS[x])
        xlabel = " / ".join(dict.fromkeys(steps)) or "step"
        ylabel = " / ".join(dict.fromkeys(ylabels)) if col == "score" and ylabels else col
        ax.set(xlabel=xlabel, ylabel=ylabel, title="train and validation score" if col == "score" else col)
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


def save_history(df: pd.DataFrame, out_dir, direction: str, **meta) -> tuple[Path, Path]:
    """Write one run's history CSV and its PNG (same name) into ``out_dir``; returns both paths.

    ``meta``: model, run, notebook, train_set, test_set, date, settings (see ``with_history_columns``).
    """
    csv = save_history_csv(df, out_dir, direction, **meta)
    return csv, plot_runs([csv], out_png=csv.with_suffix(".png"))
