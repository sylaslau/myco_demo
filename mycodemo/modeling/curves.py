"""Training curves: the score and loss of every epoch, kept as a table and a picture.

Ultralytics (YOLO) already writes ``results.csv`` (one row per epoch) in each run folder.
This module reads that file, plots it, and saves a small bundle per model version:

    reports/training_runs/<run_id>/epochs.csv      one row per epoch
    reports/training_runs/<run_id>/curves.png      the graph
    reports/training_runs/<run_id>/settings.json   model, epochs, image size, best epoch, notes

For older runs where only the printed Kaggle log survives, ``parse_console_log`` recovers the
validation scores of each epoch (the training losses are not recoverable from such a log).
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
# One validation row. Kaggle log lines start with "12.3s 45 "; plain console output does not.
ROW = re.compile(r"^(?:[\d.]+s \d+\s+)?all\s+(\d+)\s+(\d+)\s+((?:[\d.]+(?:e-?\d+)?\s*){4,8})$")
START = re.compile(r"Starting training for (\d+) epochs")

BOX = ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]
MASK = ["precision(M)", "recall(M)", "mAP50(M)", "mAP50-95(M)"]
METRIC_COLS = ["metrics/" + c for c in BOX + MASK]
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#6b6b6b"  # colour-blind-safe pair + neutral


def read_results_csv(path) -> pd.DataFrame:
    """Read an Ultralytics ``results.csv``; column names come with stray spaces."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_console_log(path) -> list[pd.DataFrame]:
    """Recover per-epoch validation scores from a saved console log, one table per training run.

    Only the rows printed after each epoch are used, so the table has precision, recall,
    mAP50 and mAP50-95 (box, and mask for segmentation models) but no training losses.
    """
    runs, rows, n_epochs = [], [], None
    for line in Path(path).read_text(errors="replace").splitlines():
        line = ANSI.sub("", line).rstrip()
        m = START.search(line)
        if m:
            if rows:
                runs.append(rows)
            rows, n_epochs = [], int(m.group(1))
            continue
        m = ROW.match(line.strip())
        if m and n_epochs is not None and len(rows) < n_epochs:
            rows.append([float(v) for v in m.group(3).split()])
    if rows:
        runs.append(rows)
    tables = []
    for rows in runs:
        names = METRIC_COLS[: len(rows[0])]
        df = pd.DataFrame(rows, columns=names)
        df.insert(0, "epoch", range(1, len(df) + 1))
        tables.append(df)
    return tables


def best_epoch(df: pd.DataFrame) -> int:
    """Epoch Ultralytics keeps as best.pt: highest 0.1*mAP50 + 0.9*mAP50-95 (box + mask summed)."""
    fit = sum(
        0.1 * df[f"metrics/mAP50({k})"] + 0.9 * df[f"metrics/mAP50-95({k})"]
        for k in "BM"
        if f"metrics/mAP50({k})" in df
    )
    return int(df.loc[fit.idxmax(), "epoch"])


def plot_curves(df: pd.DataFrame, out_png, title: str = "") -> Path:
    """Left: training and validation loss (when present). Right: validation mAP and recall."""
    losses = [c for c in df.columns if c.startswith("train/") and c.endswith("_loss")]
    has_loss = bool(losses)
    fig, axes = plt.subplots(1, 2 if has_loss else 1, figsize=(12 if has_loss else 6.5, 4.2))
    axes = [axes] if not has_loss else list(axes)
    best = best_epoch(df)
    ax = axes[-1]
    series = [
        ("metrics/mAP50(B)", "box mAP50", BLUE, "-"),
        ("metrics/mAP50-95(B)", "box mAP50-95", BLUE, "--"),
        ("metrics/mAP50(M)", "mask mAP50", ORANGE, "-"),
        ("metrics/mAP50-95(M)", "mask mAP50-95", ORANGE, "--"),
    ]
    for col, label, colour, style in series:
        if col in df:
            ax.plot(df["epoch"], df[col], color=colour, ls=style, lw=2, label=label)
    ax.axvline(best, color=GREY, lw=1, ls=":")
    ax.annotate(f"best.pt = epoch {best}", (best, 1.0), xytext=(-4, -12),
                textcoords="offset points", color=GREY, fontsize=9, ha="right")
    ax.set(xlabel="epoch", ylabel="score on the check photos (0 to 1)", ylim=(0, 1),
           title="Check score per epoch")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    if has_loss:
        ax = axes[0]
        for col, ls in [(c, "-") for c in losses] + [
            (c, "--") for c in df.columns if c.startswith("val/") and c.endswith("_loss")
        ]:
            ax.plot(df["epoch"], df[col], ls=ls, lw=1.6, label=col.replace("_loss", ""))
        ax.set(xlabel="epoch", ylabel="loss", title="Loss per epoch (solid train, dashed check)")
        ax.legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def save_run(df: pd.DataFrame, out_dir, run_id: str, settings: dict | None = None) -> Path:
    """Write epochs.csv, curves.png and settings.json for one model version."""
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "epochs.csv", index=False)
    plot_curves(df, out / "curves.png", title=run_id)
    info = {"run_id": run_id, "epochs_run": int(df["epoch"].max()), "best_epoch": best_epoch(df)}
    (out / "settings.json").write_text(json.dumps({**info, **(settings or {})}, indent=2))
    return out


# --- History CSV: one file per run, the same layout the Kaggle notebooks write ----------------
HISTORY_FRONT = ["model", "run", "train_set", "test_set", "date", "epochs", "imgsz", "batch", "base_model"]
_LOSSES = ["train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss"]
_VAL_LOSSES = ["val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss"]


def history_filename(model: str, run: str, direction: str, date: str) -> str:
    """<model>_<run>_<direction>_<YYYYMMDD>.csv, e.g. harvest_readiness_run3_trainothers_testangelo_20261014.csv."""
    parts = [re.sub(r"[^\w-]+", "", p) for p in (model, run, direction, date.replace("-", ""))]
    return "_".join(parts) + ".csv"


def with_history_columns(df: pd.DataFrame, *, model, run, train_set, test_set, date, epochs, imgsz, batch,
                         base_model) -> pd.DataFrame:
    """Put the run's settings in front of the per-epoch table; ``date`` is ISO (2026-10-12)."""
    front = dict(model=model, run=run, train_set=train_set, test_set=test_set, date=date,
                 epochs=epochs, imgsz=imgsz, batch=batch, base_model=base_model)
    out = df.copy()
    for i, (k, v) in enumerate(front.items()):
        out.insert(i, k, v)
    return out


def log_table_to_results_layout(df: pd.DataFrame) -> pd.DataFrame:
    """Give a parsed log table the full results.csv columns; what a log cannot hold stays empty."""
    seg = "metrics/mAP50(M)" in df
    cols = ["epoch", *_LOSSES, *METRIC_COLS[: 8 if seg else 4], *_VAL_LOSSES]
    if not seg:
        cols = [c for c in cols if "seg_loss" not in c]
    return df.reindex(columns=cols)


def save_history_csv(df: pd.DataFrame, out_dir, direction: str, **meta) -> Path:
    """Write one history CSV (settings columns in front) into out_dir; returns its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / history_filename(meta["model"], meta["run"], direction, meta["date"])
    with_history_columns(df, **meta).to_csv(path, index=False)
    return path


# --- Non-YOLO models: fold scores, learning curve, error against a growing input --------------
def training_run_dir(run_id: str, out_dir=None) -> Path:
    """reports/training_runs/<run_id>/ (created), the same place save_run writes to."""
    from mycodemo.config import REPORTS_DIR

    out = Path(out_dir) if out_dir else REPORTS_DIR / "training_runs"
    out = out / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def learning_curve(folds, score_fn, fractions=(0.2, 0.4, 0.6, 0.8, 1.0), repeats=5, seed=0) -> pd.DataFrame:
    """Score against training size. ``folds`` is a list of (name, train_df, test_df);
    ``score_fn(train, test)`` returns one number. Each fraction draws ``repeats`` random
    subsets of the training rows (one draw at 1.0), so the curve shows how much more data helps."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, train, test in folds:
        for frac in fractions:
            n = max(2, int(round(frac * len(train))))
            for rep in range(1 if frac >= 1 else repeats):
                sub = train if frac >= 1 else train.iloc[rng.choice(len(train), n, replace=False)]
                rows.append({"fold": name, "fraction": frac, "n_train": len(sub), "repeat": rep,
                             "score": score_fn(sub, test)})
    return pd.DataFrame(rows)


def plot_lines(table: pd.DataFrame, x: str, ys: list[str], out_png, xlabel="", ylabel="", title="") -> Path:
    """One line per column in ``ys`` against ``x`` (table sorted by x)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    t = table.sort_values(x)
    for col, colour in zip(ys, [BLUE, ORANGE, GREY]):
        ax.plot(t[x], t[col], marker="o", lw=2, color=colour, label=col)
    ax.set(xlabel=xlabel or x, ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def plot_learning_curve(lc: pd.DataFrame, out_png, ylabel="score") -> Path:
    """Mean score over the repeats and folds against number of training rows."""
    mean = lc.groupby("fraction").agg(n_train=("n_train", "mean"), score=("score", "mean"))
    return plot_lines(mean.reset_index(), "n_train", ["score"], out_png, "training rows", ylabel,
                      "Learning curve")


def plot_fold_scores(scores: pd.DataFrame, out_png, fold="held_out_lab", model="model", value="fold_error") -> Path:
    """Grouped bars: one group per fold, one bar per model."""
    ax = scores.pivot(index=fold, columns=model, values=value).plot.bar(figsize=(7, 4))
    ax.set(ylabel=value, xlabel=fold, title="Score per cross-validation fold")
    ax.spines[["top", "right"]].set_visible(False)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(ax.figure)
    return out_png
