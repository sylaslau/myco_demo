import json

import numpy as np
import pandas as pd

from mycodemo.modeling.curves import (grouped_learning_curve, history_filename, notebook_number,
                                      tree_count_history)
from mycodemo.plot_history import plot_runs, save_history, step_column

META = dict(model="harvest_readiness", notebook="3.2.kaggle_harvest_readiness", train_set="abhi",
            test_set="angelo", date="2026-10-14", settings={"epochs": 3, "imgsz": 640})


def _csv(tmp_path, run, with_loss=True, seg=True):
    df = pd.DataFrame({"epoch": [1, 2, 3], "metrics/mAP50(B)": [0.1, 0.2, 0.3]})
    if seg:
        df["metrics/mAP50(M)"] = [0.05, 0.1, 0.2]
    if with_loss:
        df["train/box_loss"] = [2.0, 1.5, 1.2]
    return save_history(df, tmp_path, "trainabhi_testangelo", run=run, **META)


def test_history_file_name_and_columns(tmp_path):
    csv, png = _csv(tmp_path, "run3")
    assert csv.name == "3.2_harvest_readiness_run3_trainabhi_testangelo_20261014.csv"
    assert png.name == csv.with_suffix(".png").name and png.stat().st_size > 0
    assert notebook_number("2.6.kaggle_mould_detection") == "2.6"
    assert history_filename("2.5.x", "a", "r1", "trainx_testy", "2026-10-12") == "2.5_a_r1_trainx_testy_20261012.csv"
    df = pd.read_csv(csv)
    assert list(df.columns[:7]) == ["model", "run", "notebook", "train_set", "test_set", "date", "settings"]
    assert json.loads(df["settings"][0]) == {"epochs": 3, "imgsz": 640}
    assert step_column(df) == "epoch"


def test_plot_runs_overlays_and_skips_missing_columns(tmp_path):
    a, _ = _csv(tmp_path, "run1", with_loss=False)  # like a run recovered from a log: no losses
    b, _ = _csv(tmp_path, "run3", seg=False)  # box-only model: falls back to box mAP50
    out = plot_runs([a, b], out_png=tmp_path / "overlay.png")
    assert out.stat().st_size > 0
    assert plot_runs([a, b], "metrics/mAP50(B)").parent == tmp_path  # saved next to the CSVs


def test_plot_runs_handles_any_step_column(tmp_path):
    trees = pd.DataFrame({"n_trees": [10, 50, 100], "train_score": [0.9, 0.95, 0.96],
                          "val_score": [0.5, 0.6, 0.62]})
    size = pd.DataFrame({"train_size": [10, 20, 40], "train_score": [1.0, 0.9, 0.8],
                         "val_score": [0.2, 0.4, 0.5]})
    meta = {**META, "model": "m", "notebook": "9.9.x"}
    t, _ = save_history(trees, tmp_path, "cv", run="r1", **meta)
    s, _ = save_history(size, tmp_path, "cv", run="r2", **{**meta, "settings": {"score": "R2"}})
    assert step_column(pd.read_csv(t)) == "n_trees" and step_column(pd.read_csv(s)) == "train_size"
    out = plot_runs([t, s], out_png=tmp_path / "mix.png")
    assert out.stat().st_size > 0
    e, _ = _csv(tmp_path, "run9")
    assert plot_runs([e, t], out_png=tmp_path / "all.png").stat().st_size > 0


def test_grouped_learning_curve_keeps_groups_whole():
    data = pd.DataFrame({"g": np.repeat(list("abcd"), 5), "x": np.arange(20.0)})
    seen = []

    def fit_score(train, test):
        seen.append((set(train["g"]), set(test["g"])))
        return float(len(train))

    lc = grouped_learning_curve(data, "g", fit_score, repeats=2)
    assert lc["train_size"].tolist() == [5.0, 10.0, 15.0]  # groups of 5 rows move together
    held = [(tr, te) for tr, te in seen if tr != te]  # the other calls score the training rows
    assert held and all(len(te) == 1 and not (tr & te) for tr, te in held)  # held-out never in training


def test_tree_count_history_grouped_cv():
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=60)
    h = tree_count_history(lambda n: RandomForestRegressor(n, random_state=0), X, y, np.repeat(range(12), 5),
                           counts=(5, 20), n_splits=3)
    assert h["n_trees"].tolist() == [5, 20] and (h["train_score"] > h["val_score"]).all()
