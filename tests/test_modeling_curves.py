import pandas as pd

from mycodemo.modeling.curves import best_epoch, parse_console_log, read_results_csv, save_run

LOG = """\
1.0s 1 Starting training for 2 epochs...
2.0s 2 Epoch    GPU_mem   box_loss
3.0s 3 \x1b[0m all        201       4250      0.266      0.333      0.195     0.0801      0.245      0.288      0.164     0.0557
4.0s 4 all        201       4250      0.292      0.396      0.252      0.101      0.261      0.346      0.202      0.065
5.0s 5 all        201       4250      0.9      0.9      0.9      0.9      0.9      0.9      0.9      0.9
6.0s 6 Starting training for 1 epochs...
7.0s 7 all        10       20      0.5      0.4      0.3      0.2
"""


def test_parse_console_log_keeps_only_epoch_rows(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text(LOG)
    first, second = parse_console_log(f)
    assert len(first) == 2 and first["epoch"].tolist() == [1, 2]  # post-training val row dropped
    assert first["metrics/mAP50(M)"].tolist() == [0.164, 0.202]
    assert list(second.columns) == ["epoch", *[c for c in first.columns[1:5]]]  # box only


def test_best_epoch_and_save_run(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text(LOG)
    df = parse_console_log(f)[0]
    assert best_epoch(df) == 2
    out = save_run(df, tmp_path / "runs", "demo", {"model": "yolo11n-seg.pt"})
    assert (out / "curves.png").stat().st_size > 0
    assert read_results_csv(out / "epochs.csv").shape[0] == 2
    assert pd.read_json(out / "settings.json", typ="series")["best_epoch"] == 2


def test_learning_curve_and_plots(tmp_path):
    import numpy as np

    from mycodemo.modeling.curves import (learning_curve, plot_fold_scores, plot_learning_curve,
                                          plot_lines, training_run_dir)

    df = pd.DataFrame({"x": np.arange(50.0)})
    folds = [("a", df.iloc[:40], df.iloc[40:]), ("b", df.iloc[10:], df.iloc[:10])]
    lc = learning_curve(folds, lambda tr, te: len(tr), fractions=(0.5, 1.0), repeats=3)
    assert len(lc) == 2 * (3 + 1)  # three draws at 0.5, one at 1.0, per fold
    assert lc[lc.fraction == 1.0]["n_train"].tolist() == [40, 40]
    assert plot_learning_curve(lc, tmp_path / "lc.png").exists()
    scores = pd.DataFrame({"held_out_lab": ["a", "a", "b", "b"], "model": ["m1", "m2"] * 2,
                           "fold_error": [1.5, 2.0, 1.2, 1.8]})
    assert plot_fold_scores(scores, tmp_path / "f.png").exists()
    t = pd.DataFrame({"n_photos": [3, 4, 5], "err_model": [3, 2, 1], "err_baseline": [2, 2, 2]})
    assert plot_lines(t, "n_photos", ["err_model", "err_baseline"], tmp_path / "e.png").exists()
    assert training_run_dir("demo", tmp_path).is_dir()
