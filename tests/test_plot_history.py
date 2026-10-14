import pandas as pd

from mycodemo.modeling.curves import history_filename, save_history_csv
from mycodemo.plot_history import plot_runs


def _csv(tmp_path, run, with_loss=True, seg=True):
    df = pd.DataFrame({"epoch": [1, 2, 3], "metrics/mAP50(B)": [0.1, 0.2, 0.3]})
    if seg:
        df["metrics/mAP50(M)"] = [0.05, 0.1, 0.2]
    if with_loss:
        df["train/box_loss"] = [2.0, 1.5, 1.2]
    return save_history_csv(df, tmp_path, "trainabhi_testangelo", model="harvest_readiness", run=run,
                            train_set="abhi", test_set="angelo", date="2026-10-14", epochs=3,
                            imgsz=640, batch=16, base_model="yolo11n-seg.pt")


def test_history_file_name_and_columns(tmp_path):
    p = _csv(tmp_path, "run3")
    assert p.name == "harvest_readiness_run3_trainabhi_testangelo_20261014.csv"
    assert history_filename("a", "r1", "trainx_testy", "2026-10-12") == "a_r1_trainx_testy_20261012.csv"
    assert list(pd.read_csv(p).columns[:9]) == ["model", "run", "train_set", "test_set", "date",
                                                "epochs", "imgsz", "batch", "base_model"]


def test_plot_runs_overlays_and_skips_missing_columns(tmp_path):
    a = _csv(tmp_path, "run1", with_loss=False)  # like a run recovered from a log: no losses
    b = _csv(tmp_path, "run3", seg=False)  # box-only model: falls back to box mAP50
    out = plot_runs([a, b], out_png=tmp_path / "overlay.png")
    assert out.stat().st_size > 0
    assert plot_runs([a, b], "metrics/mAP50(B)").parent == tmp_path  # saved next to the CSVs
