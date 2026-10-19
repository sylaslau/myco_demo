import pandas as pd
import pytest

from mycodemo.config import PROJ_ROOT
from mycodemo.growth import consensus_estimate
from mycodemo.presentation_data import SCHEMAS, build_presentation_data, substrate_group


def test_substrate_group():
    assert substrate_group("wheat straw") == "straw"
    assert substrate_group("tef straw") == "straw"
    assert substrate_group("cotton waste") == "cotton"
    assert substrate_group("sawdust") == "sawdust"
    assert substrate_group("wheat straw + hazelnut husk") is None
    assert substrate_group("master's mix") is None
    assert substrate_group("hazelnut husk") is None


def test_build_presentation_data(tmp_path):
    try:
        consensus_estimate.find_bundle()
    except consensus_estimate.ModelUnavailable:
        pytest.skip("no 2.2 bundle")
    for sub in ("data", "models", "reports"):
        (tmp_path / sub).symlink_to(PROJ_ROOT / sub)
    counts, skipped = build_presentation_data(tmp_path)
    for name, columns in SCHEMAS.items():
        frame = pd.read_csv(tmp_path / "presentation_data" / f"{name}.csv")
        assert list(frame.columns) == columns.split()
        assert not frame.isna().any().any()
    assert counts["heldout_predictions"] == 0
    assert counts["baseline_curves"] > 0 and counts["observations"] > 0


def test_curves_timeline_and_labels(tmp_path):
    try:
        consensus_estimate.find_bundle()
    except consensus_estimate.ModelUnavailable:
        pytest.skip("no 2.2 bundle")
    for sub in ("data", "models", "reports"):
        (tmp_path / sub).symlink_to(PROJ_ROOT / sub)
    counts, skipped = build_presentation_data(tmp_path)
    out = tmp_path / "presentation_data"
    curves = pd.read_csv(out / "baseline_curves.csv")
    assert {"Pleurotus djamor", "Pleurotus ostreatus"} <= set(curves.species)
    assert set(curves[curves.species == "Pleurotus djamor"].oyster_group) == {"tropical"}
    assert curves.temperature_c.between(18, 30).all()
    obs = pd.read_csv(out / "observations.csv")
    assert not obs.dataset.str.contains("nan").any()
    timeline = pd.read_csv(out / "app_timeline.csv")
    ost = timeline[(timeline.species == "Pleurotus ostreatus") & (timeline.temperature_c == 22) & (timeline.kind == "event")]
    days = {r["item"]: (r.start_day, r.end_day) for _, r in ost.iterrows()}
    assert days["Fully colonised"] == (15, 24) and days["First harvest"] == (26, 35)   # the app's page at 22 °C
    assert any("Pleurotus pulmonarius" in str(s) for s in skipped)
