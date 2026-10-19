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
