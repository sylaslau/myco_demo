import pandas as pd

from mycodemo.growth.plots import colonisation_plot

BAND = (10, 14)
LOG = pd.DataFrame({"day": [0, 1, 2, 3, 4], "coverage_pct": [1.0, 3.0, 8.0, 18.0, 35.0]})


def test_band_only_is_one_trace():
    fig = colonisation_plot("Cool oyster", band=BAND)
    assert len(fig.data) == 1
    assert fig.data[0].name == "Baseline range (guides)"


def test_one_point_has_no_forecast():
    fig = colonisation_plot("Cool oyster", LOG.head(1), band=BAND)
    assert len(fig.data) == 2
    assert "needs 2 photos" in [a.text for a in fig.layout.annotations]


def test_two_or_more_points_add_forecast():
    fig = colonisation_plot("Cool oyster", LOG, band=BAND)
    assert len(fig.data) == 3
    assert fig.data[2].name == "Colonisation forecast"
    assert fig.data[2].x[0] == 4 and fig.data[2].y[-1] >= 99.5
    assert not fig.layout.annotations


def test_unknown_group_without_band_raises():
    import pytest
    with pytest.raises(ValueError):
        colonisation_plot("Nothing")
