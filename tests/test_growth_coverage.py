"""Tests for Photo coverage (mycodemo.growth.coverage), using made-up images."""

import numpy as np
import pytest
from PIL import Image

from mycodemo.growth.coverage import coverage, load_coverage


def _save(tmp_path, arr, name="x.png"):
    p = tmp_path / name
    Image.fromarray(arr.astype("uint8")).save(p)
    return p


def test_white_image_is_fully_covered(tmp_path):
    p = _save(tmp_path, np.full((100, 100, 3), 255))
    assert coverage(p) == pytest.approx(1.0)


def test_dark_image_is_empty(tmp_path):
    p = _save(tmp_path, np.zeros((100, 100, 3)))
    assert coverage(p) == 0.0


def test_colourful_image_is_not_mycelium(tmp_path):
    img = np.zeros((100, 100, 3))
    img[..., 0] = 255  # bright red: bright but colourful
    assert coverage(_save(tmp_path, img)) == 0.0


def test_rim_is_ignored(tmp_path):
    img = np.zeros((100, 100, 3))
    img[:10] = 255  # white only in the outer 10% rim
    assert coverage(_save(tmp_path, img)) == 0.0


def test_half_covered(tmp_path):
    img = np.zeros((100, 100, 3))
    img[:, :50] = 255
    assert coverage(_save(tmp_path, img)) == pytest.approx(0.5, abs=0.02)


def test_load_coverage_columns_and_order():
    df = load_coverage()
    assert {"series_id", "day", "coverage"} <= set(df.columns)
    for _, g in df.groupby("series_id"):
        assert g["day"].is_monotonic_increasing
