"""Tests for Photo coverage (mycodemo.growth.coverage), using made-up images."""

from datetime import date, datetime

import numpy as np
import pytest
from PIL import Image

from mycodemo.growth.coverage import (
    SYLDD_INOCULATION,
    coverage,
    days_since,
    folder_day_mismatches,
    load_coverage,
    photo_datetime,
)


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


def _jpg(path, exif_time=None):
    img = Image.new("RGB", (8, 8))
    if exif_time:
        exif = Image.Exif()
        exif.get_ifd(0x8769)[36867] = exif_time
        img.save(path, exif=exif)
    else:
        img.save(path)
    return path


def test_photo_datetime_prefers_exif_over_file_name(tmp_path):
    p = _jpg(tmp_path / "IMG_20261001_000000.jpg", "2026:10:10 16:08:39")
    assert photo_datetime(p) == datetime(2026, 10, 10, 16, 8, 39)


def test_photo_datetime_falls_back_to_file_name(tmp_path):
    p = _jpg(tmp_path / "IMG_20261011_160014.jpg")
    assert photo_datetime(p) == datetime(2026, 10, 11, 16, 0, 14)


def test_photo_datetime_without_any_time_raises(tmp_path):
    with pytest.raises(ValueError):
        photo_datetime(_jpg(tmp_path / "holiday.jpg"))


def test_photo_datetime_reads_an_open_upload(tmp_path):
    p = _jpg(tmp_path / "x.jpg", "2026:10:09 15:58:44")
    with open(p, "rb") as f:
        assert photo_datetime(f) == datetime(2026, 10, 9, 15, 58, 44)


def test_days_count_from_inoculation_date():
    assert SYLDD_INOCULATION == date(2026, 10, 8)
    assert days_since(datetime(2026, 10, 8, 23, 59)) == 0
    assert days_since(datetime(2026, 10, 9, 0, 1)) == 1
    assert days_since(date(2026, 10, 12)) == 4


def test_folder_name_is_not_trusted(tmp_path):
    (tmp_path / "day_01").mkdir()
    _jpg(tmp_path / "day_01" / "IMG_20261012_154131.jpg")  # taken on day 4
    bad = folder_day_mismatches(tmp_path)
    assert bad[["folder_day", "photo_day"]].values.tolist() == [[1, 4]]


def test_matching_folder_is_not_reported(tmp_path):
    (tmp_path / "day_02").mkdir()
    _jpg(tmp_path / "day_02" / "IMG_20261010_160839.jpg")
    assert folder_day_mismatches(tmp_path).empty
