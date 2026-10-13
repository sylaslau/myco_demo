"""Photo coverage: from one dish photo to the fraction colonised (0-1)."""

import re
from datetime import date, datetime

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from mycodemo.config import INTERIM_DATA_DIR, RAW_DATA_DIR

COVERAGE_CSV = INTERIM_DATA_DIR / "coverage_auto.csv"
SYLDD_DIR = RAW_DATA_DIR / "syldd_2026"
SYLDD_CSV = INTERIM_DATA_DIR / "coverage_syldd.csv"
SYLDD_INOCULATION = date(2026, 10, 8)

_EXIF_DATETIME_ORIGINAL = 36867  # in the Exif sub-IFD
_EXIF_DATETIME = 306  # in the main IFD
_EXIF_IFD = 0x8769
_NAME_TIME = re.compile(r"IMG_(\d{8})_(\d{6})")


def coverage(path, v_min=170, s_max=60, border=0.10):
    """Fraction of 'white mycelium' pixels: bright (V) and not colourful (S).
    The outer 10% is cut off so the dish rim doesn't count."""
    hsv = np.asarray(Image.open(path).convert("HSV")).astype(float)
    h, w, _ = hsv.shape
    bh, bw = int(h * border), int(w * border)
    centre = hsv[bh : h - bh, bw : w - bw]
    S, V = centre[..., 1], centre[..., 2]
    return ((V > v_min) & (S < s_max)).mean()


def load_coverage(path=COVERAGE_CSV):
    """Table with series_id, dataset, day, coverage, file; sorted by growth and day."""
    return pd.read_csv(path).sort_values(["series_id", "day"]).reset_index(drop=True)


def photo_datetime(path):
    """When a photo was taken: EXIF time, else the IMG_YYYYMMDD_HHMMSS file name.
    `path` may be a file path or an open file. Raises ValueError if neither gives a time."""
    try:
        exif = Image.open(path).getexif()
        raw = exif.get_ifd(_EXIF_IFD).get(_EXIF_DATETIME_ORIGINAL) or exif.get(_EXIF_DATETIME)
        if raw:
            return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except (OSError, ValueError):
        pass  # unreadable or malformed EXIF: fall back to the file name
    name = getattr(path, "name", path)
    m = _NAME_TIME.search(str(name))
    if m is None:
        raise ValueError(f"No EXIF time and no IMG_YYYYMMDD_HHMMSS in the name: {name}")
    return datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S")


def days_since(when, inoculation=SYLDD_INOCULATION):
    """Whole calendar days from inoculation to `when` (a date or datetime)."""
    return (getattr(when, "date", lambda: when)() - inoculation).days


def build_syldd_coverage(raw_dir=SYLDD_DIR, inoculation=SYLDD_INOCULATION):
    """Table with container_id, day, coverage, file for every syldd photo.
    Day comes from the photo time, never from the folder name. Within a folder, photos
    are matched to shoot_order.csv by time order (photo 1 = position 1, ...)."""
    order = pd.read_csv(raw_dir / "shoot_order.csv").sort_values("position")["container_id"].tolist()
    rows = []
    for folder in sorted(raw_dir.glob("day_*")):
        photos = sorted(folder.glob("*.jpg"), key=photo_datetime)
        if not photos:
            continue
        if len(photos) > len(order):
            raise ValueError(f"{folder.name}: {len(photos)} photos but only {len(order)} positions")
        if len(photos) < len(order):
            logger.warning(f"{folder.name}: {len(photos)} photos for {len(order)} positions; "
                           f"matched in time order to the first {len(photos)}")
        for cid, p in zip(order, photos):
            rows.append({"container_id": cid, "day": days_since(photo_datetime(p), inoculation),
                         "coverage": coverage(p), "file": f"{folder.name}/{p.name}"})
    return pd.DataFrame(rows).sort_values(["container_id", "day"]).reset_index(drop=True)


def folder_day_mismatches(raw_dir=SYLDD_DIR, inoculation=SYLDD_INOCULATION):
    """Photos whose time-based day differs from the number in their day_XX folder name."""
    out = []
    for folder in sorted(raw_dir.glob("day_*")):
        named = int(folder.name.split("_")[1])
        for p in sorted(folder.glob("*.jpg")):
            day = days_since(photo_datetime(p), inoculation)
            if day != named:
                out.append({"file": f"{folder.name}/{p.name}", "folder_day": named, "photo_day": day})
    return pd.DataFrame(out, columns=["file", "folder_day", "photo_day"])


def load_syldd_coverage(path=SYLDD_CSV):
    """syldd table (container_id, day, coverage, file), sorted by dish and day."""
    return pd.read_csv(path).sort_values(["container_id", "day"]).reset_index(drop=True)
