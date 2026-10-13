"""Photo coverage: from one dish photo to the fraction colonised (0-1)."""

import numpy as np
import pandas as pd
from PIL import Image

from mycodemo.config import INTERIM_DATA_DIR

COVERAGE_CSV = INTERIM_DATA_DIR / "coverage_auto.csv"


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
