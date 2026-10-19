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


def _centre_box(h, w, border):
    bh, bw = int(h * border), int(w * border)
    return bh, h - bh, bw, w - bw


def coverage_mask(path, v_min=170, s_max=60, border=0.10):
    """Boolean mask (inner area only) of 'white mycelium' pixels, and the (top, bottom, left, right) box it covers."""
    if hasattr(path, "seek"):
        path.seek(0)
    hsv = np.asarray(Image.open(path).convert("HSV")).astype(float)
    top, bottom, left, right = box = _centre_box(*hsv.shape[:2], border)
    centre = hsv[top:bottom, left:right]
    S, V = centre[..., 1], centre[..., 2]
    return (V > v_min) & (S < s_max), box


def coverage(path, v_min=170, s_max=60, border=0.10):
    """Fraction of 'white mycelium' pixels: bright (V) and not colourful (S).
    The outer 10% is cut off so the dish rim doesn't count."""
    return coverage_mask(path, v_min, s_max, border)[0].mean()


# --- per-photo card: method choice, quality check, tinted picture ----------------------------
TOO_PALE = 0.95          # HSV counts this much of the frame: the substrate itself is pale, so it cannot separate
BLUR_MIN = 60.0          # variance of the Laplacian on a 800 px picture; lower = blurry
GLARE_MAX = 0.02         # share of pure-white pixels; higher = glare
DARK_MIN = 60.0          # mean brightness (0-255); lower = too dark
_QUALITY_SIDE = 800
_KMEANS_PIXELS = 20000
_MIN_L_GAP = 10.0        # Lab lightness gap between clusters below which K-means found no real split


def _rgb(path, max_side=None):
    if hasattr(path, "seek"):
        path.seek(0)
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return np.asarray(img)


def photo_quality(path):
    """Blur, glare and darkness checks: {'ok', 'reasons', 'blur', 'glare', 'brightness'}.
    Reasons are plain words: 'blurry', 'too bright (glare)', 'too dark'."""
    import cv2

    rgb = _rgb(path, _QUALITY_SIDE)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare = float((rgb.min(axis=2) >= 250).mean())
    brightness = float(gray.mean())
    reasons = []
    if blur < BLUR_MIN:
        reasons.append("blurry")
    if glare > GLARE_MAX:
        reasons.append("too bright (glare)")
    if brightness < DARK_MIN:
        reasons.append("too dark")
    return {"ok": not reasons, "reasons": reasons, "blur": blur, "glare": glare, "brightness": brightness}


def quality_message(q):
    """'' when the photo passes, else the retake message naming the actual reason(s)."""
    if q["ok"]:
        return ""
    return f"This photo is {' and '.join(q['reasons'])}. Try taking another one."


def kmeans_mask(path, border=0.10, seed=0):
    """Colonised pixels by K-means in Lab colour: the lightest, least colourful cluster of 2 (else 3) clusters.
    Returns (mask, box), or (None, box) when the clusters are not clearly different in lightness."""
    import cv2
    from sklearn.cluster import KMeans

    rgb = _rgb(path)
    top, bottom, left, right = box = _centre_box(*rgb.shape[:2], border)
    lab = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2LAB).astype(float)
    flat = lab.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    sample = flat[rng.choice(len(flat), min(_KMEANS_PIXELS, len(flat)), replace=False)]
    for k in (2, 3):
        km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(sample)
        order = np.argsort(km.cluster_centers_[:, 0])          # by lightness L
        gap = km.cluster_centers_[order[-1], 0] - km.cluster_centers_[order[-2], 0] if k == 3 else \
            km.cluster_centers_[order[-1], 0] - km.cluster_centers_[order[0], 0]
        if gap >= _MIN_L_GAP:
            top_label = order[-1]
            return (km.predict(flat) == top_label).reshape(lab.shape[:2]), box
    return None, box


def photo_coverage(path):
    """% colonised for one photo with the method used: {'coverage', 'method', 'mask', 'box', 'too_pale'}.
    HSV threshold first; if the result is 'too pale' (>= TOO_PALE) K-means in Lab colour is the backup."""
    mask, box = coverage_mask(path)
    frac, method, too_pale = float(mask.mean()), "HSV threshold", False
    if frac >= TOO_PALE:
        too_pale = True
        alt, alt_box = kmeans_mask(path)
        if alt is not None:
            mask, box, frac, method = alt, alt_box, float(alt.mean()), "K-means (Lab colour)"
        else:
            method = "HSV threshold (too pale; K-means found no split)"
    return {"coverage": frac, "method": method, "mask": mask, "box": box, "too_pale": too_pale}


def tinted_photo(path, mask, box, colour=(255, 0, 255), alpha=0.45):
    """The photo as an RGB array with the counted pixels tinted (the rest is left as taken)."""
    rgb = _rgb(path).copy()
    top, bottom, left, right = box
    region = rgb[top:bottom, left:right].astype(float)
    region[mask] = (1 - alpha) * region[mask] + alpha * np.array(colour, float)
    rgb[top:bottom, left:right] = region.astype(np.uint8)
    return rgb


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
