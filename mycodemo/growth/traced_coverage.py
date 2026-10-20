"""Photo coverage from marks (moved out of notebook 2.4): outline -> substrate mask, sample clicks -> mycelium mask by
nearest colour/texture, brush corrections, % inside the outline.

The marks are the 2.4 editor's record: 'roi' {'kind': 'polygon', 'points'} or 'full_frame_substrate_confirmed', 'seeds'
[{'label': 'mycelium'|'substrate', 'x', 'y', 'radius'}], 'strokes' [{'label', 'size', 'points'}], optional
'corrections', 'background_exclusions' and 'method'. Every coordinate is normalised 0-1 in the EXIF-oriented photo.

Coverage = mycelium pixels / substrate-mask pixels x 100; ambiguous pixels stay in the denominator. The lower and upper
values are an ambiguity envelope, not a confidence interval. The marks are the user's; nothing here is a biological
identity check.
"""

import base64
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

PARAMETERS = {'max_side': 1000, 'default_method': 'auto', 'cluster_count': 3, 'cluster_sample_pixels': 12000,
              'cluster_iterations': 25, 'random_seed': 42, 'ambiguous_margin': .18, 'minimum_cluster_score_gap': .025,
              'seed_variance_floor': .03, 'seed_min_separation': 1., 'minimum_roi_pixels': 50,
              'max_ambiguous_share_for_forecast': .35, 'forecast_tolerance_floor_pct': 10.,
              'hsv_value_min': 170 / 255., 'hsv_saturation_max': 60 / 255., 'dark_mean_threshold': .15,
              'possible_glare_threshold': .02}
METHODS = ('auto', 'seeded', 'cluster', 'hsv', 'manual')


# --- checks on the marks ------------------------------------------------------------------------
def finite(value, name, lo=None, hi=None):
    value = float(value)
    if not np.isfinite(value) or (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f'Invalid {name}: {value}')
    return value


def boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be boolean, not text.')
    return bool(value)


# --- photo and substrate mask ---------------------------------------------------------------------
def load_photo(image):
    """EXIF-oriented RGB array, reduced to max_side pixels; image = path, bytes or file object.
    Returns (rgb, original_size)."""
    source = io.BytesIO(image) if isinstance(image, (bytes, bytearray)) else image
    with Image.open(source) as raw:
        if getattr(raw, 'n_frames', 1) != 1:
            raise ValueError('Use a single still photo.')
        oriented = ImageOps.exif_transpose(raw)
        if 'A' in oriented.getbands() and np.asarray(oriented.getchannel('A')).min() < 255:
            raise ValueError('Use an opaque photograph.')
        oriented = oriented.convert('RGB')
        original_size = oriented.size
        oriented.thumbnail((PARAMETERS['max_side'], PARAMETERS['max_side']), Image.Resampling.LANCZOS)
        rgb = np.asarray(oriented).copy()
    if min(rgb.shape[:2]) < 8:
        raise ValueError('Photo too small to inspect.')
    return rgb, original_size


def polygon_points(points):
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise ValueError('Polygon needs at least three vertices.')
    pts = np.array([[finite(p[0], 'polygon x', 0, 1), finite(p[1], 'polygon y', 0, 1)] for p in points])
    if len(pts) > 256 or len(np.unique(pts, axis=0)) < 3:
        raise ValueError('Invalid polygon vertices.')
    area = abs(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))) / 2
    if area < 1e-6:
        raise ValueError('Polygon area is zero or too small.')

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if (j - i) % len(pts) in (1, len(pts) - 1):
                continue
            a, b = pts[i], pts[(i + 1) % len(pts)]
            c, d = pts[j], pts[(j + 1) % len(pts)]
            if orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0:
                raise ValueError('Polygon edges cross.')
    return pts


def polygon_mask(points, shape):
    h, w = shape
    points = polygon_points(points)
    image = Image.new('L', (w, h), 0)
    ImageDraw.Draw(image).polygon([(round(x * (w - 1)), round(y * (h - 1))) for x, y in points], fill=255)
    return np.asarray(image) > 0


def substrate_mask(record, shape, external=None):
    """(mask, None), or (None, message) when no outline is given. external = an already loaded binary mask for
    record['roi_mask_path'] (the notebook loads the file)."""
    full = boolean(record.get('full_frame_substrate_confirmed', False), 'whole-frame confirmation')
    choices = int(record.get('roi') is not None) + int(record.get('roi_mask_path') is not None) + int(full)
    if choices == 0:
        return None, 'Trace substrate boundary or explicitly confirm whole-frame substrate.'
    if choices != 1:
        raise ValueError('Choose one ROI: polygon, external mask or whole frame.')
    if full:
        mask = np.ones(shape, bool)
    elif record.get('roi_mask_path') is not None:
        if external is None:
            raise ValueError('External mask file not loaded.')
        mask = np.asarray(external, bool).copy()
    else:
        if record['roi'].get('kind') != 'polygon':
            raise ValueError('ROI must have kind=polygon.')
        mask = polygon_mask(record['roi']['points'], shape)
    for points in record.get('background_exclusions', []):
        mask &= ~polygon_mask(points, shape)
    if mask.sum() < PARAMETERS['minimum_roi_pixels']:
        raise ValueError('Substrate ROI empty or too small.')
    return mask, None


# --- colour and texture methods -------------------------------------------------------------------
def box_mean(array, radius=3):
    a = np.pad(np.asarray(array, float), radius, mode='reflect')
    k = 2 * radius + 1
    integral = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]) / (k * k)


def image_features(rgb):
    f = rgb.astype(np.float32) / 255.
    linear = np.where(f <= .04045, f / 12.92, ((f + .055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750],
                             [.0193339, .1191920, .9503041]], np.float32).T
    xyz /= np.array([.95047, 1., 1.08883])
    lab = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * lab[..., 1] - 16
    a = 500 * (lab[..., 0] - lab[..., 1])
    b = 200 * (lab[..., 1] - lab[..., 2])
    gray = f @ np.array([.2126, .7152, .0722])
    mean = box_mean(gray)
    texture = np.sqrt(np.maximum(0, box_mean(gray * gray) - mean * mean))
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    v = f.max(axis=2)
    s = np.divide(v - f.min(axis=2), v, out=np.zeros_like(v), where=v > 0)
    features = np.stack([L / 100, a / 128 * .7, b / 128 * .7, np.clip(texture / .15, 0, 2) * .35,
                         np.clip(edge / .2, 0, 2) * .2, s * .25], axis=2).astype(np.float32)
    return features, {'gray': gray, 'value': v, 'saturation': s, 'lightness': L}


def hsv_candidate(channels, roi):
    v, s = channels['value'], channels['saturation']
    v0, s0 = PARAMETERS['hsv_value_min'], PARAMETERS['hsv_saturation_max']
    return ((v > v0) & (s < s0) & roi, ((abs(v - v0) < .06) | (abs(s - s0) < .06)) & roi,
            {'method': 'HSV threshold baseline', 'notes': ['Whiteness is not biological identity.']})


def nearest_centres(flat, centres):
    distances = np.empty((len(flat), len(centres)), np.float32)
    for start in range(0, len(flat), 65536):
        block = flat[start:start + 65536]
        distances[start:start + len(block)] = np.sum((block[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    return distances


def cluster_candidate(features, roi):
    flat = features[roi]
    rng = np.random.default_rng(PARAMETERS['random_seed'])
    sample = flat[rng.choice(len(flat), min(len(flat), PARAMETERS['cluster_sample_pixels']), replace=False)]
    k = min(PARAMETERS['cluster_count'], len(sample))
    if k < 2:
        raise ValueError('Insufficient ROI pixels for clustering.')
    order = np.argsort(sample[:, 0])
    centres = sample[order[np.linspace(0, len(sample) - 1, k).astype(int)]].copy()
    for _ in range(PARAMETERS['cluster_iterations']):
        labels = nearest_centres(sample, centres).argmin(1)
        updated = np.array([sample[labels == j].mean(0) if np.any(labels == j) else centres[j] for j in range(k)])
        if np.allclose(updated, centres, atol=1e-4):
            break
        centres = updated
    centres = np.unique(np.round(centres, 5), axis=0)
    k = len(centres)
    if k < 2:
        return roi.copy(), roi.copy(), {'method': 'per-image Lab/texture clustering', 'cluster_score_gap': 0.,
                                        'notes': ['Uniform ROI: no separable clusters; all pixels uncertain.']}
    distances = nearest_centres(flat, centres)
    scores = centres[:, 0] - .35 * np.hypot(centres[:, 1] / .7, centres[:, 2] / .7)
    chosen = int(np.argmax(scores))
    other = [j for j in range(k) if j != chosen]
    dm = distances[:, chosen]
    ds = distances[:, other].min(1)
    candidate = dm <= ds
    ambiguous = abs(dm - ds) / (dm + ds + 1e-8) < PARAMETERS['ambiguous_margin']
    # Intermediate colour/texture clusters have no verified class identity.
    assigned = distances.argmin(1)
    darkest = int(np.argmin(scores))
    ambiguous |= (assigned != chosen) & (assigned != darkest)
    gap = float(np.sort(scores)[-1] - np.sort(scores)[-2])
    notes = ['Bright low-chroma cluster is an unverified candidate; inspect pale substrate and glare.']
    if gap < PARAMETERS['minimum_cluster_score_gap']:
        ambiguous[:] = True
        notes.append('No clear cluster separation; add examples or corrections.')
    mask = np.zeros(roi.shape, bool)
    uncertain = mask.copy()
    mask[roi] = candidate
    uncertain[roi] = ambiguous
    return mask, uncertain, {'method': 'per-image Lab/texture clustering', 'cluster_score_gap': gap, 'notes': notes}


def seed_masks(seeds, roi):
    h, w = roi.shape
    yy, xx = np.ogrid[:h, :w]
    masks = {label: np.zeros(roi.shape, bool) for label in ['mycelium', 'substrate']}
    counts = {label: 0 for label in masks}
    for seed in seeds:
        label = seed.get('label')
        if label not in masks:
            raise ValueError('Seed label must be mycelium or substrate.')
        x = finite(seed['x'], 'seed x', 0, 1) * (w - 1)
        y = finite(seed['y'], 'seed y', 0, 1) * (h - 1)
        if not roi[round(y), round(x)]:
            raise ValueError('Example outside substrate ROI.')
        radius = max(1., finite(seed.get('radius', .015), 'seed radius', .001, .1) * min(h, w))
        masks[label] |= (((xx - x) ** 2 + (yy - y) ** 2) <= radius ** 2) & roi
        counts[label] += 1
    if np.any(masks['mycelium'] & masks['substrate']):
        raise ValueError('Contradictory example patches overlap.')
    return masks, counts


def seeded_candidate(features, roi, seeds):
    masks, counts = seed_masks(seeds, roi)
    if any(not masks[label].any() for label in masks):
        raise ValueError('Need examples of both mycelium and exposed substrate.')
    examples = [features[masks[label]] for label in ['mycelium', 'substrate']]
    centres = np.array([v.mean(0) for v in examples])
    scale = np.sqrt(np.mean([v.var(0) for v in examples], axis=0) + PARAMETERS['seed_variance_floor'] ** 2)
    separation = float(np.linalg.norm((centres[0] - centres[1]) / scale))
    distances = nearest_centres(features[roi] / scale, centres / scale)
    candidate = distances[:, 0] <= distances[:, 1]
    ambiguous = abs(distances[:, 0] - distances[:, 1]) / (distances.sum(1) + 1e-8) < PARAMETERS['ambiguous_margin']
    ambiguous |= distances.min(1) > 25.
    notes = ['Per-image example matching, not a biological identity detector.']
    if separation < PARAMETERS['seed_min_separation']:
        ambiguous[:] = True
        notes.append('Examples poorly separated; add informative examples.')
    mask = np.zeros(roi.shape, bool)
    uncertain = mask.copy()
    mask[roi] = candidate
    uncertain[roi] = ambiguous
    return mask, uncertain, {'method': 'per-image colour/texture examples', 'example_clicks': counts,
                             'sample_pixels': {k: int(v.sum()) for k, v in masks.items()}, 'separation': separation,
                             'notes': notes}


def stroke_mask(stroke, shape):
    h, w = shape
    if stroke.get('label') not in ('mycelium', 'substrate'):
        raise ValueError('Brush label must be mycelium or substrate.')
    size = finite(stroke.get('size'), 'brush size', .002, .5)
    points = stroke.get('points')
    if not isinstance(points, (list, tuple)) or not 1 <= len(points) <= 20000:
        raise ValueError('A brush stroke needs 1 to 20000 points.')
    xy = [(finite(p[0], 'stroke x', 0, 1) * (w - 1), finite(p[1], 'stroke y', 0, 1) * (h - 1)) for p in points]
    diameter = max(1., size * min(h, w))
    image = Image.new('L', (w, h), 0)
    pen = ImageDraw.Draw(image)
    if len(xy) > 1:
        pen.line(xy, fill=255, width=int(round(diameter)), joint='curve')
    for x, y in xy:
        pen.ellipse([x - diameter / 2, y - diameter / 2, x + diameter / 2, y + diameter / 2], fill=255)
    return np.asarray(image) > 0


def segment(rgb, roi, record):
    """(mycelium mask, ambiguous mask, method details, channels, method comparison %) inside roi."""
    features, channels = image_features(rgb)
    method = record.get('method', PARAMETERS['default_method'])
    seeds = record.get('seeds', [])
    if method not in METHODS:
        raise ValueError('Unknown method.')
    hsv = hsv_candidate(channels, roi)
    comparisons = {'HSV baseline': 100 * float(hsv[0].sum() / roi.sum())}
    if method == 'auto':
        method = 'seeded' if {'mycelium', 'substrate'}.issubset({s.get('label') for s in seeds}) else 'cluster'
    if method == 'hsv':
        mask, uncertain, details = hsv
    elif method == 'seeded':
        mask, uncertain, details = seeded_candidate(features, roi, seeds)
    elif method == 'manual':
        mask = np.zeros(roi.shape, bool)
        uncertain = roi.copy()
        details = {'method': 'manual correction polygons', 'notes': ['Unlabelled ROI begins uncertain; last correction wins.']}
    else:
        if seeds:
            seed_masks(seeds, roi)
        mask, uncertain, details = cluster_candidate(features, roi)
    comparisons[details['method']] = 100 * float(mask.sum() / roi.sum())
    for correction in record.get('corrections', []):
        label = correction.get('label')
        if label not in ('mycelium', 'substrate', 'uncertain'):
            raise ValueError('Unknown correction label.')
        region = polygon_mask(correction['points'], roi.shape) & roi
        if label == 'mycelium':
            mask[region] = True
            uncertain[region] = False
        elif label == 'substrate':
            mask[region] = False
            uncertain[region] = False
        else:
            uncertain[region] = True
    for stroke in record.get('strokes', []):  # brush strokes are applied after the polygons, in the order painted
        region = stroke_mask(stroke, roi.shape) & roi
        mask[region] = stroke['label'] == 'mycelium'
        uncertain[region] = False
    return mask & roi, uncertain & roi, details, channels, comparisons


# --- coverage, quality, overlay -------------------------------------------------------------------
def coverage_statistics(mask, uncertain, roi):
    n = int(roi.sum())
    if n == 0:
        raise ValueError('Empty substrate denominator.')
    nominal = 100 * float((mask & roi).sum()) / n
    lower = 100 * float((mask & roi & ~uncertain).sum()) / n
    upper = 100 * float(((mask & roi & ~uncertain) | (uncertain & roi)).sum()) / n
    return {'substrate_pixels': n, 'nominal_mycelium_pixels': int((mask & roi).sum()),
            'ambiguous_pixels': int((uncertain & roi).sum()),
            'coverage_pct': nominal, 'coverage_lower_pct': lower, 'coverage_upper_pct': upper,
            'uncertainty_pct': max(PARAMETERS['forecast_tolerance_floor_pct'], nominal - lower, upper - nominal)}


def photo_quality(rgb, roi, channels):
    gray = channels['gray']
    brightness = float(gray[roi].mean())
    glare = float((rgb.min(2)[roi] >= 250).mean())
    lap = -4 * gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
    dark = brightness < PARAMETERS['dark_mean_threshold']
    notes = []
    if dark:
        notes.append('Very dark substrate region; retake or review lighting.')
    if glare > PARAMETERS['possible_glare_threshold']:
        notes.append('Possible glare/very white surface; inspect the overlay.')
    return {'mean_brightness': brightness, 'possible_glare_fraction': glare, 'laplacian_variance': float(lap[roi].var()),
            'dark': dark, 'notes': notes}


def overlay_image(rgb, mask, uncertain, roi):
    out = rgb.astype(float).copy()
    region = mask & ~uncertain
    out[region] = .55 * out[region] + .45 * np.array([225, 20, 185])
    region = uncertain & roi
    out[region] = .35 * out[region] + .65 * np.array([255, 176, 0])
    pad = np.pad(roi, 1, constant_values=False)
    inside = roi & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    out[roi & ~inside] = [0, 230, 235]
    return np.clip(out, 0, 255).astype(np.uint8)


def editor_image(image):
    """The photo as the editor must show it (EXIF-oriented, max_side pixels) as a JPEG data URL, so the editor's 0-1
    coordinates fall on the same pixels as load_photo here. The counting itself always uses the original photo."""
    rgb, _ = load_photo(image)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def coverage_from_marks(image, marks):
    """Coverage of one photo from the user's marks.

    image = path, bytes or file object; marks = the editor record (outline, sample clicks, brush strokes).
    Returns {'coverage_pct', 'method', 'uncertainty', 'lower_pct', 'upper_pct', 'ambiguous_share', 'notes', 'quality'};
    uncertainty is in percentage points (at least forecast_tolerance_floor_pct). Raises ValueError when the marks
    cannot give a number (no outline, a click outside it, only one kind of click with 'seeded', ...)."""
    rgb, _ = load_photo(Path(image) if isinstance(image, str) else image)
    roi, message = substrate_mask(marks, rgb.shape[:2])
    if roi is None:
        raise ValueError(message)
    mask, uncertain, details, channels, _ = segment(rgb, roi, marks)
    stats = coverage_statistics(mask, uncertain, roi)
    quality = photo_quality(rgb, roi, channels)
    return {'coverage_pct': stats['coverage_pct'], 'method': details['method'], 'uncertainty': stats['uncertainty_pct'],
            'lower_pct': stats['coverage_lower_pct'], 'upper_pct': stats['coverage_upper_pct'],
            'ambiguous_share': stats['ambiguous_pixels'] / stats['substrate_pixels'],
            'notes': details['notes'] + quality['notes'], 'quality': quality}


def circle(cx, cy, r, n=48):
    """A circle outline as normalised points (the notebook's hand-set dish outlines)."""
    return [[cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)] for k in range(n)]
