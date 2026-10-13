"""Colonisation forecast: fit an S-curve to % colonised per day and forecast the 90% / 95% day."""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

MIN_PHOTOS = 3


def logistic(t, k, t50):
    return 1 / (1 + np.exp(-k * (t - t50)))


def gompertz(t, mu, lag):  # A fixed at 1 (100%)
    return np.exp(-np.exp(mu * np.e * (lag - t) + 1))


def avrami(t, k, n):
    return 1 - np.exp(-((k * np.clip(t, 0, None)) ** n))


MODELS = {
    "logistic": (logistic, [1.0, 5.0]),
    "gompertz": (gompertz, [0.2, 2.0]),
    "avrami": (avrami, [0.15, 2.0]),
}


def fit_curve(t, c, model="logistic"):
    """Fit one curve. Returns the parameters, or None if the fit fails."""
    f, p0 = MODELS[model]
    t, c = np.asarray(t, float), np.asarray(c, float)
    if len(t) < 2:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)  # spread unknown with few points
            params, _ = curve_fit(f, t, c, p0=p0, maxfev=10000)
    except (RuntimeError, ValueError):
        return None
    return params


def day_reaching(f, params, p, t_max=60):
    """First day the curve reaches fraction p, or NaN if it never does within t_max."""
    tt = np.linspace(0, t_max, 6001)
    hit = np.where(f(tt, *params) >= p)[0]
    return tt[hit[0]] if len(hit) else np.nan


def forecast(t, c, model="logistic"):
    """Fit and return {'params', 'day90', 'day95'}; values are None / NaN if the fit failed."""
    params = fit_curve(t, c, model)
    if params is None:
        return {"params": None, "day90": np.nan, "day95": np.nan}
    f = MODELS[model][0]
    return {
        "params": params,
        "day90": day_reaching(f, params, 0.90),
        "day95": day_reaching(f, params, 0.95),
    }


def pace_vs_typical(day90, typical=(10, 14)):
    """Compare the 90% day with the typical oyster range: 'ahead', 'within', 'behind' or 'unknown'."""
    if day90 is None or np.isnan(day90):
        return "unknown"
    if day90 < typical[0]:
        return "ahead"
    if day90 > typical[1]:
        return "behind"
    return "within"


def forecast_by_photo(t, c, model="logistic", min_photos=2):
    """How the forecast changes as each photo is added (photos in day order)."""
    t, c = np.asarray(t, float), np.asarray(c, float)
    rows = []
    for n in range(min_photos, len(t) + 1):
        r = forecast(t[:n], c[:n], model)
        rows.append({"n_photos": n, "last_day": t[n - 1], "day90": r["day90"], "day95": r["day95"]})
    return pd.DataFrame(rows)


def forecast_warnings(t, c, result):
    """Warnings for a weak forecast: too few photos, or 90% not reached within 60 days."""
    out = []
    if len(t) < MIN_PHOTOS:
        out.append(f"Only {len(t)} photo(s): at least {MIN_PHOTOS} are needed for a forecast you can trust.")
    if result["params"] is None:
        out.append("The curve could not be fitted to these photos.")
    elif np.isnan(result["day90"]):
        out.append("90% is not reached within 60 days on this curve.")
    if len(c) and np.max(c) < 0.90 and not np.isnan(result["day90"]):
        out.append(
            f"No photo has reached 90% yet (highest {100 * np.max(c):.0f}%), "
            "so the 90% day is an extrapolation."
        )
    return out
