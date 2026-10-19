"""Colonisation time estimate for oysters (Pleurotus): days to colonise, from plot group, substrate group and temperature.

Notebook 2.2 fits and checks the model; this module holds what the app needs. A coefficients file written by
the notebook (``COEFFICIENTS_PATH``) says whether the fitted model is used or the 2.1 band is kept
(``use_model``). Without the file, ``estimate_days`` falls back to the usemycro.com 10-14 day range the app
uses today.

The model is a linear regression on log(typical days):

    log(days) = intercept + group effect + substrate group effect + temperature term (+ log C:N, optional)

with straw and Cool oyster as the reference levels. The temperature term is the distance (°C) from the
group's optimum, or ``-log(rate)`` of a cardinal-temperature rate (1 at the optimum, straight lines down to
the limits). The min and max days are the typical days times the median min/typ and max/typ ratios of the data.
Spawn % is not an input; ``spawn_warning`` only flags a spawn share outside 5-20 % of the total mix.
"""

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

GROUPS = ["Cool oyster", "Tropical oyster", "King oyster"]
SUBSTRATE_GROUPS = ["straw", "supplemented sawdust", "compost", "cotton waste", "other"]
REFERENCE = {"group": GROUPS[0], "substrate": SUBSTRATE_GROUPS[0]}
TEMP_FORMS = ("none", "distance", "cardinal")
SPAWN_RANGE = (5.0, 20.0)                 # % of the total mix (spawn / (wet substrate + spawn))
RATE_FLOOR = 0.15                         # slowest relative growth rate allowed by the cardinal rule
FALLBACK_BAND = (10.0, 12.0, 14.0)        # usemycro.com guidance, used when there is no coefficients file
COEFFICIENTS_PATH = Path(__file__).with_name("colonisation_time_estimate.json")


class Estimate(NamedTuple):
    min_days: float
    typical_days: float
    max_days: float
    spawn_warning: str | None
    source: str                           # "our model" or "typical range" (names used by pipeline.SOURCES)


def spawn_warning(spawn_pct_of_mix):
    """Warning text if the spawn share is outside 5-20 % of the total mix, else None (also None if not given)."""
    if spawn_pct_of_mix is None or np.isnan(spawn_pct_of_mix):
        return None
    lo, hi = SPAWN_RANGE
    if lo <= spawn_pct_of_mix <= hi:
        return None
    return (f"Spawn is {spawn_pct_of_mix:.1f}% of the total mix. The days are only checked for {lo:g}-{hi:g}%, "
            "so treat them as a rough guess.")


def cardinal_rate(temp_c, t_opt, t_lo, t_hi):
    """Relative growth rate: 1 at t_opt, straight lines down to 0 at t_lo and t_hi, never below RATE_FLOOR."""
    t = np.asarray(temp_c, dtype=float)
    rate = np.where(t < t_opt, (t - t_lo) / (t_opt - t_lo), (t_hi - t) / (t_hi - t_opt))
    return np.clip(rate, RATE_FLOOR, 1.0)


def _group_opt(spec, groups):
    fallback = spec["group_opt"]["all"]
    return np.array([spec["group_opt"].get(g, fallback) for g in groups], dtype=float)


def _design(df, spec):
    """Columns of the regression for the terms named in spec['terms'] (without the intercept)."""
    cols = []
    for name in spec["terms"]:
        if name.startswith("group="):
            cols.append((df["plot_group"] == name[6:]).to_numpy(float))
        elif name.startswith("substrate="):
            cols.append((df["substrate_group"] == name[10:]).to_numpy(float))
        elif name == "temp":
            temp = df["temp_c"].to_numpy(float)
            if spec["temp_form"] == "distance":
                cols.append(np.abs(temp - _group_opt(spec, df["plot_group"])))
            else:
                lo, hi = spec["t_limits"]
                cols.append(-np.log(cardinal_rate(temp, spec["t_opt"], lo, hi)))
        elif name == "log_cn":
            cols.append(np.log(df["cn"].to_numpy(float)) - np.log(spec["cn_ref"]))
        else:
            raise ValueError(f"unknown term {name!r}")
    return np.column_stack(cols) if cols else np.empty((len(df), 0))


def fit_model(df, temp_form="cardinal", use_cn=False, t_opt=25.0, t_limits=(10.0, 35.0), cn_ref=70.0):
    """Fit log(days_typ) by least squares. ``df`` needs plot_group, substrate_group, temp_c, days_typ (and cn if use_cn).

    A level that is absent from ``df`` gets no term (effect 0). Levels that cannot be told apart (for example
    a group and a substrate group that share the same single row) are split evenly by the minimum-norm solution
    and flagged by ``rank_deficient``. Returns a JSON-ready dict.
    """
    if temp_form not in TEMP_FORMS:
        raise ValueError(f"temp_form must be one of {TEMP_FORMS}")
    terms = [f"group={g}" for g in GROUPS if g != REFERENCE["group"] and (df["plot_group"] == g).any()]
    terms += [f"substrate={s}" for s in SUBSTRATE_GROUPS
              if s != REFERENCE["substrate"] and (df["substrate_group"] == s).any()]
    if temp_form != "none":
        terms.append("temp")
    if use_cn:
        terms.append("log_cn")
    group_opt = {g: float(v) for g, v in df.groupby("plot_group")["temp_c"].median().items()}
    group_opt["all"] = float(df["temp_c"].median())
    spec = {"temp_form": temp_form, "t_opt": float(t_opt), "t_limits": [float(t) for t in t_limits],
            "group_opt": group_opt, "cn_ref": float(cn_ref), "terms": terms}
    X = np.column_stack([np.ones(len(df)), _design(df, spec)])
    beta, _, rank, _ = np.linalg.lstsq(X, np.log(df["days_typ"].to_numpy(float)), rcond=None)
    spec.update(intercept=float(beta[0]), coef={t: float(b) for t, b in zip(terms, beta[1:])},
                n_rows=len(df), rank_deficient=bool(rank < X.shape[1]))
    return spec


def predict_typical(model, df):
    """Typical days (array) for the rows of ``df`` from a ``fit_model`` dict."""
    X = _design(df, {**model, "terms": list(model["coef"])})
    beta = np.array(list(model["coef"].values()))
    return np.exp(model["intercept"] + X @ beta)


def band_ratios(df):
    """Median min/typ and max/typ over the rows that give all three days."""
    ok = df[["days_min", "days_typ", "days_max"]].notna().all(axis=1)
    d = df[ok]
    return {"min_over_typ": float((d["days_min"] / d["days_typ"]).median()),
            "max_over_typ": float((d["days_max"] / d["days_typ"]).median()), "n_rows": int(ok.sum())}


def baseline_band(df):
    """The 2.1 band per plot group: median min, typical and max days (NaN-skipping), plus 'all' rows together."""
    def med(d):
        return {k: float(d[c].median()) for k, c in [("min", "days_min"), ("typ", "days_typ"), ("max", "days_max")]}
    out = {g: med(d) for g, d in df.groupby("plot_group")}
    out["all"] = med(df)
    return out


def percent_effects(model):
    """Each term as a plain % change in days (group and substrate: against the reference level)."""
    return {t: float(np.expm1(b) * 100) for t, b in model["coef"].items() if t != "temp"}


def load_coefficients(path=None):
    """The coefficients dict, or None if the file is not there."""
    path = Path(path or COEFFICIENTS_PATH)
    return json.loads(path.read_text()) if path.exists() else None


def estimate_days(group, substrate_group, temp_c=None, spawn_pct_of_mix=None, coefficients=None):
    """(min, typical, max) days to colonise, plus the spawn warning and where the numbers came from.

    ``coefficients`` is the dict from the coefficients file; None reads the file, and without it the
    usemycro.com 10-14 day range is returned. If the file says ``use_model`` is false (the fitted model did not
    beat the 2.1 baseline) the group's 2.1 band is returned and the temperature and substrate are ignored.
    ``temp_c`` None means "at the optimum". Unknown group or substrate group: effect 0 (reference level).
    """
    warn = spawn_warning(spawn_pct_of_mix)
    coef = coefficients if coefficients is not None else load_coefficients()
    if coef is None:
        return Estimate(*FALLBACK_BAND, warn, "typical range")
    if not coef.get("use_model"):
        band = coef["baseline"].get(group, coef["baseline"]["all"])
        return Estimate(band["min"], band["typ"], band["max"], warn, "typical range")
    model = coef["model"]
    if temp_c is None:
        temp_c = model["t_opt"] if model["temp_form"] == "cardinal" else _group_opt(model, [group])[0]
    row = pd.DataFrame({"plot_group": [group], "substrate_group": [substrate_group], "temp_c": [float(temp_c)],
                        "cn": [model["cn_ref"]]})
    typical = float(predict_typical(model, row)[0])
    ratios = coef["band_ratios"]
    return Estimate(typical * ratios["min_over_typ"], typical, typical * ratios["max_over_typ"], warn, "our model")
