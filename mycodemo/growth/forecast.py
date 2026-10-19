"""Colonisation forecast: fit an S-curve to % colonised per day and forecast the 90% / 95% day."""

import html
import io
import json
import warnings
from datetime import date, datetime, timedelta

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


def biological_efficiency(fresh_g, dry_g):
    """Measured biological efficiency in % = fresh harvest / dry substrate x 100 (NaN if either is missing)."""
    if not fresh_g or not dry_g or dry_g <= 0:
        return np.nan
    return 100 * fresh_g / dry_g


def loss_rate_text(lost, total):
    """Loss rate with its denominator, e.g. '2 of 13 containers lost'."""
    return f"{lost} of {total} containers lost"


def cost_summary(costs, harvest_g, price_per_kg=None):
    """Batch costs: {'total', 'per_kg', 'margin_per_kg'}; per_kg and margin are NaN until there is a harvest
    (margin also needs a selling price)."""
    total = float(sum(costs.get(k) or 0 for k in ("substrate", "spawn", "supplies", "labour")))
    per_kg = total / (harvest_g / 1000) if harvest_g and harvest_g > 0 else np.nan
    margin = price_per_kg - per_kg if price_per_kg is not None and not np.isnan(per_kg) else np.nan
    return {"total": total, "per_kg": per_kg, "margin_per_kg": margin}


# --- backup / restore and printable labels -----------------------------------------------------
BACKUP_KIND = "mycodemo-grow-log"
BACKUP_VERSION = 1
BACKUP_SECTIONS = ["dataset", "dish", "batch", "dish_fields", "stage_log", "photos",
                   "harvest", "dry_g", "losses", "costs"]
CSV_REQUIRED = ["batch_label", "species", "substrate", "spawn_pct", "inoculation_date",
                "containers_n", "container_id", "day", "coverage_pct"]


def batch_key(cid, is_syldd):
    """Batch of a container: 'B-WS05' for syldd IDs like SYLDD-WS05-D1, else 'reference'."""
    parts = str(cid).split("-")
    return f"B-{parts[1]}" if is_syldd and len(parts) >= 3 else "reference"


def _day(x, what):
    try:
        return date.fromisoformat(str(x))
    except ValueError:
        raise ValueError(f"Backup has a bad date for {what}: {x!r}") from None


def _plain(o):
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"cannot save {type(o).__name__}")


def backup_to_json(payload):
    """Session log (dict with BACKUP_SECTIONS) -> JSON text."""
    doc = {"kind": BACKUP_KIND, "version": BACKUP_VERSION,
           "created": datetime.now().isoformat(timespec="seconds"), **payload}
    return json.dumps(doc, default=_plain, indent=1)


def backup_from_json(text):
    """JSON text -> session log with real dates. Raises ValueError with a plain message if it does not match."""
    try:
        doc = json.loads(text)
    except ValueError:
        raise ValueError("This file is not valid JSON, so it is not a Grow Log backup.") from None
    if not isinstance(doc, dict) or doc.get("kind") != BACKUP_KIND:
        raise ValueError("This JSON file is not a Grow Log backup (wrong or missing 'kind').")
    if doc.get("version") != BACKUP_VERSION:
        raise ValueError(f"Backup version {doc.get('version')!r} is not supported (expected {BACKUP_VERSION}).")
    missing = [k for k in BACKUP_SECTIONS if k not in doc]
    if missing:
        raise ValueError("Backup is missing: " + ", ".join(missing) + ".")
    if doc["dataset"] not in ("syldd", "ref"):
        raise ValueError(f"Backup has an unknown dataset: {doc['dataset']!r}.")
    try:
        p = {k: doc[k] for k in BACKUP_SECTIONS}
        p["batch"] = {**p["batch"], "inoc_date": _day(p["batch"]["inoc_date"], "inoculation")}
        p["stage_log"] = {sid: [{**e, "date": _day(e["date"], f"stage of {sid}")} for e in v]
                          for sid, v in p["stage_log"].items()}
        p["harvest"] = {sid: [{"date": _day(e["date"], f"harvest of {sid}"), "g": float(e["g"])} for e in v]
                        for sid, v in p["harvest"].items()}
        p["losses"] = {sid: {**v, "date": _day(v["date"], f"loss of {sid}")} for sid, v in p["losses"].items()}
        p["photos"] = [{"series_id": r["series_id"], "day": r["day"], "coverage": r["coverage"],
                        "file": r["file"]} for r in p["photos"]]
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        if isinstance(e, ValueError) and str(e).startswith("Backup has a bad date"):
            raise
        raise ValueError(f"Backup does not have the expected structure ({type(e).__name__}: {e}).") from None
    if doc.get("pipeline"):                  # optional: the batch record (older backups have none)
        p["pipeline"] = doc["pipeline"]
    if doc.get("environments"):              # optional: name -> environment (older backups have none)
        p["environments"] = doc["environments"]
    p["replace"] = True
    return p


def _nn(x):
    """NaN / missing -> None, numpy scalar -> Python."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return x.item() if hasattr(x, "item") else x


def csv_to_payload(text):
    """Card 2 CSV (one row per photo) -> session log for its containers. Raises ValueError if it does not match.
    Stage dates are rebuilt from the stage shown on each photo; photo rows stay with the dataset."""
    try:
        df = pd.read_csv(io.StringIO(text))
    except (ValueError, pd.errors.ParserError):
        raise ValueError("This file could not be read as a Grow Log CSV.") from None
    missing = [c for c in CSV_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("This CSV does not match the Grow Log download; missing columns: "
                         + ", ".join(missing) + ".")
    if df.empty:
        raise ValueError("This CSV has no rows.")
    first = df.iloc[0]
    inoc = _day(first["inoculation_date"], "inoculation")
    cids = list(dict.fromkeys(df["container_id"].astype(str)))
    is_syldd = cids[0].upper().startswith("SYLDD")
    temp = _nn(first.get("incubation_temp_c"))
    room = str(temp) == "room"
    label = _nn(first["batch_label"])
    p = {"dataset": "syldd" if is_syldd else "ref", "dish": cids[0], "replace": False,
         "batch": {"batch_label": "" if label is None else str(label),
                   "species": str(first["species"]), "inoc_date": inoc,
                   "particle_mm": _nn(first.get("particle_size_mm")), "n_containers": int(first["containers_n"]),
                   "room_temp": room, "temp_c": None if room or temp is None else float(temp),
                   "rh_pct": _nn(first.get("rh_pct"))},
         "dish_fields": {}, "stage_log": {}, "photos": [], "harvest": {}, "dry_g": {}, "losses": {}, "costs": {}}
    for cid in cids:
        d = df[df["container_id"].astype(str) == cid].sort_values("day")
        r = d.iloc[0]
        p["dish_fields"][cid] = {"substrate": str(r["substrate"]), "spawn_pct": float(r["spawn_pct"])}
        log, prev = [], None
        for _, row in d.iterrows():
            stage = _nn(row.get("stage"))
            if stage and stage != prev:
                note = _nn(row.get("stage_note")) or ""
                log.append({"stage": stage, "date": inoc + timedelta(days=int(row["day"])), "note": str(note)})
                prev = stage
        if log:
            p["stage_log"][cid] = log
        if _nn(r.get("harvest_flushes")):
            try:
                p["harvest"][cid] = [{"date": _day(a.split(":")[0], f"harvest of {cid}"), "g": float(a.split(":")[1])}
                                     for a in str(r["harvest_flushes"]).split(";") if a.strip()]
            except (IndexError, ValueError):
                raise ValueError(f"The harvest_flushes column for {cid} is not in 'date:grams; ...' form.") from None
        if _nn(r.get("dry_substrate_g")):
            p["dry_g"][cid] = float(r["dry_substrate_g"])
        if str(r.get("lost")) == "True":
            p["losses"][cid] = {"stage": str(r["loss_stage"]), "cause": str(_nn(r["loss_cause"]) or "unknown"),
                                "date": _day(r["loss_date"], f"loss of {cid}")}
        if "currency" in df.columns:
            p["costs"][batch_key(cid, is_syldd)] = {
                "currency": str(r["currency"]), "price": _nn(r.get("price_per_kg")),
                **{k: _nn(r.get(f"cost_{k}")) for k in ("substrate", "spawn", "supplies", "labour")}}
    return p


def restore_from_text(text):
    """Backup JSON or card 2 CSV text -> session log; ValueError with a clear message if neither matches."""
    return backup_from_json(text) if text.lstrip().startswith("{") else csv_to_payload(text)


LABELS_PER_PAGE = 8
LABEL_CSS = """
@page { size: A4; margin: 10mm; }
body { font-family: sans-serif; margin: 0; }
.note { padding: 6px; font-size: 12px; }
.page { display: grid; grid-template-columns: 95mm 95mm; grid-auto-rows: 69mm; gap: 0;
        page-break-after: always; }
.label { box-sizing: border-box; border: 0.3mm dashed #888; padding: 4mm 5mm; overflow: hidden; }
.label h1 { font-size: 18pt; margin: 0 0 1mm; } .label h2 { font-size: 14pt; margin: 0 0 3mm; }
.label p { font-size: 10.5pt; margin: 1.2mm 0; } .small { font-size: 8pt; color: #444; }
.flush { margin-top: 5mm; font-size: 12pt; }
@media print { .note { display: none; } }
"""


def build_labels_html(labels):
    """Printable A4 page(s), 8 labels per page. Each label: batch, container, species, inoculation,
    expected (dates as text), expected_note (optional)."""
    e = html.escape
    pages = []
    for i in range(0, max(len(labels), 1), LABELS_PER_PAGE):
        cells = "".join(
            f'<div class="label"><h1>{e(lb["batch"])}</h1><h2>{e(lb["container"])}</h2>'
            f'<p>Species: {e(lb["species"])}</p><p>Inoculated: {e(lb["inoculation"])}</p>'
            f'<p>Expected colonisation: {e(lb["expected"])}</p>'
            + (f'<p class="small">{e(lb["expected_note"])}</p>' if lb.get("expected_note") else "")
            + '<p class="flush">flush: ____________</p></div>'
            for lb in labels[i:i + LABELS_PER_PAGE])
        pages.append(f'<div class="page">{cells}</div>')
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Container labels</title>"
            f"<style>{LABEL_CSS}</style></head><body>"
            "<div class='note'>Print at 100% scale on A4 (Ctrl+P), default margins.</div>"
            + "".join(pages) + "</body></html>")
