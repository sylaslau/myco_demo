"""Input CSVs for notebooks/0.1.presentation_visualisations.ipynb, built from files already in the repo.

Nothing is invented: every value comes from an existing file or from the 2.2/2.5 package code. A row that lacks a
required value is dropped (the notebook raises on blanks). Skipped curves are returned so the caller can report them.
masters_mix is left out of the project (Sylas 10-19).
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mycodemo import pipeline
from mycodemo.cultivation import FRUITING_CONDITIONS, baseline_guide
from mycodemo.growth import consensus_estimate
from mycodemo.growth.forecast_update import DEFAULT_UPDATE_RULES, LOG9, logistic_coverage

SCHEMAS = {
    "model_metrics": "dataset split model metric value run",
    "observations": "dataset experiment observation_id genus species oyster_group temperature_c days colonisation_pct",
    "baseline_curves": "dataset genus species oyster_group temperature_c days predicted_pct model",
    "heldout_predictions": "dataset split model observation_id days observed_pct predicted_pct",
    "stages": "dataset experiment temperature_c species stage start_day end_day",
    "events": "dataset experiment temperature_c species event day",
    "app_timeline": "species common_name temperature_c item kind start_day end_day in_range source",
}
ESTIMATE = "Colonisation time estimate"
FORECAST = "Colonisation forecast"
LITERATURE = "literature consensus"
GROUPS = {"straw", "cotton", "sawdust", "bagasse", "corncob", "logs"}   # the 2.2 groups, masters_mix excluded
OMITTED = {"masters_mix"}
# Sylas 10-19 named P. djamor as tropical; every other Pleurotus in the data is treated as cool. Not inferred from a
# climate rule: extend this set only when the user says so.
TROPICAL_SPECIES = {"Pleurotus djamor"}
SLIDER_C = range(18, 31)              # colonisation figures: 1 °C steps, about the range the 2.2 model covers
TIMELINE_C = range(15, 33)            # app timeline: 1 °C steps
TIMELINE_SUBSTRATE = "sawdust"        # the app's default substrate (baseline_guide.FALLBACK_SUBSTRATE)
SENSOR_FILE_GLOB = "data/raw/syldd_2026/sensors/*.csv"


def substrate_group(name):
    """2.2 substrate group for a substrate name, or None when there is no clear match (mixtures never match)."""
    name = str(name).strip().lower()
    if " + " in name:
        return None
    name = consensus_estimate.SUBSTRATE_ALIASES.get(name, name)
    if name in OMITTED:
        return None
    if name in GROUPS:
        return name
    return "straw" if name.endswith("straw") else None


def oyster_group(species):
    species = str(species)
    if not species.startswith("Pleurotus"):
        return "not_oyster"
    return "tropical" if species in TROPICAL_SPECIES else "cool"


def _dataset_label(dataset, substrate):
    """Dataset ID plus the substrate name from the sample/substrate tables; just the ID when there is none."""
    substrate = "" if pd.isna(substrate) else str(substrate).strip()
    return f"{dataset} · {substrate}" if substrate else dataset


def _clean(frame, name):
    columns = SCHEMAS[name].split()
    frame = frame[columns].dropna().reset_index(drop=True)
    return frame


def _syldd_temperature(root):
    values = []
    for path in sorted(Path(root).glob(SENSOR_FILE_GLOB)):
        sensor = pd.read_csv(path)
        values.append(pd.to_numeric(sensor["Temperature_Celsius(°C)"], errors="coerce").dropna())
    return float(pd.concat(values).mean()) if values else None


def build_observations(root):
    ex = Path(root) / "data/interim/extracted"
    m = pd.read_csv(ex / "measurement.csv").dropna(subset=["day_of_growth", "surface_colonised_pct"])
    m = m[m.dataset_id != "d002"]
    s = pd.read_csv(ex / "sample.csv")[["sample_id", "substrate_id", "fermentation_id"]]
    f = pd.read_csv(ex / "fermentation.csv")[["fermentation_id", "genus", "species", "incubation_temp_c"]]
    sub = pd.read_csv(ex / "substrate.csv")[["substrate_id", "plant_name"]]
    g = m.merge(s, on="sample_id", how="left").merge(f, on="fermentation_id", how="left").merge(sub, on="substrate_id", how="left")
    g["species"] = g.genus + " " + g.species
    g["group"] = g.plant_name.map(substrate_group)
    g["oyster_group"] = g.species.map(oyster_group)
    rows = pd.DataFrame({
        "dataset": [_dataset_label(d, x) if sp.startswith("Pleurotus") else d
                    for d, x, sp in zip(g.dataset_id, g.plant_name, g.species.fillna(""))],
        "experiment": g.sample_id, "observation_id": g.sample_id + "-day" + g.day_of_growth.astype(str),
        "genus": g.genus, "species": g.species, "oyster_group": g.oyster_group, "temperature_c": g.incubation_temp_c,
        "days": g.day_of_growth, "colonisation_pct": g.surface_colonised_pct, "group": g.group})
    cov = pd.read_csv(Path(root) / "data/interim/coverage_syldd.csv")
    temp = _syldd_temperature(root)
    syldd = pd.DataFrame({
        "dataset": "syldd", "experiment": cov.container_id, "observation_id": cov.container_id + "-day" + cov.day.astype(str),
        "genus": "Pleurotus", "species": "Pleurotus ostreatus", "oyster_group": "cool", "temperature_c": temp,
        "days": cov.day, "colonisation_pct": cov.coverage * 100, "group": None})
    return pd.concat([rows, syldd], ignore_index=True)


def build_baseline_curves(root, bundle=None):
    """2.2 baseline curve for every Pleurotus species in the bundle at every slider temperature (same call as the app,
    air temperature, logistic rule: 90 % at t90, midpoint 0.60 x t90). The substrate is the species' main group: the
    first one listed for it in reference_reproduction.csv. Returns (curves, skipped); skipped lists
    (species, substrate, temperature, reason)."""
    bundle = bundle or consensus_estimate.find_bundle(Path(root) / "models/colonisation")
    cm = consensus_estimate.load_bundle(bundle)
    midpoint = DEFAULT_UPDATE_RULES["curve_midpoint_fraction"]
    ref = pd.read_csv(bundle / "reference_reproduction.csv")
    frames, skipped = [], []
    for species, g in ref[ref.species.str.startswith("Pleurotus")].groupby("species", sort=False):
        group = g.substrate_code.iloc[0]
        if group not in GROUPS:
            skipped.append((species, group, None, "no 2.2 substrate group"))
            continue
        for temp in SLIDER_C:
            try:
                adjusted = cm.estimate_days(species, group, temperature_c=float(temp), temperature_basis="air")["adjusted_t90_min_typ_max"]
            except ValueError as exc:
                skipped.append((species, group, temp, str(exc)))
                continue
            if adjusted is None:
                skipped.append((species, group, temp, "no numeric adjusted estimate at this temperature"))
                continue
            t90 = adjusted[1]
            slope = LOG9 / ((1 - midpoint) * t90)
            days = np.round(np.arange(0, np.ceil(2 * t90 * 1.5) / 2 + 0.5, 0.5), 2)
            frames.append(pd.DataFrame({
                "dataset": "2.2 baseline · " + group, "genus": "Pleurotus", "species": species,
                "oyster_group": oyster_group(species), "temperature_c": float(temp), "days": days,
                "predicted_pct": 100 * logistic_coverage(days, t90, slope), "model": ESTIMATE}))
    curves = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMAS["baseline_curves"].split())
    return curves, skipped


def build_app_timeline(root=None):
    """What the app shows a user, per species x temperature, by calling the app's own code (pipeline.colonisation_estimate,
    colonisation_forecast with no photos, dates_step, fruiting_estimate) on a Food batch with the sawdust recipe.
    Days are since inoculation; an event holds the app's range (start_day-end_day) and is drawn at its middle.
    Species the app cannot show (no matching FRUITING_CONDITIONS entry) are returned in skipped."""
    base = baseline_guide.load()
    rows, skipped = [], []
    for species, g in base.groupby("species", sort=False):
        common = g.common_name.iloc[0]
        entries = [s for s in FRUITING_CONDITIONS if s["latin"] == species]
        entry = next((s for s in entries if s["name"] == common), entries[0] if entries else None)
        if entry is None:
            skipped.append((species, common, "not in the app's species list (FRUITING_CONDITIONS)"))
            continue
        for temp in TIMELINE_C:
            batch = pipeline.Batch(goal="Food", choice=entry["name"], recipe={"preset": "suppsaw"},
                                   temp_c=float(temp), inoc_date=date.today())
            row, row_note = pipeline.baseline_row(batch)
            est = pipeline.colonisation_estimate(batch, float(temp))
            fc = pipeline.colonisation_forecast(batch, est)
            dates = pipeline.dates_step(batch, fc)
            if est.value is None or not dates.value:
                skipped.append((species, common, f"no estimate at {temp} °C"))
                continue
            inc = baseline_guide.colonisation(row)["incubation_temp_C"]
            in_range = inc["min"] <= temp <= inc["max"]
            source = f"{fc.source}; {row_note}"
            ev = {n: ((a - batch.inoc_date).days, (b - batch.inoc_date).days) for n, a, b in dates.value}
            common_row = dict(species=species, common_name=common, temperature_c=float(temp), in_range=in_range, source=source)
            full, marker = ev[pipeline.ENDPOINT_LABEL], ev.get(pipeline.MARKER_LABEL)
            mid = lambda r: (r[0] + r[1]) / 2
            events = [("Inoculation", ev["Inoculation"]), ("Fully colonised", full)]
            stages = [("Colonisation", 0.0, mid(full))]
            if marker:
                stages += [("Induction planning marker (2 days)", mid(full), mid(marker)),
                           ("Fruiting (start induction to first harvest)", mid(marker), mid(ev["First harvest"]))]
                events += [("Start induction", marker), ("Pinning", ev["Pinning"]), ("First harvest", ev["First harvest"])]
            rows += [{**common_row, "item": i, "kind": "stage", "start_day": a, "end_day": b} for i, a, b in stages]
            rows += [{**common_row, "item": i, "kind": "event", "start_day": r[0], "end_day": r[1]} for i, r in events]
    return pd.DataFrame(rows, columns=SCHEMAS["app_timeline"].split()), skipped


def build_model_metrics(root, bundle):
    root = Path(root)
    omitted_refs = _omitted_refs(bundle)
    cv = pd.read_csv(bundle / "reference_reconstruction_cv.csv")
    cv = cv[~cv.ref_id.isin(omitted_refs)]
    parts = [pd.DataFrame({
        "dataset": "literature consensus (reconstruction of book values, not real-world validation)",
        "split": "reference reconstruction", "model": ESTIMATE, "metric": "absolute_error_days",
        "value": cv.absolute_error_days, "run": bundle.name + " · " + cv.ref_id})]
    runs = root / "reports/training_runs"
    fv = pd.read_csv(runs / "colonisation_forecast_20261014/error_vs_photos.csv")
    for column, model in (("err_model", FORECAST), ("err_baseline", FORECAST + " baseline")):
        parts.append(pd.DataFrame({"dataset": "colonisation forecast test (2026-10-14)", "split": "n_photos", "model": model,
                                   "metric": "forecast_error", "value": fv[column], "run": fv.n_photos.astype(str)}))
    fs = pd.read_csv(runs / "property_model_20261014/fold_scores.csv")
    parts.append(pd.DataFrame({"dataset": "property model (2026-10-14)", "split": "held_out_lab",
                               "model": "Material strength · " + fs.model, "metric": "fold_error",
                               "value": fs.fold_error, "run": fs.held_out_lab}))
    for folder, label in sorted((p, "run " + p.name.split("_")[2]) for p in runs.glob("harvest_run2_*")):
        last = pd.read_csv(folder / "epochs.csv").iloc[-1]
        for metric in ("metrics/mAP50(B)", "metrics/mAP50(M)"):
            parts.append(pd.DataFrame({"dataset": "harvest readiness validation photos", "split": label,
                                       "model": "Harvest readiness", "metric": metric, "value": [last[metric]],
                                       "run": folder.name}))
    return pd.concat(parts, ignore_index=True)


def _omitted_refs(bundle):
    refs = pd.read_csv(bundle / "prepared_consensus.csv")
    return set(refs.loc[refs.substrate_code.isin(OMITTED), "ref_id"])


def build_stages(root, bundle):
    ref = pd.read_csv(bundle / "reference_reproduction.csv")
    ref = ref[~ref.substrate_code.isin(OMITTED)]
    base = pd.read_csv(Path(root) / "data/processed/cultivation_baseline.csv")[["ref_id", "incubation_temp_opt_C"]]
    ref = ref.merge(base, on="ref_id", how="left")
    post = consensus_estimate.load_bundle(bundle).post_90_days
    colonised = ref.t90_typical + post
    label = ref.species + " · " + ref.substrate_code
    common = {"dataset": LITERATURE, "experiment": "reference reproduction (literature typical values)",
              "temperature_c": ref.incubation_temp_opt_C, "species": label}
    return pd.concat([
        pd.DataFrame({**common, "stage": "colonisation", "start_day": 0.0, "end_day": colonised}),
        pd.DataFrame({**common, "stage": "pinning", "start_day": colonised, "end_day": colonised + ref.pinning_typical}),
        pd.DataFrame({**common, "stage": "development", "start_day": colonised + ref.pinning_typical,
                      "end_day": colonised + ref.pinning_typical + ref.development_typical}),
    ], ignore_index=True)


def build_events(bundle):
    ctx = pd.read_csv(bundle / "context_only.csv").dropna(subset=["endpoint_day"])
    return pd.DataFrame({"dataset": ctx.dataset_id, "experiment": ctx.sample_id, "temperature_c": ctx.temperature_c,
                         "species": ctx.species, "event": "reported full colonisation", "day": ctx.endpoint_day})


def build_presentation_data(root):
    """Write the seven CSVs into <root>/presentation_data/. Returns (row counts, skipped curves and timeline species)."""
    root = Path(root)
    out = root / "presentation_data"
    out.mkdir(exist_ok=True)
    bundle = consensus_estimate.find_bundle(root / "models/colonisation")
    observations = build_observations(root)
    curves, skipped = build_baseline_curves(root, bundle)
    timeline, skipped_timeline = build_app_timeline()
    tables = {
        "observations": _clean(observations, "observations"),
        "baseline_curves": _clean(curves, "baseline_curves"),
        "model_metrics": _clean(build_model_metrics(root, bundle), "model_metrics"),
        "stages": _clean(build_stages(root, bundle), "stages"),
        "events": _clean(build_events(bundle), "events"),
        "app_timeline": _clean(timeline, "app_timeline"),
        "heldout_predictions": pd.DataFrame(columns=SCHEMAS["heldout_predictions"].split()),
    }
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    return {name: len(frame) for name, frame in tables.items()}, skipped + skipped_timeline
