"""Monitor: one box per environment (residence) with its conditions, and the batches growing there."""

import io
from datetime import date

import streamlit as st

from mycodemo import pipeline
from mycodemo.app import flow
from mycodemo.cultivation import baseline_guide
from mycodemo.growth.coverage import days_since, photo_coverage, photo_datetime, photo_quality
from mycodemo.growth.forecast import restore_from_text

st.set_page_config(page_title="Monitor", page_icon="📷", layout="wide")
flow.init()
st.title("Monitor")

# session-only stores (nothing is written to disk): stage_log {dish: [{stage, date, note}]},
# harvest {dish: [{date, g}]}, losses {dish: {...}}, dry_g {dish: g}, costs {batch: {...}},
# batch_fields {dataset: {...}}, dish_fields {dish: {...}}, upload_rows / restored_photos {key: [photo rows]},
# environments {name: pipeline.Environment} (kept for backups), monitor_env {residence: conditions},
# sample_photos {sample ID: [photo rows]}
ss = st.session_state
STAGE_LOG = ss.setdefault("stage_log", {})
HARVEST = ss.setdefault("harvest", {})
LOSSES = ss.setdefault("losses", {})
DRY_G = ss.setdefault("dry_g", {})
COSTS = ss.setdefault("costs", {})
BATCH_FIELDS = ss.setdefault("batch_fields", {})
DISH_FIELDS = ss.setdefault("dish_fields", {})
UPLOAD_ROWS = ss.setdefault("upload_rows", {})
RESTORED_PHOTOS = ss.setdefault("restored_photos", {})
ENVIRONMENTS = ss.setdefault("environments", {})
SAMPLE_PHOTOS = ss.setdefault("sample_photos", {})
MONITOR_ENV = ss.setdefault("monitor_env", {})
RADIO = {"ref": "d005 (reference)", "syldd": "syldd (my batch)"}
WIDGET_PREFIXES = ("b_", "h_", "l_", "c_")      # their values come from the stores above after a restore
PLAN = ss["batch"]


def apply_restore(p):
    """Refill the stores from a restored log; a backup replaces everything, a CSV only its own containers."""
    if p["replace"]:
        for store in (STAGE_LOG, HARVEST, LOSSES, DRY_G, COSTS, DISH_FIELDS, UPLOAD_ROWS, RESTORED_PHOTOS):
            store.clear()
    STAGE_LOG.update(p["stage_log"])
    HARVEST.update(p["harvest"])
    LOSSES.update(p["losses"])
    DRY_G.update(p["dry_g"])
    COSTS.update(p["costs"])
    DISH_FIELDS.update(p["dish_fields"])
    BATCH_FIELDS[p["dataset"]] = p["batch"]
    if p["replace"]:
        ENVIRONMENTS.clear()
    ENVIRONMENTS.update({k: pipeline.Environment.from_dict(v) for k, v in p.get("environments", {}).items()})
    for r in p["photos"]:
        if str(r["file"]).startswith("upload:"):
            RESTORED_PHOTOS.setdefault(r["series_id"], []).append(r)
    for k in [k for k in ss if k.startswith(WIDGET_PREFIXES)]:
        del ss[k]                                # widgets restart from the restored values
    ss["dataset_radio"] = RADIO[p["dataset"]]
    ss["dish_select"] = p["dish"]
    if p.get("pipeline"):                        # the batch record (older backups and CSVs have none)
        pb = pipeline.Batch.from_dict(p["pipeline"])
        ss["pipeline_batch"] = pb
        if pb.goal:                              # the plan keys the pages share follow the restored batch
            ss.update(goal=pb.goal, choice=pb.choice, custom_name=pb.custom_name, environment=pb.environment,
                      temp_c=pb.temp_c, recipe=pb.recipe, inoc_date=pb.inoc_date)


# --- helpers -------------------------------------------------------------------------------
# book ranges for CO2 (ppm): colonisation tolerates a lot, fruiting wants fresh air
CO2_COLONISING, CO2_FRUITING = (0.0, 20000.0), (500.0, 1000.0)
LIGHTS = {"green": "#2e9e4f", "red": "#d64545", "yellow": "#e0a800"}
STAGE_ROWS = [["Spawning", "Colonising", "Fruiting"], ["Ready for induction", "Ready for harvest"]]


def residences():
    """Residences entered on the Environment page, in that page's order."""
    names = [n for n in ("Plastic container", "Shelf or vitrine", "Tent", "Room")
             if n in ss.get("env_size", {}) or n == ss.get("env_where")]
    return names


def current_stage(batch, today=None):
    """Where the batch is now, from the pipeline's dates (inoculation, 90% day, induction marker, pinning, first harvest)."""
    today = today or date.today()
    ev = {name: (a, b) for name, a, b in pipeline.run(batch).dates.value}
    if today <= batch.inoc_date:
        return "Spawning"
    if pipeline.ENDPOINT_LABEL not in ev or pipeline.MARKER_LABEL not in ev:        # Materials or no dates
        return "Colonising"
    if today < ev[pipeline.MARKER_LABEL][0]:
        return "Colonising"
    if today < ev["Pinning"][0]:
        return "Ready for induction"
    if today < ev["First harvest"][0]:
        return "Fruiting"
    return "Ready for harvest"


def wanted_ranges(batch, stage):
    """{'temp': (lo, hi, optimum), 'rh': (lo, hi, target), 'co2': (lo, hi, target)} from the baseline, or None."""
    br = pipeline.baseline_row(batch)
    if br is None:
        return None
    if stage in ("Spawning", "Colonising"):
        c = baseline_guide.colonisation(br[0])
        t, lo_h, hi_h = c["incubation_temp_C"], *pipeline.HUMIDITY_RANGE
        return {"temp": (t["min"], t["max"], t["opt"]), "rh": (lo_h, hi_h, float(c["incubation_rh_pct"])),
                "co2": (*CO2_COLONISING, 20000.0)}
    f = baseline_guide.fruiting(br[0])
    t, h = f["fruiting_temp_C"], f["fruiting_rh_pct"]
    return {"temp": (t["min"], t["max"], t["opt"]), "rh": (h["min"], h["max"], (h["min"] + h["max"]) / 2),
            "co2": (*CO2_FRUITING, 750.0)}


def light(value, rng):
    """green in range, red out of range, yellow when not entered or no range is known."""
    if value is None or rng is None:
        return "yellow"
    return "green" if rng[0] <= value <= rng[1] else "red"


def worst(colours):
    return "red" if "red" in colours else "yellow" if "yellow" in colours else "green"


def dot(colour):
    return f'<span style="color:{LIGHTS[colour]}">●</span>'


def bubble(colour, icon, label, text):
    return (f'<div style="background:{LIGHTS[colour]}22;border:2px solid {LIGHTS[colour]};border-radius:999px;'
            f'padding:0.2rem 0.9rem;margin-bottom:0.3rem;display:inline-block">{icon} <b>{label}</b> {text}</div>')


@st.cache_data(show_spinner=False, max_entries=64)
def analyse_photo(data: bytes):
    """Photo coverage (method choice) and quality check of one photo."""
    buf = io.BytesIO(data)
    cov_ = photo_coverage(buf)
    return {"coverage": cov_["coverage"], "method": cov_["method"],
            "quality": ", ".join(photo_quality(buf)["reasons"])}


def batch_record():
    """The one pipeline batch, with every sample photo and the stage log copied in."""
    b = flow.get_batch()
    if b.goal is None:                           # no plan from the earlier pages: the page's own default
        b.goal, b.choice = "Food", "Blue Oyster"
    if PLAN:
        b.inoc_date, b.temp_c = PLAN["inoc_date"], PLAN["temp_c"]
    b.photos = [pipeline.Photo(day=r["day"], file=r["file"], coverage=r["coverage"], method=r["method"],
                               quality=r["quality"]) for rows in SAMPLE_PHOTOS.values() for r in rows]
    return b


# --- sidebar: the environment list above Restore --------------------------------------------
with st.sidebar:
    env_list = st.container()
    st.header("Restore")
    restore_file = st.file_uploader("Restore from backup", type=["json", "csv"], key="restore_file",
                                    help="A 'Download backup (JSON)' file, or the 'Download my log (CSV)' file.")
    if restore_file is not None:
        try:
            restored = restore_from_text(restore_file.getvalue().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as err:
            st.error(f"Could not restore {restore_file.name}: {err}")
        else:
            if ss.get("restored_id") != restore_file.file_id:
                apply_restore(restored)
                ss["restored_id"] = restore_file.file_id
            st.success(f"Restored from {restore_file.name}.")

# --- main area: one box per environment ---------------------------------------------------
batch = batch_record()
stage = current_stage(batch) if PLAN else "Colonising"
want = wanted_ranges(batch, stage)
home = ss.get("env_where")                       # the residence the plan batch grows in
side_lights = {}
for name in residences():
    with st.container(border=True):
        st.subheader(name)
        has_batch = PLAN and name == home
        env_stage = stage if has_batch else "Colonising"
        rng = wanted_ranges(batch, env_stage)
        saved = MONITOR_ENV.setdefault(name, {})
        kind = "grow" if env_stage in ("Spawning", "Colonising") else "fruit"   # prefill follows the stage
        fields = [("temp", "🌡️", "Temperature", "°C", 0.5, "%.1f"), ("rh", "💧", "Humidity", "%", 1.0, "%.0f"),
                  ("co2", "🫧", "CO2", "ppm", 100.0, "%.0f")]
        cols = st.columns(len(fields))
        values, colours = {}, []
        for col, (key, icon, label, unit, step, fmt) in zip(cols, fields):
            holder = col.empty()
            default = float(rng[key][2]) if rng else None
            values[key] = col.number_input(f"{label} ({unit})", value=default, step=step, format=fmt,
                                           key=f"mon_{key}_{name}_{kind}", label_visibility="collapsed")
            colour = light(values[key], rng[key][:2] if rng else None)
            colours.append(colour)
            holder.markdown(bubble(colour, icon, label,
                                   "not entered" if values[key] is None else f"{values[key]:g} {unit}"),
                            unsafe_allow_html=True)
        m_col, s_col = st.columns([3, 1], vertical_alignment="bottom")
        measured = m_col.selectbox("Measured by", pipeline.MEASURED_BY, index=2, key=f"mon_by_{name}")
        if s_col.button("Save", key=f"mon_save_{name}", use_container_width=True):
            saved.update(values, measured_by=measured)
            ENVIRONMENTS[name] = pipeline.Environment(name, "Tub", float(values["temp"] if values["temp"] is not None
                                                                         else 21.0), values["rh"], measured)
        side_lights[name] = worst(colours)

        if has_batch:                            # the plan batch grows in this residence
            with st.container(border=True):
                st.markdown(f"**{PLAN['batch_label']}**")
                for row in STAGE_ROWS:
                    for col, s in zip(st.columns(len(STAGE_ROWS[0])), row):
                        col.button(s, key=f"stage_{name}_{s}",
                                   type="primary" if s == stage else "secondary", use_container_width=True)
                n = int(PLAN["n_containers"])
                samples = [f"{PLAN['batch_label']}-{i:02d}" for i in range(1, n + 1)]
                sample = st.selectbox("Sample (one bag or jar)", samples, key=f"sample_{name}")
                up = st.file_uploader("Photo of this sample", type=["jpg", "jpeg", "png"], key=f"photo_{name}_{sample}")
                if up is not None:
                    try:
                        taken = photo_datetime(up).date()
                    except ValueError:
                        taken = date.today()
                    a_ = analyse_photo(up.getvalue())
                    row = {"series_id": sample, "day": days_since(taken, batch.inoc_date), "file": f"upload: {up.name}",
                           "coverage": a_["coverage"], "method": a_["method"], "quality": a_["quality"]}
                    rows = SAMPLE_PHOTOS.setdefault(sample, [])
                    if not any((r["file"], r["day"]) == (row["file"], row["day"]) for r in rows):
                        rows.append(row)
                    ss["pipeline_result"] = pipeline.run(batch_record())     # Photo coverage -> Colonisation forecast
                    st.success(f"Photo saved under {sample}.")

if not residences():
    st.info("Choose where you grow on the Environment page first.")
    st.page_link("pages/2_Environment.py", label="Environment", icon="📦")

with env_list:
    for name, colour in side_lights.items():
        st.markdown(f"{dot(colour)} {name}", unsafe_allow_html=True)
