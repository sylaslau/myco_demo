"""Grow Log & Photos: % colonised per day, S-curve, and the Colonisation forecast as photos are added."""

from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mycodemo.growth.coverage import (
    SYLDD_INOCULATION,
    coverage,
    days_since,
    load_coverage,
    load_syldd_coverage,
    photo_datetime,
)
from mycodemo.growth.forecast import (
    MIN_PHOTOS,
    MODELS,
    backup_to_json,
    batch_key,
    biological_efficiency,
    build_labels_html,
    cost_summary,
    forecast,
    forecast_by_photo,
    forecast_warnings,
    loss_rate_text,
    pace_vs_typical,
    restore_from_text,
)

st.set_page_config(page_title="Grow Log & Photos", page_icon="📷", layout="wide")
st.title("Grow Log & Photos")
st.caption(
    "Each photo gives % colonised (**Photo coverage**). The S-curve fitted to those dots "
    "gives the day the dish reaches 90% and 95% (**Colonisation forecast**)."
)
batch_box = st.container()

# session-only stores (nothing is written to disk): stage_log {dish: [{stage, date, note}]},
# harvest {dish: [{date, g}]}, losses {dish: {stage, cause, date}}, dry_g {dish: g}, costs {batch: {...}},
# batch_fields {dataset: {...}}, dish_fields {dish: {substrate, spawn_pct}},
# upload_rows / restored_photos {dish: [photo rows]}
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
RADIO = {"ref": "d005 (reference)", "syldd": "syldd (my batch)"}
WIDGET_PREFIXES = ("b_", "h_", "l_", "c_")      # their values come from the stores above after a restore


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
    for r in p["photos"]:
        if str(r["file"]).startswith("upload:"):
            RESTORED_PHOTOS.setdefault(r["series_id"], []).append(r)
    for k in [k for k in ss if k.startswith(WIDGET_PREFIXES)]:
        del ss[k]                                # widgets restart from the restored values
    ss["dataset_radio"] = RADIO[p["dataset"]]
    ss["dish_select"] = p["dish"]


with st.sidebar:
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

with st.sidebar:
    st.header("Pick a growth")
    source = st.radio("Dataset", list(RADIO.values()), key="dataset_radio")

is_syldd = source.startswith("syldd")
if is_syldd:
    cov = load_syldd_coverage().rename(columns={"container_id": "series_id"})
    st.info("Photo coverage not yet checked on phone photos (threshold tuned on d005 scans)")
else:
    cov = load_coverage()

dishes = sorted(cov["series_id"].unique())
if ss.get("dish_select") not in dishes:
    ss.pop("dish_select", None)
with st.sidebar:
    sid = st.selectbox("Dish", dishes, key="dish_select")

g = cov[cov["series_id"] == sid].sort_values("day")

# --- own photos (syldd only; kept in this browser session, never written to disk) ---------
if is_syldd:
    with st.sidebar:
        st.subheader(f"Add your own photos to {sid}")
        ups = st.file_uploader("Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        extra = []
        for up in ups:
            try:
                default = photo_datetime(up).date()
            except ValueError:
                default = date.today()
            d = st.date_input(up.name, default, key=f"date_{sid}_{up.file_id}")
            up.seek(0)
            extra.append({"series_id": sid, "day": days_since(d, SYLDD_INOCULATION),
                          "coverage": coverage(up), "file": f"upload: {up.name}"})
        st.caption("Uploads stay in this session only; nothing is saved.")
    if extra:
        UPLOAD_ROWS[sid] = extra
    else:
        UPLOAD_ROWS.pop(sid, None)
    own = {(r["file"], r["day"]): r for r in RESTORED_PHOTOS.get(sid, []) + UPLOAD_ROWS.get(sid, [])}
    if own:
        g = pd.concat([g, pd.DataFrame(list(own.values()))]).sort_values("day")

t, c = g["day"].to_numpy(float), g["coverage"].to_numpy(float)

# --- batch box, stages and CSV (session only) --------------------------------------------
SUBSTRATES = ["wood chips", "sawdust", "straw", "hay", "stem", "other"]
# container ID code -> (substrate, spawn %)
SYLDD_CODES = {"WS05": ("wood chips", 5), "WS10": ("wood chips", 10), "WS20": ("wood chips", 20),
               "HY": ("hay", 10), "CS": ("stem", 10), "WD": ("sawdust", 10)}
STAGES = ["Inoculated", "Colonising", "Fruiting", "Harvested", "Contaminated"]
TERMINAL = {"Harvested", "Contaminated"}
LOSS_STAGES = ["Inoculated", "Colonising", "Fruiting"]
CURRENCIES = ["EUR", "GBP", "USD", "CHF"]
COST_FIELDS = [("substrate", "Substrate"), ("spawn", "Spawn"), ("supplies", "Supplies"), ("labour", "Labour")]


def batch_of(cid):
    return batch_key(cid, is_syldd)


def fresh_g(cid):
    return float(sum(x["g"] for x in HARVEST.get(cid, [])))


def container_be(cid):
    return biological_efficiency(fresh_g(cid), DRY_G.get(cid))


def na(x, fmt="{:.1f}"):
    return "n/a" if x is None or pd.isna(x) else fmt.format(x)


def batch_figures(key, containers):
    """Totals for one batch: containers, lost, harvested g, mean BE %, cost summary."""
    harvest_g = sum(fresh_g(cid) for cid in containers)
    bes = [b for b in (container_be(cid) for cid in containers) if not pd.isna(b)]
    cost = COSTS.get(key, {})
    return {"containers": len(containers), "lost": sum(cid in LOSSES for cid in containers),
            "harvest_g": harvest_g, "mean_be": float(np.mean(bes)) if bes else np.nan,
            "cost": cost_summary(cost, harvest_g, cost.get("price")), "currency": cost.get("currency", "EUR")}


code = str(sid).split("-")[1] if is_syldd and str(sid).count("-") >= 2 else None
pre_substrate, pre_spawn = SYLDD_CODES.get(code, ("wood chips", 10))
ds = "syldd" if is_syldd else "ref"          # widget keys follow the dataset...
dk = f"{ds}_{sid}"                           # ...and the dish, so prefills follow the dish
inoc_default = SYLDD_INOCULATION if is_syldd else date.today()

bf = BATCH_FIELDS.get(ds, {})
df_ = DISH_FIELDS.get(sid, {})
saved_sub = df_.get("substrate", pre_substrate)
start_sub = saved_sub if saved_sub in SUBSTRATES else "other"

with batch_box:
    with st.container(border=True):
        st.subheader("Batch")
        b1, b2, b3 = st.columns(3)
        batch_label = b1.text_input("Batch label", bf.get("batch_label", "syldd" if is_syldd else ""),
                                    key=f"b_label_{ds}")
        species = b2.text_input("Species", bf.get("species", "P. ostreatus"), key=f"b_species_{ds}")
        inoc_date = b3.date_input("Inoculation date", bf.get("inoc_date", inoc_default), key=f"b_inoc_{ds}")
        s1, s2, s3 = st.columns(3)
        substrate = s1.selectbox("Substrate", SUBSTRATES, index=SUBSTRATES.index(start_sub),
                                 key=f"b_sub_{dk}")
        if substrate == "other":
            substrate = s1.text_input("Other substrate", value="" if saved_sub in SUBSTRATES else saved_sub,
                                      key=f"b_sub_other_{dk}").strip() or "other"
        particle_mm = s2.number_input("Particle size (mm), typical size, optional", min_value=0.0,
                                      value=bf.get("particle_mm"), step=0.5, key=f"b_part_{ds}")
        spawn_pct = s3.number_input("Spawn %", 0.0, 100.0, float(df_.get("spawn_pct", pre_spawn)), step=1.0,
                                    key=f"b_spawn_{dk}")
        s3.caption("% of the total mix (spawn / (wet substrate + spawn))")
        m1, m2, m3 = st.columns(3)
        n_containers = m1.number_input("Number of containers", 1, 500, int(bf.get("n_containers", 13 if is_syldd else 1)),
                                       step=1, key=f"b_n_{ds}")
        room_temp = m2.checkbox("Incubation: room, not measured", value=bf.get("room_temp", False),
                                key=f"b_room_{ds}")
        temp_c = None if room_temp else m2.number_input("Incubation temperature (C)", -10.0, 50.0,
                                                        value=bf.get("temp_c"), step=0.5, key=f"b_temp_{ds}")
        rh_pct = m3.number_input("RH (%)", 0.0, 100.0, value=bf.get("rh_pct"), step=1.0, key=f"b_rh_{ds}")
        BATCH_FIELDS[ds] = {"batch_label": batch_label, "species": species, "inoc_date": inoc_date,
                            "particle_mm": particle_mm, "n_containers": int(n_containers),
                            "room_temp": room_temp, "temp_c": temp_c, "rh_pct": rh_pct}
        DISH_FIELDS[sid] = {"substrate": substrate, "spawn_pct": float(spawn_pct)}

        # stages: {dish: [(stage, date), ...]}, session only
        log = STAGE_LOG.setdefault(sid, [])
        st.markdown(f"**Stage of {sid}**")
        cols = st.columns(len(STAGES))
        for col, stage in zip(cols, STAGES):
            if col.button(stage, key=f"stage_{dk}_{stage}", use_container_width=True):
                if stage == "Contaminated":
                    st.session_state["contam_pending"] = sid
                else:
                    st.session_state["contam_pending"] = None
                    log.append({"stage": stage, "date": date.today(), "note": ""})
        if st.session_state.get("contam_pending") == sid:
            note = st.text_input("Note (optional), cause may be 'unknown'", key=f"contam_note_{dk}")
            if st.button("Confirm contaminated", key=f"contam_ok_{dk}"):
                log.append({"stage": "Contaminated", "date": date.today(), "note": note.strip() or "unknown"})
                st.session_state["contam_pending"] = None
                st.rerun()
        if log:
            rows = []
            for i, e in enumerate(log):
                if e["stage"] in TERMINAL:
                    days = 0
                else:
                    end = log[i + 1]["date"] if i + 1 < len(log) else date.today()
                    days = (end - e["date"]).days
                rows.append({"Stage": e["stage"], "Date": e["date"].isoformat(),
                             "Days in stage": days, "Note": e["note"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.caption("No stage recorded yet.")

        # --- harvest, losses, costs (session only) ----------------------------------------
        cur_batch = batch_of(sid)
        batch_containers = sorted(cov[cov["series_id"].map(batch_of) == cur_batch]["series_id"].unique())
        st.markdown(f"**Harvest of {sid}** (fresh weight per flush)")
        h1, h2, h3, h4 = st.columns([1, 1, 1, 1])
        flush_date = h1.date_input("Flush date", date.today(), key=f"h_date_{dk}")
        flush_g = h2.number_input("Fresh weight (g)", 0.0, 100000.0, value=None, step=10.0, key=f"h_g_{dk}")
        if h2.button("Add flush", key=f"h_add_{dk}") and flush_g:
            HARVEST.setdefault(sid, []).append({"date": flush_date, "g": float(flush_g)})
            st.rerun()
        dry_in = h3.number_input("Dry substrate in this container (g)", 0.0, 100000.0,
                                 value=DRY_G.get(sid), step=10.0, key=f"h_dry_{dk}")
        if dry_in:
            DRY_G[sid] = float(dry_in)
        else:
            DRY_G.pop(sid, None)
        h4.metric("Cumulative fresh harvest", f"{fresh_g(sid):.0f} g")
        h4.metric("Biological efficiency (measured)",
                  na(container_be(sid), "{:.1f} %"))
        if HARVEST.get(sid):
            st.dataframe(pd.DataFrame([{"Date": x["date"].isoformat(), "Fresh (g)": x["g"]}
                                       for x in HARVEST[sid]]), hide_index=True)
            if st.button("Remove last flush", key=f"h_del_{dk}"):
                HARVEST[sid].pop()
                st.rerun()
        else:
            st.caption("No harvest recorded yet. Biological efficiency = fresh harvest / dry substrate x 100, "
                       "needs the dry substrate weight.")

        st.markdown(f"**Losses in {cur_batch}**")
        lost_now = LOSSES.get(sid)
        if lost_now:
            st.warning(f"{sid} lost at stage {lost_now['stage']} on {lost_now['date'].isoformat()}; "
                       f"cause: {lost_now['cause']}.")
            if st.button("Undo loss", key=f"l_undo_{dk}"):
                LOSSES.pop(sid)
                st.rerun()
        else:
            l1, l2, l3 = st.columns(3)
            last = [e["stage"] for e in log if e["stage"] in LOSS_STAGES]
            loss_stage = l1.selectbox("Stage at loss", LOSS_STAGES,
                                      index=LOSS_STAGES.index(last[-1]) if last else 0, key=f"l_stage_{dk}")
            loss_cause = l2.text_input("Cause", "unknown", key=f"l_cause_{dk}")
            loss_date = l3.date_input("Loss date", date.today(), key=f"l_date_{dk}")
            if st.button("Record container lost", key=f"l_add_{dk}"):
                LOSSES[sid] = {"stage": loss_stage, "cause": loss_cause.strip() or "unknown", "date": loss_date}
                st.rerun()
        lost_n = sum(cid in LOSSES for cid in batch_containers)
        st.caption(f"Loss rate: {loss_rate_text(lost_n, len(batch_containers))} in {cur_batch}"
                   + (f"; {loss_rate_text(sum(cid in LOSSES for cid in cov['series_id'].unique()), n_containers)} overall"
                      if is_syldd else "") + ".")

        st.markdown(f"**Costs of {cur_batch}**")
        cost = COSTS.setdefault(cur_batch, {})
        k_cur, *k_fields, k_price = st.columns(7)
        cost["currency"] = k_cur.selectbox("Currency", CURRENCIES,
                                           index=CURRENCIES.index(cost.get("currency", "EUR")), key=f"c_cur_{dk}")
        for col, (field, label) in zip(k_fields, COST_FIELDS):
            cost[field] = col.number_input(f"{label} ({cost['currency']})", 0.0, value=cost.get(field),
                                           step=1.0, key=f"c_{field}_{dk}")
        price_in = k_price.number_input(f"Selling price per kg ({cost['currency']}), optional", 0.0,
                                        value=cost.get("price"), step=1.0, key=f"c_price_{dk}")
        cost["price"] = price_in
        fig_b = batch_figures(cur_batch, batch_containers)
        q1, q2, q3 = st.columns(3)
        q1.metric("Total cost", f"{fig_b['cost']['total']:.2f} {cost['currency']}")
        q2.metric("Cost per kg fresh harvest", na(fig_b["cost"]["per_kg"], "{:.2f} " + cost["currency"]))
        q3.metric("Margin per kg", na(fig_b["cost"]["margin_per_kg"], "{:.2f} " + cost["currency"]))
        st.caption(f"Costs are for the whole batch ({fig_b['containers']} containers); cost per kg uses the "
                   f"batch's total fresh harvest ({fig_b['harvest_g']:.0f} g). n/a until there is a harvest.")

        # CSV: one row per photo (all photos, whatever the slider says)
        full = forecast(t, c)
        label_cids = sorted(cov["series_id"].unique()) if is_syldd else [sid]
        day90_by = {}
        for cid in label_cids:
            gg = g if cid == sid else cov[cov["series_id"] == cid].sort_values("day")
            day90_by[cid] = forecast(gg["day"].to_numpy(float), gg["coverage"].to_numpy(float))["day90"]
        csv_rows = []
        for _, r in g.iterrows():
            photo_date = inoc_date + timedelta(days=int(r["day"]))
            past = [e for e in log if e["date"] <= photo_date]
            csv_rows.append({
                "batch_label": batch_label, "species": species, "substrate": substrate,
                "particle_size_mm": particle_mm, "spawn_pct": spawn_pct,
                "inoculation_date": inoc_date.isoformat(), "containers_n": n_containers,
                "incubation_temp_c": "room" if room_temp else temp_c, "rh_pct": rh_pct,
                "container_id": sid, "day": int(r["day"]), "coverage_pct": round(100 * r["coverage"], 2),
                "stage": past[-1]["stage"] if past else "",
                "stage_note": past[-1]["note"] if past else "",
                "forecast_day90": None if np.isnan(full["day90"]) else round(full["day90"], 1),
                "forecast_day95": None if np.isnan(full["day95"]) else round(full["day95"], 1),
                "harvest_flushes": "; ".join(f"{x['date'].isoformat()}:{x['g']:g}" for x in HARVEST.get(sid, [])),
                "harvest_fresh_g": fresh_g(sid),
                "dry_substrate_g": DRY_G.get(sid),
                "be_measured_pct": None if pd.isna(container_be(sid)) else round(container_be(sid), 2),
                "lost": sid in LOSSES,
                "loss_stage": LOSSES.get(sid, {}).get("stage", ""),
                "loss_cause": LOSSES.get(sid, {}).get("cause", ""),
                "loss_date": LOSSES[sid]["date"].isoformat() if sid in LOSSES else "",
                "loss_rate": loss_rate_text(lost_n, len(batch_containers)),
                "currency": cost["currency"],
                **{f"cost_{k}": cost.get(k) for k, _ in COST_FIELDS},
                "cost_total": fig_b["cost"]["total"],
                "cost_per_kg": None if pd.isna(fig_b["cost"]["per_kg"]) else round(fig_b["cost"]["per_kg"], 2),
                "price_per_kg": cost.get("price"),
                "margin_per_kg": None if pd.isna(fig_b["cost"]["margin_per_kg"])
                else round(fig_b["cost"]["margin_per_kg"], 2),
            })
        st.download_button("Download my log (CSV)", pd.DataFrame(csv_rows).to_csv(index=False),
                           file_name=f"{batch_label or 'grow'}_{sid}_log.csv", mime="text/csv")
        st.caption("Nothing is saved; download the CSV to keep it.")

        # labels: one per container (syldd: all containers; d005: the picked dish)
        labels = []
        for cid in label_cids:
            d90 = day90_by[cid]
            if np.isnan(d90):
                due, note = inoc_date + timedelta(days=14), "typical, usemycro.com guidance"
            else:
                due, note = inoc_date + timedelta(days=round(d90)), "Colonisation forecast, 90% day"
            labels.append({"batch": batch_label or "(no batch label)", "container": str(cid),
                           "species": species, "inoculation": inoc_date.isoformat(),
                           "expected": due.isoformat(), "expected_note": note})
        st.download_button(f"Print labels ({len(labels)}, 8 per A4 page)", build_labels_html(labels),
                           file_name=f"{batch_label or 'grow'}_labels.html", mime="text/html")

        # backup: the whole session log as JSON
        photo_rows = [{"series_id": r.series_id, "day": int(r.day), "coverage": float(r.coverage),
                       "file": str(r.file)} for r in cov.itertuples()]
        for rows_ in list(UPLOAD_ROWS.values()) + list(RESTORED_PHOTOS.values()):
            photo_rows += [{**r, "day": int(r["day"]), "coverage": float(r["coverage"])} for r in rows_]
        photo_rows = list({(r["series_id"], r["file"], r["day"]): r for r in photo_rows}.values())
        backup = {"dataset": ds, "dish": sid, "batch": BATCH_FIELDS[ds], "dish_fields": DISH_FIELDS,
                  "stage_log": STAGE_LOG, "photos": photo_rows, "harvest": HARVEST, "dry_g": DRY_G,
                  "losses": LOSSES, "costs": COSTS}
        st.download_button("Download backup (JSON)", backup_to_json(backup),
                           file_name=f"{batch_label or 'grow'}_backup.json", mime="application/json")
        st.caption("Restore a backup (or the CSV) from the sidebar.")

n = st.slider("Photos used so far", 1, len(t), len(t),
              help="Move left to see the forecast as it looked with fewer photos.")
tn, cn = t[:n], c[:n]
res = forecast(tn, cn)

# --- warnings -------------------------------------------------------------------------
warns = forecast_warnings(tn, cn, res)
for w in warns:
    st.warning(w)

# --- plain-words summary ----------------------------------------------------------------
if len(tn) < MIN_PHOTOS:
    summary = "Not enough photos yet for a Colonisation forecast."
else:
    d90 = res["day90"]
    est = "not reached within 60 days" if np.isnan(d90) else f"around day {d90:.0f}"
    summary = (f"Latest photo: day {tn[-1]:.0f}, {100 * cn[-1]:.0f}% colonised (Photo coverage). "
               f"Colonisation forecast: 90% {est} (estimate).")
st.info(summary)
pace = pace_vs_typical(res["day90"]) if len(tn) >= MIN_PHOTOS else "unknown"
pace_text = {"ahead": "ahead of", "within": "within", "behind": "behind",
             "unknown": "not yet comparable with"}[pace]
st.markdown(f"Pace: **{pace_text}** the typical range. Typical oyster colonisation is 10-14 days "
            "(usemycro.com guidance, not our data).")

# --- chart ----------------------------------------------------------------------------
f = MODELS["logistic"][0]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t[n:], 100 * c[n:], "o", color="lightgrey", label="later photos (not used)")
ax.plot(tn, 100 * cn, "ko", label="photos used")
if res["params"] is not None:
    tt = np.linspace(0, max(t.max(), np.nan_to_num(res["day95"], nan=0)) + 3, 300)
    ax.plot(tt, 100 * f(tt, *res["params"]), color="tab:green", label="S-curve (logistic)")
    for p, key, style in [(90, "day90", "--"), (95, "day95", ":")]:
        if not np.isnan(res[key]):
            ax.axvline(res[key], color="tab:red", ls=style, label=f"{p}% on day {res[key]:.1f}")
ax.set_xlabel("day")
ax.set_ylabel("% colonised")
ax.set_ylim(0, 105)
ax.legend(loc="upper left", fontsize=8)
st.pyplot(fig)

c1, c2, c3 = st.columns(3)
c1.metric("Photos used", n)
c2.metric("90% reached on day", "n/a" if np.isnan(res["day90"]) else f"{res['day90']:.1f}")
c3.metric("95% reached on day", "n/a" if np.isnan(res["day95"]) else f"{res['day95']:.1f}")

# --- comparison across batches ------------------------------------------------------------
if is_syldd:
    st.subheader("Batch comparison (syldd)")
    rows = []
    for code_, _ in SYLDD_CODES.items():
        key = f"B-{code_}"
        members = sorted(cov[cov["series_id"].map(batch_of) == key]["series_id"].unique())
        days = [day90_by[cid] for cid in members]
        fb = batch_figures(key, members)
        rows.append({"Batch": key, "Containers": fb["containers"], "Lost": fb["lost"],
                     "Harvested (g)": fb["harvest_g"], "Mean BE (%, measured)": na(fb["mean_be"]),
                     "Cost per kg": na(fb["cost"]["per_kg"], "{:.2f} " + fb["currency"]),
                     "Forecast 90% colonisation day": na(np.nanmean(days) if not np.all(np.isnan(days)) else np.nan)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Differences between batches are observations, not proof of cause.")

# --- forecast as each photo is added ----------------------------------------------------
with st.expander("Details"):
    st.subheader("How the Colonisation forecast changed as photos were added")
    hist = forecast_by_photo(t, c)
    if hist.empty:
        st.info("Needs at least 2 photos.")
    else:
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(hist["n_photos"], hist["day90"], "o-", label="90% day")
        ax2.plot(hist["n_photos"], hist["day95"], "s--", label="95% day")
        ax2.axvline(n, color="grey", lw=0.8)
        ax2.set_xlabel("photos used")
        ax2.set_ylabel("forecast day")
        ax2.legend(fontsize=8)
        st.pyplot(fig2)
        show = hist.rename(columns={"n_photos": "Photos", "last_day": "Last photo (day)",
                                    "day90": "90% day", "day95": "95% day"})
        st.dataframe(show.style.format({"Last photo (day)": "{:.0f}", "90% day": "{:.1f}",
                                        "95% day": "{:.1f}"}, na_rep="n/a"), hide_index=True)

with st.expander("Notes"):
    st.markdown(
        "- d005 dots come from `data/interim/coverage_auto.csv` (whole-dish scans); "
        "syldd dots from `data/interim/coverage_syldd.csv` (phone photos, day counted from "
        "inoculation on 2026-10-08 using the photo time).\n"
        "- A weak forecast is flagged when there are fewer than 3 photos or 90% is not reached within 60 days.\n"
        "- Uploaded photos are not saved."
    )
