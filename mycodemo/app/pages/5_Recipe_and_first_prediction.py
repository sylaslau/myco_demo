"""Recipe and first prediction: shows what the pipeline returns (recipe, colonisation timeline, yield or strength)."""

import pandas as pd
import streamlit as st

from mycodemo import pipeline
from mycodemo.app import flow

st.set_page_config(page_title="Recipe and first prediction", page_icon="🧪", layout="wide")
flow.init()
ss = st.session_state
flow.need("environment", "Environment", "pages/3_Environment.py")

st.title("Recipe and first prediction")
ss["inoc_date"] = st.date_input("Planned inoculation date", ss["inoc_date"])
result = flow.run()
batch = flow.get_batch()
what = flow.choice_label()
st.caption(f"{what} · {ss['environment']} · {ss['temp_c']:g} °C")

# --- recipe ---------------------------------------------------------------------------------
st.subheader("Baseline recipe")
rv = result.recipe.value
rec, n, spawn_kg = rv["rec"], rv["n"], rv["spawn_kg"]
pb = rec["per_block"]
ing = pd.DataFrame([{"Ingredient": i["name"], "Per container (kg)": i["per_block"], "Total (kg)": i["total"]}
                    for i in rec["ingredients"]]
                   + [{"Ingredient": f"Water (to {rv['moisture']} % moisture)", "Per container (kg)": pb["water"],
                       "Total (kg)": pb["water"] * n},
                      {"Ingredient": f"Spawn ({rv['spawn_pct']} % of the whole mix)", "Per container (kg)": spawn_kg,
                       "Total (kg)": spawn_kg * n}])
st.dataframe(ing.style.format({"Per container (kg)": "{:.2f}", "Total (kg)": "{:.2f}"}), hide_index=True)
st.caption(f"{n} container(s). Amounts are dry matter for the substrate. {result.recipe.note}")

# --- colonisation timeline -----------------------------------------------------------------
st.subheader("Colonisation time estimate")
est, fc, dates = result.colonisation_estimate, result.colonisation_forecast, result.dates
if est.is_preview:
    flow.preview("Colonisation time estimate", est.note)
else:
    lo, hi = est.value["days"]
    c1, c2 = st.columns(2)
    c1.metric("Colonised (90%)", f"{dates.value[1][1].strftime('%d %b')} – {dates.value[1][2].strftime('%d %b')}")
    c2.metric("Duration", f"{lo:.0f}–{hi:.0f} days")
    st.progress(0.0, text=f"Day 0 → day {hi:.0f}")
    flow.show_source(est)
    if fc.value is not None and fc.value["method"] != "estimate":
        st.caption(f"Your photos have changed this: {fc.note}")
    else:
        st.caption("Your photos in Monitor will update it into the Colonisation forecast.")

if len(dates.value) > 2:
    st.subheader("Next dates")
    for name, a, b in dates.value[2:]:
        st.write(f"**{name}:** {a.strftime('%d %b')}" + ("" if a == b else f" – {b.strftime('%d %b')}"))

# --- warnings -----------------------------------------------------------------------------------
for w in result.warnings.value:
    st.warning(w)

# --- yield or strength ----------------------------------------------------------------------
out = result.outcome
st.subheader(out.name + (" estimate" if out.name == "Material strength" else ""))
if out.is_preview:
    flow.preview(out.name, out.note)
elif ss["goal"] == "Food":
    st.metric("Estimated fresh yield, whole batch", f"{out.value['kg']:.1f} kg")
    st.caption(out.note)
else:
    m1, m2 = st.columns(2)
    m1.metric("Stress at 10 % strain", f"{out.value['stress_mpa']:.3f} MPa")
    m2.metric("Density", f"{out.value['density']:.0f} kg/m³")
    st.caption(out.note)

if result.harvest_readiness is not None:
    flow.preview("Harvest readiness", result.harvest_readiness.note)

# --- hand over to Monitor ----------------------------------------------------------------
if st.button("Save this plan and open Monitor", type="primary"):
    ss["recipe"] = {"preset": pipeline.BASELINE, "moisture": pipeline.MOISTURE, "spawn_pct": pipeline.SPAWN_PCT,
                    "substrate": "sawdust"}
    row = flow.species_row()
    ss["batch"] = {"batch_label": f"{what} {ss['inoc_date'].isoformat()}", "species": row["latin"] if row else what,
                   "inoc_date": ss["inoc_date"], "n_containers": n, "temp_c": ss["temp_c"],
                   "substrate": "sawdust", "spawn_pct": float(pipeline.SPAWN_PCT)}
    st.switch_page("pages/6_Monitor.py")
