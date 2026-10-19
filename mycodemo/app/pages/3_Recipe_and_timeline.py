"""Recipe and first prediction: shows what the pipeline returns (recipe, colonisation timeline, yield or strength)."""

import pandas as pd
import streamlit as st

from mycodemo import pipeline
from mycodemo.app import flow

st.set_page_config(page_title="Recipe and first prediction", page_icon="🧪", layout="wide")
flow.init()
ss = st.session_state
flow.need("environment", "Environment", "pages/2_Environment.py")

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

# --- stage 1: colonisation -----------------------------------------------------------------
est, fc, dates, fruit = result.colonisation_estimate, result.colonisation_forecast, result.dates, result.fruiting
st.subheader("Stage 1: Colonisation")
if est.is_preview:
    flow.preview("Colonisation time estimate", est.note)
else:
    lo, hi = est.value["days"]
    full = next((a, b) for n, a, b in dates.value if n == "Fully colonised")
    c1, c2 = st.columns(2)
    c1.metric("Fully colonised", f"{full[0].strftime('%d %b')} – {full[1].strftime('%d %b')}")
    c2.metric("Duration", f"{lo:.0f}–{hi:.0f} days", help=f"Typical {est.value['typical']:.0f} days")
    st.progress(0.0, text=f"Day 0 → day {hi:.0f}")
    st.caption(flow.FULLY_COLONISED_NOTE)
    flow.show_source(est)
    if fc.value is not None and fc.value["method"] != "estimate":
        st.caption(f"Your photos have changed this: {fc.note} The photo forecast's 90% day is a surface marker, "
                   "not the end of stage 1.")
    else:
        st.caption("Your photos in Monitor will update it into the Colonisation forecast.")

# --- stage 2: fruiting ---------------------------------------------------------------------
if fruit is not None:
    st.subheader("Stage 2: Fruiting")
    if fruit.is_preview:
        flow.preview("Fruiting", fruit.note)
    else:
        fv = fruit.value
        ev = {n: (a, b) for n, a, b in dates.value}

        def span(name):
            a, b = ev[name]
            return a.strftime("%d %b") + ("" if a == b else f" – {b.strftime('%d %b')}")

        f1, f2, f3 = st.columns(3)
        f1.metric("Start induction", span("Start induction"))
        f2.metric("Pinning", span("Pinning"), help=f"{fv['days_to_pinning']} days after the start of induction")
        f3.metric("First harvest", span("First harvest"), help=f"{fv['first_harvest']} days after the start of induction")
        ia = fv["induction_actions"]
        steps = [f"**{k}:** {ia[key]}" for k, key in (("Temperature", "temperature"), ("Light", "light"),
                                                   ("Fresh air", "fresh_air"), ("Humidity", "humidity"),
                                                   ("Other", "other")) if isinstance(ia[key], str) and ia[key].strip()]
        st.markdown("**Induction actions**")
        if isinstance(ia["overview"], str) and ia["overview"].strip():
            st.write(ia["overview"])
        st.markdown("\n".join(f"- {x}" for x in steps))
        if fv["inferred"]:
            st.markdown(":orange-badge[Inferred] These induction actions are inferred from other columns of the "
                        "guide, not stated in it.")
        t, rh = fv["fruiting_temp_C"], fv["fruiting_rh_pct"]
        g1, g2, g3 = st.columns(3)
        g1.metric("Fruiting temperature", f"{t['min']}–{t['max']} °C", help=f"Optimum {t['opt']} °C")
        g2.metric("Fruiting humidity", f"{rh['min']}–{rh['max']} % RH")
        g3.metric("CO2", f"below {fv['co2_max_ppm']} ppm")
        flow.show_source(fruit)
        st.caption(dates.note)

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
    st.switch_page("pages/4_Monitor.py")
