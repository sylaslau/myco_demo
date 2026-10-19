"""Recipe and timeline: shows what the pipeline returns (recipe, colonisation timeline, yield or strength)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mycodemo import pipeline
from mycodemo.app import flow
from mycodemo.cultivation import baseline_guide as bg

st.set_page_config(page_title="Recipe and timeline", page_icon="🧪", layout="wide")
flow.init()
ss = st.session_state
flow.need("environment", "Environment", "pages/2_Environment.py")

st.title("Recipe and timeline")
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

# --- stage timeline (stage 1 block below is the fallback) -------------------------------------
est, fc, dates, fruit = result.colonisation_estimate, result.colonisation_forecast, result.dates, result.fruiting
base = pipeline.baseline_row(batch)
if base is not None:
    r, row_note = base
    temp = ss["temp_c"]
    species = r["species"]
    COLONISED_NOTE = flow.FULLY_COLONISED_NOTE
    st.subheader("Stage timeline")
    t = bg.load_text()
    t = t[t["ref_id"] == r["ref_id"]].iloc[0]
    col, fr = bg.colonisation(r), bg.fruiting(r)
    inc = col["incubation_temp_C"]
    state = bg.temp_check(temp, inc["min"], inc["max"])
    (st.success if state == "inside" else st.warning)(
        f"{temp:g} °C is {state} the colonisation range ({inc['min']}–{inc['max']} °C)."
        + ("" if state == "inside" else " The days below assume the guide's own conditions."))

    tl = bg.stages(r).set_index("stage")

    # --- horizontal timeline ---------------------------------------------------------------
    fig = go.Figure()
    full = tl.loc["Fully colonised"]
    fig.add_trace(go.Bar(y=["Timeline"], x=[full["end_day"] - full["start_day"]], base=[full["start_day"]],
                         orientation="h", name="Fully colonised (min–max)", marker_color="#2e8b57", opacity=0.45,
                         width=0.35, hovertemplate=f"Fully colonised: day {full['start_day']}–{full['end_day']}<extra></extra>"))
    marks = [("Fully colonised", "typical", full["typical_day"], "#2e8b57"),
             ("Start induction", "", full["typical_day"], "#1f77b4"),
             ("Pinning", "", tl.loc["Pinning", "typical_day"], "#ff7f0e"),
             ("First harvest", "", tl.loc["First harvest", "typical_day"], "#d62728")]
    for label, _, day, color in marks:
        if label == "Start induction":
            continue  # same day as Fully colonised; shown in the cards
        fig.add_trace(go.Scatter(x=[day], y=["Timeline"], mode="markers+text", name=f"{label} (typical)",
                                 marker=dict(symbol="line-ns", size=36, line=dict(width=3, color=color)),
                                 text=[f"{label}<br>day {int(day)}"], textposition="top center"))
    fig.add_vline(x=0, line_dash="dot", annotation_text="Inoculation (day 0)")
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=40), xaxis_title="Days since inoculation",
                      showlegend=False, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green band: Fully colonised, minimum to maximum days; ticks: typical day. Start induction falls on the "
               "Fully colonised day. Pinning and First harvest are the typical days counted from the start of induction.")


    def days(label):
        s = tl.loc[label]
        return f"day {s['start_day']}–{s['end_day']} (typical {s['typical_day']})"


    # --- cards -----------------------------------------------------------------------------
    inferred = fr["induction_evidence_status"] == "inferred_from_existing_columns"
    ia = fr["induction_actions"]
    cold = {"yes": "Yes", "no": "No"}.get(str(r["induction_cold_shock_required"]).lower(), "Unknown")

    with st.container(border=True):
        st.markdown(f"**Inoculation** · day 0")
        st.write(t["preparation_actions"])
        st.caption(t["colonisation_cautions"])
    with st.container(border=True):
        st.markdown(f"**Fully colonised** · {days('Fully colonised')}")
        st.write(COLONISED_NOTE)
        st.caption(t["colonisation_cautions"])
    with st.container(border=True):
        head = f"**Start induction** · {days('Start induction')}"
        st.markdown(head + ("  &nbsp; :orange-badge[inferred]" if inferred else ""))
        tgt = r["induction_temperature_target_C"]
        st.write(f"Temperature: **{t['induction_temperature_action']}**"
                 + (f", target **{tgt:g} °C**" if str(tgt) not in ("", "nan") else "") + f" · Cold shock: **{cold}**")
        st.write(f"Light: {ia['light']}  \nFresh air: {ia['fresh_air']}  \nHumidity: {ia['humidity']}  \nOther: {ia['other']}")
        if inferred:
            st.caption("Inferred from other columns of the guide, not stated in it.")
        if species == "Lentinula edodes":
            st.info(f"Unresolved: {t['requirements_unresolved']}")
    with st.container(border=True):
        st.markdown(f"**Pinning** · {days('Pinning')}")
        st.write(f"Keep {fr['fruiting_temp_C']['min']}–{fr['fruiting_temp_C']['max']} °C, "
                 f"{fr['fruiting_rh_pct']['min']}–{fr['fruiting_rh_pct']['max']}% RH, CO2 below {fr['co2_max_ppm']} ppm.")
        st.caption(t["fruiting_cautions"])
    with st.container(border=True):
        st.markdown(f"**First harvest** · {days('First harvest')}")
        st.write(f"About {fr['pinning_to_harvest']} days from pinning to harvest.")
        st.caption(t["fruiting_cautions"])

    st.caption(f"Source: {col['source']} · evidence: {r['evidence_type']} · confidence: {col['confidence']}. "
               f"Row used: {row_note}. Consensus values, not a prediction.")
else:
    st.subheader("Stage 1: Colonisation")
    if est.is_preview:
        flow.preview("Colonisation time estimate", est.note)
    else:
        lo, hi = est.value["days"]
        ev1 = {n: (a, b) for n, a, b in dates.value}
        full = ev1[pipeline.ENDPOINT_LABEL]

        def span1(a, b):
            return f"{a.strftime('%d %b')} – {b.strftime('%d %b')}"

        c1, c2, c3 = st.columns(3)
        c1.metric(pipeline.ENDPOINT_LABEL, span1(*full), help="Project assumption")
        mk = ev1.get(pipeline.MARKER_LABEL)
        if mk:
            c2.metric(pipeline.MARKER_LABEL, span1(*mk), help="Project assumption: the 90% day + 2 days")
        c3.metric("Duration to 90%", f"{lo:.0f}–{hi:.0f} days", help=f"Typical {est.value['typical']:.0f} days")
        st.progress(0.0, text=f"Day 0 → day {hi:.0f}")
        st.caption(flow.ENDPOINT_ASSUMPTION_NOTE)
        flow.show_source(est)
        if fc.value is not None and fc.value["method"] != "estimate":
            st.caption(f"Your photos have changed this: {fc.note}")
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
        f1.metric(pipeline.MARKER_LABEL, span(pipeline.MARKER_LABEL), help="Project assumption: the 90% day + 2 days")
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
