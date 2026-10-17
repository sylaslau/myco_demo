"""Stage timeline: when each stage starts, from the consensus guide (lookup only; temperature does not change the days)."""

import plotly.graph_objects as go
import streamlit as st

from mycodemo.app import flow
from mycodemo.cultivation import baseline_guide as bg

st.set_page_config(page_title="Stage timeline", page_icon="🗓️", layout="wide")
flow.init()
ss = st.session_state

SUBSTRATE_LABELS = {"straw": "Straw", "cotton": "Cotton waste", "sawdust": "Sawdust", "bagasse": "Bagasse",
                    "logs": "Logs", "masters_mix": "Masters mix", "corncob": "Corncob"}
CONTAINERS = ["Tupperware", "Grow bag", "Bucket", "Other"]
SUBSTRATES = list(SUBSTRATE_LABELS)
COLONISED_NOTE = flow.FULLY_COLONISED_NOTE

st.title("Stage timeline")
st.write("Days since inoculation, from the consensus guide. Your temperature is checked but does not change the days here.")

c1, c2, c3 = st.columns(3)
substrate = c1.selectbox("Substrate", SUBSTRATES, format_func=SUBSTRATE_LABELS.get,
                         index=SUBSTRATES.index(ss.get("g_substrate", "sawdust")))
temp = c2.number_input("Ambient temperature (°C)", -5.0, 45.0, float(ss.get("g_temp", 22.0)), 0.5)
container = c3.selectbox("Container (shown only)", CONTAINERS,
                         index=CONTAINERS.index(ss.get("g_container", CONTAINERS[0])))
ss["g_substrate"], ss["g_temp"], ss["g_container"] = substrate, temp, container

df = bg.load()
on_sub = df[df["substrate_code"] == substrate]
if on_sub.empty:
    st.stop()
options = on_sub["species"].tolist()
names = on_sub.set_index("species")["common_name"].to_dict()
hits = bg.matches(substrate, temp)
saved = ss.get("g_species")
start = saved if saved in options else (hits["species"].iloc[0] if not hits.empty else options[0])
species = st.selectbox("Species", options, index=options.index(start), format_func=lambda s: names[s])
ss["g_species"] = species
st.markdown(f"### {names[species]}  \n<small><i>{species}</i></small>", unsafe_allow_html=True)

r, row_note = bg.row(species, substrate)
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
