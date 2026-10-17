"""Growing guide: consensus values for the 8 ref_2 species on one substrate (lookup only, no model)."""

import pandas as pd
import streamlit as st

from mycodemo.app import flow
from mycodemo.cultivation import baseline_guide as bg

st.set_page_config(page_title="Growing guide", page_icon="📖", layout="wide")
flow.init()
ss = st.session_state

SUBSTRATE_LABELS = {"straw": "Straw", "cotton": "Cotton waste", "sawdust": "Sawdust", "bagasse": "Bagasse",
                    "logs": "Logs", "masters_mix": "Masters mix", "corncob": "Corncob"}
CONTAINERS = ["Tupperware", "Grow bag", "Bucket", "Other"]
SUBSTRATES = list(SUBSTRATE_LABELS)

st.title("Growing guide")
st.write("Pick a substrate and your room temperature to see which mushrooms grow there.")

c1, c2, c3 = st.columns(3)
substrate = c1.selectbox("Substrate", SUBSTRATES, format_func=SUBSTRATE_LABELS.get,
                         index=SUBSTRATES.index(ss.get("g_substrate", "sawdust")))
temp = c2.number_input("Ambient temperature (°C)", -5.0, 45.0, float(ss.get("g_temp", 22.0)), 0.5)
container = c3.selectbox("Container (shown only)", CONTAINERS,
                         index=CONTAINERS.index(ss.get("g_container", CONTAINERS[0])))
ss["g_substrate"], ss["g_temp"], ss["g_container"] = substrate, temp, container

df = bg.load()
on_substrate = df[df["substrate_code"] == substrate]
hits = bg.matches(substrate, temp)

st.subheader(f"Mushrooms that colonise {SUBSTRATE_LABELS[substrate].lower()} at {temp:g} °C")
if hits.empty:
    st.warning("No species in the guide colonises this substrate at this temperature. "
               "Colonisation ranges per species are listed below.")
else:
    st.dataframe(pd.DataFrame({
        "Common name": hits["common_name"], "Latin name": hits["species"],
        "Colonisation range (°C)": hits["incubation_temp_min_C"].astype(str) + "–" + hits["incubation_temp_max_C"].astype(str),
        "Optimum (°C)": hits["incubation_temp_opt_C"],
        "Colonisation (days)": hits["colonisation_days_min"].astype(str) + "–" + hits["colonisation_days_max"].astype(str),
    }), hide_index=True, use_container_width=True)

# optional species: any species with a row on this substrate; default = the first match
options = on_substrate["species"].tolist()
if not options:
    st.stop()
names = on_substrate.set_index("species")["common_name"].to_dict()
default = hits["species"].iloc[0] if not hits.empty else options[0]
saved = ss.get("g_species")
start = saved if saved in options else default
species = st.selectbox("Species", options, index=options.index(start),
                       format_func=lambda s: names[s])
ss["g_species"] = species
st.markdown(f"### {names[species]}  \n<small><i>{species}</i></small>", unsafe_allow_html=True)

r, row_note = bg.row(species, substrate)
text = bg.load_text()
t = text[text["ref_id"] == r["ref_id"]].iloc[0]
col, fr = bg.colonisation(r), bg.fruiting(r)
inc = col["incubation_temp_C"]
state = bg.temp_check(temp, inc["min"], inc["max"])

a, b = st.columns(2)
with a:
    st.markdown("**Spawn and moisture**")
    spawn = float(r["spawn_rate_pct"])
    st.write(f"Spawn rate, source basis (% of wet substrate): **{spawn:g}%**  \n"
             f"Spawn rate, our basis (% of wet substrate + spawn): **{bg.spawn_on_our_basis(spawn):.1f}%**")
    st.write(f"Substrate moisture: **{int(r['substrate_moisture_pct'])}%**")
    st.markdown("**Colonisation**")
    st.write(f"Temperature range: **{inc['min']}–{inc['max']} °C**, optimum **{inc['opt']} °C**")
    msg = {"inside": f"{temp:g} °C is inside the range.",
           "below": f"{temp:g} °C is below the range: expect slow or stalled growth.",
           "above": f"{temp:g} °C is above the range: risk of overheating or stalled growth."}[state]
    (st.success if state == "inside" else st.warning)(msg)
    st.write(f"Incubation humidity: **{col['incubation_rh_pct']}% RH**")
with b:
    st.markdown("**Fruiting**")
    ft, rh = fr["fruiting_temp_C"], fr["fruiting_rh_pct"]
    st.write(f"Temperature: **{ft['min']}–{ft['max']} °C** (optimum {ft['opt']} °C)  \n"
             f"Humidity: **{rh['min']}–{rh['max']}% RH**  \n"
             f"CO2: below **{fr['co2_max_ppm']} ppm**")
    st.markdown("**Preparation**")
    st.write(t["preparation_actions"])
    st.markdown("**Supplementation**")
    st.write(t["substrate_supplementation"])

st.caption("Consensus values from books and extension guides, not a prediction. "
           f"Source: {col['source']}; confidence {col['confidence']}. Row used: {row_note}.")
st.page_link("pages/2_Stage_timeline.py", label="Next: stage timeline", icon="➡️")
