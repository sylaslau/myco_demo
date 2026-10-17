"""Environment: where and in what you grow, and the conditions (temperature, humidity)."""

import streamlit as st

from mycodemo import pipeline
from mycodemo.app import flow
from mycodemo.cultivation import baseline_guide

st.set_page_config(page_title="Where does it/ do they live?",
                   page_icon="📦", layout="wide")
flow.init()
ss = st.session_state
flow.need("choice", "What do you want to grow?", "pages/3_What_to_grow.py")
for key, value in {"env_where": None, "env_size": {}, "env_container": None, "rh_pct": 80.0}.items():
    ss.setdefault(key, value)

# Where -> (label the pipeline plans the recipe scale from, slider label, min, max, default, step)
WHERE = {
    "Plastic container": ("Small plastic container (about 1 L)", "Size (litres)", 0.5, 5.0, 1.0, 0.5),
    "Shelf or vitrine": ("Shelf or grow tent", "Size (number of shelves)", 1, 8, 3, 1),
    "Tent": ("Shelf or grow tent", "Size (floor area, m²)", 0.25, 4.0, 1.0, 0.25),
    "Room": ("Room", "Size (floor area, m²)", 2, 50, 12, 1),
}
CONTAINERS = {
    "Tupperware": ("Size (litres)", 0.5, 5.0, 1.0, 0.5),
    "Grow bag": ("Size (litres)", 0.5, 10.0, 1.0, 0.5),
    "Bucket": ("Size (litres)", 0.5, 10.0, 1.0, 0.5),
}


def sync_room(src, dst):
    """Keep the room slider and its number box showing the same size."""
    ss[dst] = ss[src]


st.title("Environment")
st.write("Where do you grow them, and in what?")
where_col, cont_col = st.columns(2)

with where_col:
    st.subheader("Residence")
    for name, (label, size_label, lo, hi, default, step) in WHERE.items():
        on = ss["env_where"] == name
        if st.button(name, key=f"where_{name}", type="primary" if on else "secondary", use_container_width=True):
            ss["env_where"], ss["environment"] = name, label
            ss["recipe"] = ss["batch"] = None
            st.rerun()
        size = ss["env_size"].get(name, default)
        if name == "Room":
            ss.setdefault("room_sl", size)
            ss.setdefault("room_num", size)
            c1, c2 = st.columns([3, 1])
            c1.slider(size_label, lo, hi, key="room_sl", disabled=not on,
                      on_change=sync_room, args=("room_sl", "room_num"))
            c2.number_input("Exact (m²)", 1, 500, key="room_num", disabled=not on, on_change=sync_room,
                            args=("room_num", "room_sl"))
            size = ss["room_num"]
        else:
            size = st.slider(size_label, lo, hi, size, step,
                             key=f"size_{name}", disabled=not on)
        if on:
            ss["env_size"][name] = size

with cont_col:
    st.subheader("Vessels")
    for name in CONTAINERS:
        on = ss["env_container"] == name
        if st.button(name, key=f"cont_{name}", type="primary" if on else "secondary", use_container_width=True):
            ss["env_container"] = name
            st.rerun()

if ss["environment"]:
    n, kg = flow.ENVIRONMENTS[ss["environment"]]
    st.caption(
        f"We plan {n} container(s) of {kg} kg dry substrate each (our default; the recipe page shows the numbers).")

st.divider()
st.header("Conditions")
st.markdown("**Input method**")
st.markdown(":green[●] Slider (active)")

t_col, h_col = st.columns(2)
temp = t_col.slider("Temperature (°C)", 10.0, 40.0,
                    ss["temp_c"] if ss["temp_c"] is not None else 21.0, 0.5)
ss["temp_c"] = float(temp)
ss["rh_pct"] = float(h_col.slider(
    "Humidity (% RH)", 20, 100, int(ss["rh_pct"]), 1))

flow.preview("Use weather data from your phone",
             "Take the ambient values from your phone.")
flow.preview("Connect an external device",
             "Read temperature and humidity from a sensor (for example a SwitchBot).")


row = flow.species_row()
br = pipeline.baseline_row(flow.get_batch())
if row is None or br is None:
    st.info(
        "No temperature range is known for this mushroom, so we cannot check it.")
else:
    inc = baseline_guide.colonisation(br[0])["incubation_temp_C"]
    lo, hi = inc["min"], inc["max"]
    if lo <= temp <= hi:
        st.success(
            f"{temp:g} °C is inside the colonisation range of {row['name']} ({lo}–{hi} °C, consensus guide).")
    else:
        st.warning(f"{temp:g} °C is outside the colonisation range of {row['name']} ({lo}–{hi} °C, consensus guide). "
                   "Colonisation will be slower, or may stall; consider a warmer or cooler spot.")
    st.caption(f"Row used: {br[1]}.")
    if ss["goal"] == "Food":
        ft = baseline_guide.fruiting(br[0])["fruiting_temp_C"]
        st.caption(f"Fruiting later wants {ft['min']}–{ft['max']} °C (consensus guide).")

if ss["environment"]:
    st.page_link("pages/5_Recipe_and_first_prediction.py",
                 label="Next: recipe and preview", icon="➡️")
else:
    st.caption("Choose where you grow to continue.")
