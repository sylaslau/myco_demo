"""Environment: where and in what you grow, and the conditions (temperature, humidity)."""

import streamlit as st

from mycodemo.app import flow

st.set_page_config(page_title="Environment", page_icon="📦", layout="wide")
flow.init()
ss = st.session_state
flow.need("choice", "What to grow", "pages/1_What_to_grow.py")
for key, value in {"env_where": None, "env_size": {}, "env_container": None, "rh_pct": 80.0}.items():
    ss.setdefault(key, value)

# Where -> (label the pipeline plans the recipe scale from, slider label, min, max, default, step)
WHERE = {
    "Plastic container": ("Small plastic container (about 1 L)", "Size (litres)", 0.5, 5.0, 1.0, 0.5),
    "Shelf or vitrine": ("Shelf or grow tent", "Size (number of shelves)", 1, 8, 3, 1),
    "Tent": ("Shelf or grow tent", "Size (floor area, m²)", 0.25, 4.0, 1.0, 0.25),
    "Room": ("Room", "Size (floor area, m²)", 2, 50, 12, 1),
}
CONTAINERS = ["Tupperware", "Grow bag", "Bucket"]


def sync_room(src, dst):
    """Keep the room slider and its number box showing the same size."""
    ss[dst] = ss[src]


st.title("Environment")
st.write("Where will you grow, and in what?")
where_col, cont_col = st.columns(2)

with where_col:
    st.subheader("Where")
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
            c1.slider(size_label, lo, hi, key="room_sl", disabled=not on, on_change=sync_room, args=("room_sl", "room_num"))
            c2.number_input("Exact (m²)", 1, 500, key="room_num", disabled=not on, on_change=sync_room,
                            args=("room_num", "room_sl"))
            size = ss["room_num"]
        else:
            size = st.slider(size_label, lo, hi, size, step, key=f"size_{name}", disabled=not on)
        if on:
            ss["env_size"][name] = size

with cont_col:
    st.subheader("Containers")
    for name in CONTAINERS:
        on = ss["env_container"] == name
        if st.button(name, key=f"cont_{name}", type="primary" if on else "secondary", use_container_width=True):
            ss["env_container"] = name
            st.rerun()

if ss["environment"]:
    n, kg = flow.ENVIRONMENTS[ss["environment"]]
    st.caption(f"We plan {n} container(s) of {kg} kg dry substrate each (our default; the recipe page shows the numbers).")

st.divider()
st.header("Conditions")
st.markdown("**Input method**")
st.markdown(":green[●] Slider (active)")
flow.preview("Connect an external device", "Read temperature and humidity from a sensor (for example a SwitchBot).")
flow.preview("Use weather data from your phone", "Take the ambient values from your phone.")

t_col, h_col = st.columns(2)
temp = t_col.slider("Temperature (°C)", -10.0, 50.0, ss["temp_c"] if ss["temp_c"] is not None else 21.0, 0.5)
ss["temp_c"] = float(temp)
ss["rh_pct"] = float(h_col.slider("Humidity (% RH)", 0, 100, int(ss["rh_pct"]), 1))

row = flow.species_row()
if row is None:
    st.info("No temperature range is known for your own mushroom, so we cannot check it.")
else:
    lo, hi = row["colonizeC"]
    if lo <= temp <= hi:
        st.success(f"{temp:g} °C is inside the colonisation range of {row['name']} ({lo}–{hi} °C).")
    else:
        st.warning(f"{temp:g} °C is outside the colonisation range of {row['name']} ({lo}–{hi} °C). "
                   "Colonisation will be slower, or may stall; consider a warmer or cooler spot.")
    if ss["goal"] == "Food":
        st.caption(f"Fruiting later wants {row['fruitC'][0]}–{row['fruitC'][1]} °C.")

if ss["environment"]:
    st.page_link("pages/5_Recipe_and_first_prediction.py", label="Next: recipe and first prediction", icon="➡️")
else:
    st.caption("Choose where you grow to continue.")
