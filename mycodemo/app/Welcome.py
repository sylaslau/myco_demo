"""MycoDemo app: welcome page (the entry file).

Run from the project folder (with the .venv active):
    streamlit run mycodemo/app/Welcome.py
"""

import streamlit as st

from mycodemo.app import flow

st.set_page_config(page_title="MycoDemo", page_icon="🍄", layout="wide")
flow.init()

st.title("MycoDemo: grow planner")
st.write(
    "Plan a grow in a few steps, then follow it with photos. MycoDemo is built on the open-source "
    "mycro-grow maths (MIT licence, the tools of [usemycro.com](https://usemycro.com)) plus our own "
    "data: your colonisation photos update the forecast as the grow goes on."
)
st.subheader("Your steps")
st.page_link("pages/1_Start.py", label="1. Start: food or materials", icon="🌱")
st.page_link("pages/2_What_to_grow.py", label="2. What to grow", icon="🍄")
st.page_link("pages/3_Your_environment.py", label="3. Your environment", icon="📦")
st.page_link("pages/4_Ambient_temperature.py", label="4. Ambient temperature", icon="🌡️")
st.page_link("pages/5_Recipe_and_first_prediction.py", label="5. Recipe and first prediction", icon="🧪")
st.page_link("pages/6_Grow_Log_and_Photos.py", label="6. Grow Log and photos", icon="📷")
st.page_link("pages/7_Calendar_and_tasks.py", label="7. Calendar and tasks (preview)", icon="📅")
st.caption("Greyed boxes marked PREVIEW show features that are planned but not active yet.")
st.info(
    "Yield is an estimate from an assumed biological efficiency (BE), "
    "not a prediction: no dataset yet links growth to harvest weight."
)
