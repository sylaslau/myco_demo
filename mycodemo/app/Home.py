"""MycoDemo app: start page.

Run from the project folder (with the .venv active):
    streamlit run mycodemo/app/Home.py
"""

import streamlit as st

st.set_page_config(page_title="MycoDemo", page_icon="🍄", layout="wide")

st.title("MycoDemo: oyster mushroom grow planner")
st.write(
    "A Python version of the [usemycro.com](https://usemycro.com) grow tools, "
    "built on the open-source mycro-grow maths (MIT licence). "
    "Each page shows three versions side by side, so you can see what changes and why:"
)
st.markdown(
    """
| Column | What it is |
|---|---|
| **Original** | mycro-grow's maths, unchanged |
| **Corrected** | starting moisture, yield from true dry matter, spawn shown on both bases |
| **MycoDemo** | the corrected maths plus warnings; later a colonisation forecast that updates from your photos |
"""
)

st.subheader("Pages")
st.page_link("pages/1_Grow_Calculator.py", label="Grow Calculator", icon="🧮")
st.caption(
    "Coming next: Grow Log & Photos (colonisation forecast), Substrate Recipe, "
    "Fruiting Conditions, Harvest & BE, How good are the models, Materials (extension)."
)

st.info(
    "Yield is an estimate from an assumed biological efficiency (BE), "
    "not a prediction: no dataset yet links growth to harvest weight."
)
