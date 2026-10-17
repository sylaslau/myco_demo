"""Shroomify app: welcome page (the entry file).

Run from the project folder (with the .venv active):
    streamlit run mycodemo/app/Welcome.py
"""

import streamlit as st

from mycodemo.app import flow

st.set_page_config(page_title="Shroomify", page_icon="🍄", layout="wide")
flow.init()

FLOW_CSS = """
<style>
.sf-box {border: 2px solid #2e8b57; border-radius: 10px; padding: 0.55rem 1rem; text-align: center;
  font-weight: 600; background: rgba(46,139,87,0.15);}
.sf-box.off {border-color: #9aa0a6; background: rgba(150,150,150,0.12); color: #8a8f94;}
.sf-arrow {text-align: center; font-size: 1.3rem; line-height: 1.6rem; color: #8a8f94;}
</style>
"""


def stage_flow(title, stages):
    """One vertical flow: boxes top to bottom with a down arrow between; (name, active) pairs."""
    parts = [f"<h3>{title}</h3>"]
    for i, (name, active) in enumerate(stages):
        if i:
            parts.append('<div class="sf-arrow">↓</div>')
        parts.append(f'<div class="sf-box{"" if active else " off"}">{name}</div>')
    return "".join(parts)


st.title("Shroomify")
st.markdown(
    "<div style='font-size:1.05rem'>Hi! I'm Shroomify, your mycelium cultivation buddy. I sit by your side to:"
    "<ul>"
    "<li>🍄 help you pick what to grow</li>"
    "<li>🧪 hand you a starter recipe</li>"
    "<li>📷 keep an eye on your photos</li>"
    "<li>⏰ tell you when the next stage is coming</li>"
    "</ul></div>",
    unsafe_allow_html=True,
)

st.markdown(FLOW_CSS, unsafe_allow_html=True)
food, thing = st.columns(2)
food.markdown(stage_flow("Food", [("Culture", False), ("Spawn", False), ("Block", True),
                                  ("Flush", True), ("Flush", True)]), unsafe_allow_html=True)
thing.markdown(stage_flow("Thing", [("Culture", False), ("Spawn", False), ("Pretreatment", False),
                                    ("Block", True), ("Posttreatment", False)]), unsafe_allow_html=True)

st.write("")
st.caption("Built on the open-source mycro-grow maths (MIT licence, the tools of [usemycro.com](https://usemycro.com)) "
           "plus our own data. Greyed boxes marked PREVIEW show features that are planned but not active yet.")
st.info(
    "Yield is an estimate from an assumed biological efficiency (BE), "
    "not a prediction: no dataset yet links growth to harvest weight."
)
