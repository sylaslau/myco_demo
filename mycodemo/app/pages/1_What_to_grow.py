"""What to grow: one pick in total, a Food (mushroom) or a Thing (mycomaterial)."""

import streamlit as st

from mycodemo.app import flow
from mycodemo.cultivation import FRUITING_CONDITIONS

st.set_page_config(page_title="What to grow", page_icon="🍄", layout="wide")
flow.init()
ss = st.session_state

# shown name -> FRUITING_CONDITIONS name (what the pipeline looks up); None = not in our table (kept as a custom pick)
FOODS = {"Oyster": "Blue Oyster", "King oyster": "King Oyster", "Shiitake": "Shiitake",
         "Lion's mane": "Lion's Mane", "Enoki": "Enoki", "Button mushroom": None}
LATIN_EXTRA = {"Button mushroom": "Agaricus bisporus"}      # not in FRUITING_CONDITIONS
THINGS = ["Packaging foam", "Acoustic panel", "Mycoleather", "Bricks or blocks", "Game pieces"]
FOOD_OWN, THING_OWN = "Your next culinary invention", "Your next thingiverse invention"
latin = {s["name"]: s["latin"] for s in FRUITING_CONDITIONS}


def pick(goal, choice, custom_name=""):
    """Save the one pick the way the old Start and What to grow pages did; a new pick resets the recipe."""
    ss["goal"], ss["choice"], ss["custom_name"] = goal, choice, custom_name
    ss["recipe"] = ss["batch"] = None
    st.rerun()


def is_food(shown):
    if FOODS[shown] is None:
        return ss["goal"] == "Food" and ss["choice"] == "Custom" and ss["custom_name"] == shown
    return ss["goal"] == "Food" and ss["choice"] == FOODS[shown]


st.title("What to grow")
st.write("Pick one, a food or a thing.")
left, right = st.columns(2)

with left:
    st.subheader("Food")
    for shown, name in FOODS.items():
        on = is_food(shown)
        if st.button(shown, key=f"food_{shown}", type="primary" if on else "secondary", use_container_width=True):
            pick("Food", name or "Custom", shown if name is None else "")
        if on:
            st.caption(f"*{latin.get(name) or LATIN_EXTRA[shown]}*")
    own = ss["goal"] == "Food" and ss["choice"] == "Custom" and ss["custom_name"] not in FOODS
    if st.button(FOOD_OWN, key="food_own", type="primary" if own else "secondary", use_container_width=True):
        pick("Food", "Custom", "")
    if own:
        name = st.text_input("What is it called?", value=ss["custom_name"], key="food_own_in")
        ss["custom_name"] = name.strip()
        st.caption("For your own mushroom we have no temperature range or yield figure; "
                   "the plan uses the oyster recipe and says so.")

with right:
    st.subheader("Thing")
    for p in THINGS:
        on = ss["goal"] == "Materials" and ss["choice"] == p
        if st.button(p, key=f"thing_{p}", type="primary" if on else "secondary", use_container_width=True):
            pick("Materials", p)
    own = ss["goal"] == "Materials" and ss["choice"] not in THINGS
    if st.button(THING_OWN, key="thing_own", type="primary" if own else "secondary", use_container_width=True):
        pick("Materials", "Your own invention", "")
    if own:
        name = st.text_input("What is it called?", value=ss["custom_name"], key="thing_own_in")
        ss["custom_name"] = name.strip()
        ss["choice"] = name.strip() or "Your own invention"
    st.caption("The mycelium is Pleurotus ostreatus, the species in our material data.")

if ss["choice"]:
    st.success(f"Chosen: {flow.choice_label()}")
    st.page_link("pages/3_Environment.py", label="Next: your environment", icon="➡️")
