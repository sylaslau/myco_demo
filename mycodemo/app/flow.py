"""Shared helpers for the grower pages: choices carried in st.session_state, the Preview style, lists."""

from dataclasses import replace
from datetime import date

import streamlit as st

from mycodemo import pipeline
from mycodemo.growth.forecast import (
    pace_vs_typical,  # noqa: F401  (typical range lives with the forecast)
)

MATERIAL_SPECIES = pipeline.MATERIAL_SPECIES
PRODUCTS, ENVIRONMENTS, BE_KEY = pipeline.PRODUCTS, pipeline.ENVIRONMENTS, pipeline.BE_KEY
FULLY_COLONISED_NOTE = ("Start induction when the substrate is white all the way through, bottom included; "
                        "90% on a photo is only a surface marker.")
ENDPOINT_ASSUMPTION_NOTE = (
    "Project assumption, not a measured result: the consensus typical colonisation day is taken as **90% visible "
    "coverage**, and the **Induction planning marker** is that day + 2 days. The day the substrate is white all the "
    "way through, bottom included, cannot be predicted; check by eye before starting induction.")
STAGES_FOOD = ["Inoculated", "Colonising", "Fruiting", "Harvested", "Contaminated"]
STAGES_MATERIAL = ["Pre-treatment", "Colonisation", "Liquid fermentation", "Post-treatment", "Contaminated"]

PREVIEW_CSS = """
<style>
.mc-preview {border: 2px dashed #9aa0a6; border-radius: 8px; padding: 0.7rem 1rem; margin: 0.6rem 0;
  background: rgba(150,150,150,0.12); color: #8a8f94; filter: grayscale(1);}
.mc-preview b {color: #8a8f94;}
.mc-tag {display: inline-block; border: 1px solid #9aa0a6; border-radius: 4px; padding: 0 6px;
  font-size: 0.75rem; letter-spacing: 0.05em; margin-right: 8px; color: #8a8f94;}
</style>
"""


def init():
    """Set the defaults once and load the Preview style; call at the top of every grower page."""
    for key, value in {"goal": None, "choice": None, "custom_name": "", "environment": None,
                       "temp_c": None, "recipe": None, "batch": None, "inoc_date": date.today(),
                       "pipeline_batch": None}.items():
        st.session_state.setdefault(key, value)
    st.markdown(PREVIEW_CSS, unsafe_allow_html=True)


def preview(title, body=""):
    """A greyed, dashed box: marks something pretend and inactive (a Preview)."""
    st.markdown(f'<div class="mc-preview"><span class="mc-tag">PREVIEW · inactive</span><b>{title}</b>'
                f'<br>{body}</div>', unsafe_allow_html=True)


def need(key, label, page):
    """Stop the page with a link back when an earlier choice is missing."""
    if not st.session_state.get(key):
        st.info(f"Please finish “{label}” first.")
        st.page_link(page, label=label)
        st.stop()


def get_batch():
    """The one batch record (kept in st.session_state): the page choices copied in, photos and conditions kept."""
    ss = st.session_state
    if ss.get("pipeline_batch") is None:
        ss["pipeline_batch"] = pipeline.Batch()
    b = ss["pipeline_batch"]
    b.goal, b.choice, b.custom_name = ss["goal"], ss["choice"], ss["custom_name"]
    b.environment, b.temp_c, b.recipe, b.inoc_date = ss["environment"], ss["temp_c"], ss["recipe"], ss["inoc_date"]
    b.plan_size = ss.get("plan_size")
    return b


def in_forecast(row):
    """A photo marked on Check image counts only once "I checked the result" was ticked; an HSV photo always counts."""
    return row.get("method") != pipeline.MARKED_METHOD or bool(row.get("checked"))


def sample_batch(exclude=None, force_checked=None):
    """The batch with every Monitor sample photo (ss['sample_photos']) copied in, for pipeline.run (the Recipe page
    reads the same record). exclude / force_checked = (sample ID, file) of one photo to leave out / to count as checked
    (Check image feedback): then a copy is returned and the shared record is left as it is."""
    ss = st.session_state
    b = get_batch()
    if b.goal is None:                           # no plan from the earlier pages: Monitor's own default
        b.goal, b.choice = "Food", "Blue Oyster"
    plan = ss.get("batch")
    if plan:
        b.inoc_date, b.temp_c = plan["inoc_date"], plan["temp_c"]
    photos = []
    for sample, rows in ss.get("sample_photos", {}).items():
        for r in rows:
            key = (sample, r["file"])
            forced = key == force_checked
            if key == exclude or not (forced or in_forecast(r)):
                continue
            photos.append(pipeline.Photo(day=r["day"], file=r["file"], coverage=r["coverage"], method=r["method"],
                                         quality=r["quality"], sample=sample, checked=forced or bool(r.get("checked"))))
    if exclude is None and force_checked is None:
        b.photos = photos
        return b
    return replace(b, photos=photos)


def run():
    """pipeline.run on the current batch; every page shows this and nothing else."""
    return pipeline.run(get_batch())


def species_row():
    """The FRUITING_CONDITIONS entry for the chosen species (materials use MATERIAL_SPECIES); None if custom."""
    return pipeline.species_row(get_batch())


def choice_label():
    return pipeline.choice_label(get_batch())


def be_percent(row):
    return pipeline.be_percent(row)


def predicted_dates():
    """Predicted events as (label, from_date, to_date) from run(); empty list while no estimate exists."""
    return run().dates.value


def show_source(step):
    """A one-line source tag and note under a step's answer."""
    st.caption(f"Source: **{step.source}**. {step.note}")
