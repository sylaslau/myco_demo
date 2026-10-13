"""Grow Log & Photos: % colonised per day, S-curve, and the Colonisation forecast as photos are added."""

from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mycodemo.growth.coverage import (
    SYLDD_INOCULATION,
    coverage,
    days_since,
    load_coverage,
    load_syldd_coverage,
    photo_datetime,
)
from mycodemo.growth.forecast import (
    MIN_PHOTOS,
    MODELS,
    forecast,
    forecast_by_photo,
    forecast_warnings,
    pace_vs_typical,
)

st.set_page_config(page_title="Grow Log & Photos", page_icon="📷", layout="wide")
st.title("Grow Log & Photos")
st.caption(
    "Each photo gives % colonised (**Photo coverage**). The S-curve fitted to those dots "
    "gives the day the dish reaches 90% and 95% (**Colonisation forecast**)."
)

with st.sidebar:
    st.header("Pick a growth")
    source = st.radio("Dataset", ["d005 (reference)", "syldd (my batch)"])

is_syldd = source.startswith("syldd")
if is_syldd:
    cov = load_syldd_coverage().rename(columns={"container_id": "series_id"})
    st.info("Photo coverage not yet checked on phone photos (threshold tuned on d005 scans)")
else:
    cov = load_coverage()

with st.sidebar:
    sid = st.selectbox("Dish", sorted(cov["series_id"].unique()))

g = cov[cov["series_id"] == sid].sort_values("day")

# --- own photos (syldd only; kept in this browser session, never written to disk) ---------
if is_syldd:
    with st.sidebar:
        st.subheader(f"Add your own photos to {sid}")
        ups = st.file_uploader("Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        extra = []
        for up in ups:
            try:
                default = photo_datetime(up).date()
            except ValueError:
                default = date.today()
            d = st.date_input(up.name, default, key=f"date_{sid}_{up.file_id}")
            up.seek(0)
            extra.append({"series_id": sid, "day": days_since(d, SYLDD_INOCULATION),
                          "coverage": coverage(up), "file": f"upload: {up.name}"})
        st.caption("Uploads stay in this session only; nothing is saved.")
    if extra:
        g = pd.concat([g, pd.DataFrame(extra)]).sort_values("day")

t, c = g["day"].to_numpy(float), g["coverage"].to_numpy(float)

n = st.slider("Photos used so far", 1, len(t), len(t),
              help="Move left to see the forecast as it looked with fewer photos.")
tn, cn = t[:n], c[:n]
res = forecast(tn, cn)

# --- warnings -------------------------------------------------------------------------
warns = forecast_warnings(tn, cn, res)
for w in warns:
    st.warning(w)

# --- plain-words summary ----------------------------------------------------------------
if len(tn) < MIN_PHOTOS:
    summary = "Not enough photos yet for a Colonisation forecast."
else:
    d90 = res["day90"]
    est = "not reached within 60 days" if np.isnan(d90) else f"around day {d90:.0f}"
    summary = (f"Latest photo: day {tn[-1]:.0f}, {100 * cn[-1]:.0f}% colonised (Photo coverage). "
               f"Colonisation forecast: 90% {est} (estimate).")
st.info(summary)
pace = pace_vs_typical(res["day90"]) if len(tn) >= MIN_PHOTOS else "unknown"
pace_text = {"ahead": "ahead of", "within": "within", "behind": "behind",
             "unknown": "not yet comparable with"}[pace]
st.markdown(f"Pace: **{pace_text}** the typical range. Typical oyster colonisation is 10-14 days "
            "(usemycro.com guidance, not our data).")

# --- chart ----------------------------------------------------------------------------
f = MODELS["logistic"][0]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t[n:], 100 * c[n:], "o", color="lightgrey", label="later photos (not used)")
ax.plot(tn, 100 * cn, "ko", label="photos used")
if res["params"] is not None:
    tt = np.linspace(0, max(t.max(), np.nan_to_num(res["day95"], nan=0)) + 3, 300)
    ax.plot(tt, 100 * f(tt, *res["params"]), color="tab:green", label="S-curve (logistic)")
    for p, key, style in [(90, "day90", "--"), (95, "day95", ":")]:
        if not np.isnan(res[key]):
            ax.axvline(res[key], color="tab:red", ls=style, label=f"{p}% on day {res[key]:.1f}")
ax.set_xlabel("day")
ax.set_ylabel("% colonised")
ax.set_ylim(0, 105)
ax.legend(loc="upper left", fontsize=8)
st.pyplot(fig)

c1, c2, c3 = st.columns(3)
c1.metric("Photos used", n)
c2.metric("90% reached on day", "n/a" if np.isnan(res["day90"]) else f"{res['day90']:.1f}")
c3.metric("95% reached on day", "n/a" if np.isnan(res["day95"]) else f"{res['day95']:.1f}")

# --- forecast as each photo is added ----------------------------------------------------
with st.expander("Details"):
    st.subheader("How the Colonisation forecast changed as photos were added")
    hist = forecast_by_photo(t, c)
    if hist.empty:
        st.info("Needs at least 2 photos.")
    else:
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(hist["n_photos"], hist["day90"], "o-", label="90% day")
        ax2.plot(hist["n_photos"], hist["day95"], "s--", label="95% day")
        ax2.axvline(n, color="grey", lw=0.8)
        ax2.set_xlabel("photos used")
        ax2.set_ylabel("forecast day")
        ax2.legend(fontsize=8)
        st.pyplot(fig2)
        show = hist.rename(columns={"n_photos": "Photos", "last_day": "Last photo (day)",
                                    "day90": "90% day", "day95": "95% day"})
        st.dataframe(show.style.format({"Last photo (day)": "{:.0f}", "90% day": "{:.1f}",
                                        "95% day": "{:.1f}"}, na_rep="n/a"), hide_index=True)

with st.expander("Notes"):
    st.markdown(
        "- d005 dots come from `data/interim/coverage_auto.csv` (whole-dish scans); "
        "syldd dots from `data/interim/coverage_syldd.csv` (phone photos, day counted from "
        "inoculation on 2026-10-08 using the photo time).\n"
        "- A weak forecast is flagged when there are fewer than 3 photos or 90% is not reached within 60 days.\n"
        "- Uploaded photos are not saved."
    )
