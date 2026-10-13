"""Grow Log & Photos: % colonised per day, S-curve, and the forecast as photos are added."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from mycodemo.growth.coverage import load_coverage
from mycodemo.growth.forecast import MODELS, forecast, forecast_by_photo, forecast_warnings

st.set_page_config(page_title="Grow Log & Photos", page_icon="📷", layout="wide")
st.title("Grow Log & Photos")
st.caption(
    "Each photo gives % colonised (**Photo coverage**). The S-curve fitted to those dots "
    "gives the day the dish reaches 90% and 95% (**Growth forecast**)."
)

cov = load_coverage()

with st.sidebar:
    st.header("Pick a growth")
    sid = st.selectbox("Dish (d005)", sorted(cov["series_id"].unique()))

g = cov[cov["series_id"] == sid].sort_values("day")
t, c = g["day"].to_numpy(float), g["coverage"].to_numpy(float)

n = st.slider("Photos used so far", 1, len(t), len(t),
              help="Move left to see the forecast as it looked with fewer photos.")
tn, cn = t[:n], c[:n]
res = forecast(tn, cn)

# --- warnings -------------------------------------------------------------------------
warns = forecast_warnings(tn, cn, res)
for w in warns:
    st.warning(w)

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
st.subheader("How the forecast changed as photos were added")
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
        "- Dots come from `data/interim/coverage_auto.csv` (d005 whole-dish scans).\n"
        "- A weak forecast is flagged when there are fewer than 3 photos or the dish never reaches 90%.\n"
        "- syldd photos will be added later."
    )
