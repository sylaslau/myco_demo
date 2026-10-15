"""Colonisation plot for one group: baseline band, the user's logged % colonised, and the Colonisation forecast."""

import numpy as np
import plotly.graph_objects as go

from mycodemo.growth.forecast import forecast, logistic

# days to 95% (fastest, slowest) per plot_group: lowest Min and highest Max day over the group's rows in
# colonisation_consensus.csv, as printed by notebook 2.1 section 6. Guide values, not measurements.
BASELINE_95_DAYS = {
    "Cool oyster": (10.0, 30.0),
    "Tropical oyster": (8.0, 30.0),
    "King oyster": (12.0, 25.0),
    "Volvariella": (4.0, 10.0),
    "Hericium": (10.0, 21.0),
    "Flammulina": (12.0, 25.0),
    "Ganoderma": (10.0, 28.0),
    "Cyclocybe": (14.0, 28.0),
    "Pholiota": (14.0, 30.0),
    "Trametes": (14.0, 30.0),
    "Auricularia": (18.0, 35.0),
    "Lentinula": (18.0, 70.0),
    "Grifola": (25.0, 60.0),
}

BAND_NAME = "Baseline range (guides)"
MID_FRAC = 0.6          # 50% at 0.6 of the days to 95%, the same illustrative shape as notebook 2.1


def logistic_to_95(t, d95):
    """% colonised at day t: logistic with 95% at day d95 (50% at MID_FRAC * d95)."""
    t50 = MID_FRAC * d95
    k = np.log(19) / (d95 - t50)
    return 100 * logistic(np.asarray(t, dtype=float), k, t50)


def colonisation_plot(group, user_log=None, band=None):
    """Plotly figure, x = days since inoculation, y = % colonised.

    band: (fast_days, slow_days) to 95%; default is the group's baseline range.
    user_log: DataFrame with columns day, coverage_pct (0-100).
    Traces: 1 band, +1 for the user's points, +1 for the forecast (needs 2 points).
    """
    if band is None:
        if group not in BASELINE_95_DAYS:
            raise ValueError(f"No baseline range for group {group!r}; pass band=(fast_days, slow_days).")
        band = BASELINE_95_DAYS[group]
    fast, slow = sorted(band)

    log = None
    if user_log is not None and len(user_log):
        log = user_log[["day", "coverage_pct"]].dropna().sort_values("day")
    x_max = 1.1 * slow
    if log is not None and len(log):
        x_max = max(x_max, float(log["day"].max()) + 3)
    days = np.linspace(0, x_max, 200)

    fig = go.Figure()
    fig.add_scatter(x=np.r_[days, days[::-1]], y=np.r_[logistic_to_95(days, fast), logistic_to_95(days, slow)[::-1]],
                    fill="toself", fillcolor="rgba(31,119,180,0.18)", line=dict(width=0), name=BAND_NAME,
                    hoverinfo="skip")

    note = None
    if log is None or not len(log):
        note = "needs 2 photos"
    else:
        fig.add_scatter(x=log["day"], y=log["coverage_pct"], mode="lines+markers", name="Your colonisation",
                        line=dict(color="black", width=2), marker=dict(size=8))
        if len(log) < 2:
            note = "needs 2 photos"
        else:
            res = forecast(log["day"].to_numpy(float), log["coverage_pct"].to_numpy(float) / 100)
            if res["params"] is None:
                note = "forecast could not be fitted"
            else:
                last = float(log["day"].iloc[-1])
                tt = np.linspace(last, max(60, last), 600)
                yy = 100 * logistic(tt, *res["params"])
                keep = np.where(yy >= 99.5)[0]
                end = keep[0] + 1 if len(keep) else len(tt)
                fig.add_scatter(x=tt[:end], y=yy[:end], mode="lines", name="Colonisation forecast",
                                line=dict(color="crimson", dash="dot", width=2))
    if note:
        fig.add_annotation(text=note, xref="paper", yref="paper", x=0.02, y=0.96, showarrow=False)
    fig.update_layout(title=group, xaxis_title="days since inoculation", yaxis_title="% colonised",
                      yaxis_range=[0, 102], height=480)
    return fig
