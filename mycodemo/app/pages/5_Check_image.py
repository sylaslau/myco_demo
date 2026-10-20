"""Check image: mark one uploaded photo with the 2.4 marking tool; the checked % replaces the HSV guess in the forecast."""

from datetime import date

import streamlit as st

from mycodemo import pipeline
from mycodemo.app import components, flow
from mycodemo.growth.coverage import days_since
from mycodemo.growth.traced_coverage import coverage_from_marks, editor_image

st.set_page_config(page_title="Check image", page_icon="🔍", layout="wide")
flow.init()
st.title("Check image")

ss = st.session_state
SAMPLE_PHOTOS = ss.setdefault("sample_photos", {})      # {sample ID: [photo rows]}, filled on Monitor
SAVES = ss.setdefault("check_saves", {})                # {(sample, file): saved_at of the last Save handled}
PLAN = ss.get("batch")


@st.cache_data(show_spinner=False, max_entries=16)
def photo_url(data: bytes):
    return editor_image(data)


def label(r):
    tag = " · checked by you" if r.get("checked") else " · marked, not ticked" if r.get("method") == pipeline.MARKED_METHOD else ""
    return f"{r['file'].removeprefix('upload: ')} (day {r['day']}){tag}"


samples = [s for s, rows in SAMPLE_PHOTOS.items() if any(r.get("data") for r in rows)]
if not samples:
    st.info("Upload a photo of a sample on Monitor first, then press Check image there.")
    st.page_link("pages/4_Monitor.py", label="Monitor", icon="📷")
    st.stop()

target = ss.get("check_target") or {}
s_col, p_col = st.columns(2)
sample = s_col.selectbox("Sample", samples, index=samples.index(target["sample"]) if target.get("sample") in samples else 0)
rows = [r for r in SAMPLE_PHOTOS[sample] if r.get("data")]
files = [r["file"] for r in rows]
default = files.index(target["file"]) if target.get("sample") == sample and target.get("file") in files else len(rows) - 1
row = rows[p_col.selectbox("Photo", range(len(rows)), index=default, format_func=lambda i: label(rows[i]))]
key = (sample, row["file"])
ss["check_target"] = {"sample": sample, "file": row["file"]}

batch_label = PLAN["batch_label"] if PLAN else ""
start = PLAN.get("inoc_date") if PLAN else None
value = components.check_image(photo=photo_url(row["data"]), photo_id=f"{sample}|{row['file']}", file_name=row["file"].removeprefix("upload: "),
                    batch_id=batch_label, batch_start=start.isoformat() if start else "", sample_id=sample,
                    date=row.get("date", ""), record=row.get("marks"), key=f"check_{sample}_{row['file']}", default=None)

# --- a new Save: work out the % from the marks and store it in this photo's row ----------------------
if value and value.get("photo_id") == f"{sample}|{row['file']}" and SAVES.get(key) != value.get("saved_at"):
    SAVES[key] = value.get("saved_at")
    marks = value["record"]
    try:
        res = coverage_from_marks(row["data"], marks)
    except ValueError as err:
        st.error(f"No % from these marks: {err} Change the marks and Save again.")
    else:
        row.setdefault("hsv_coverage", row["coverage"] if row.get("method") != pipeline.MARKED_METHOD else None)
        row.setdefault("hsv_method", row["method"])
        row.update(coverage=res["coverage_pct"] / 100, method=pipeline.MARKED_METHOD, checked=bool(value.get("reviewed")),
                   marks=marks, uncertainty_pct=res["uncertainty"], ambiguous_share=res["ambiguous_share"])
        if marks.get("date") and start:                 # the date can be corrected in the editor
            try:
                row.update(date=marks["date"], day=days_since(date.fromisoformat(marks["date"]), start))
            except ValueError:
                pass
        ss["pipeline_result"] = pipeline.run(flow.sample_batch())

# --- result and feedback (shown once this photo has been marked) ---------------------------------------
if row.get("method") == pipeline.MARKED_METHOD:
    st.subheader("Result")
    pct = 100 * row["coverage"]
    before = row.get("hsv_coverage")
    st.markdown(f"**{pct:.0f}% colonised**, marked by you"
                + (f" (the quick guess was {100 * before:.0f}%, {row.get('hsv_method', 'HSV')})." if before is not None else "."))
    if row.get("ambiguous_share", 0) > 0.35:
        st.warning(f"{100 * row['ambiguous_share']:.0f}% of the outline is 'not sure' (amber). More clicks or the brush "
                   "make the % firmer.")

    missing = []
    if not PLAN:
        missing.append("no plan batch yet (make one on Recipe and timeline)")
    elif not start:
        missing.append("no inoculation date in the plan batch")
    if missing:
        st.info(f"{pct:.0f}% of the surface colonised. No typical range and no induction day: {'; '.join(missing)}.")
    else:
        with_batch = flow.sample_batch(force_checked=key)        # as if ticked, so an unticked photo can say what it would do
        band = pipeline.typical_band(with_batch, row["day"])
        st.markdown(pipeline.pace_text(pct, row["day"], band))
        if band is None:
            st.caption("No Colonisation time estimate for this mushroom, so there is no typical range.")
        new = pipeline.induction_date(with_batch)
        old = pipeline.induction_date(flow.sample_batch(exclude=key))
        if new is None or old is None:
            st.info("No induction day: there is no Colonisation forecast for this mushroom.")
        elif row.get("checked"):
            st.markdown(pipeline.induction_text(new, old))
        else:
            st.markdown(pipeline.induction_text(new, old).replace("Estimated induction day:", "With this photo the "
                                                                  "estimated induction day would be"))
    if not row.get("checked"):
        st.warning("This photo is not used in the forecast until you tick “I checked the result” and Save again.")
    st.caption("Induction day = the Colonisation forecast's 90% day + 2 days (project assumption), compared with the "
               "forecast without this photo. Typical range = the Colonisation time estimate curve on this day, with its "
               "90% day at the earliest and the latest day of its range.")
