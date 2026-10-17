"""Calendar and tasks: a Preview page that shows the dates from the pipeline as events."""

import streamlit as st

from mycodemo.app import flow

st.set_page_config(page_title="Calendar and tasks", page_icon="📅", layout="wide")
flow.init()
ss = st.session_state
flow.need("batch", "Recipe and first prediction", "pages/5_Recipe_and_first_prediction.py")

st.title("Calendar and tasks")
result = flow.run()
dates = result.dates
if dates.value:
    rows = "".join(f"<tr><td>{a.strftime('%a %d %b')}"
                   f"{'' if a == b else ' – ' + b.strftime('%a %d %b')}</td><td>{name}</td></tr>"
                   for name, a, b in dates.value)
    body = f"Predicted dates from your plan:<table>{rows}</table>Source: {dates.source}. {dates.note}"
else:
    body = dates.note
flow.preview("Calendar", body)
for w in result.warnings.value:
    st.warning(w)
flow.preview("Tasks and reminders", "Watering, checking the photos, starting post-treatment: reminders on your phone.")
st.button("Add to my calendar", disabled=True)
st.button("Send me reminders", disabled=True)
