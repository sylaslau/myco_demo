"""Grow Calculator: original vs corrected vs MycoDemo, side by side."""

import pandas as pd
import streamlit as st

from mycodemo.cultivation import SPECIES_BE, grow_plan
from mycodemo.cultivation.corrected import grow_plan_corrected, spawn_corrected

st.set_page_config(page_title="Grow Calculator", page_icon="🧮", layout="wide")
st.title("Grow Calculator")
st.caption("Same inputs, three answers. Masses in one unit; prices per that unit.")

# --- inputs ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Your batch")
    unit = st.radio("Unit", ["kg", "lb"], horizontal=True)
    blocks = st.number_input("Number of blocks / bags", 1, 10000, 10)
    weighed = st.number_input(f"Substrate per block, as weighed ({unit})", 0.1, 100.0, 2.0, 0.1)
    start_m = st.slider("Moisture already in it (%)", 0, 60, 10,
                        help="Bagged pellets and dry straw are usually about 10 %. 0 = oven-dried.")
    target_m = st.slider("Target moisture (%)", 40, 75, 60)
    st.divider()
    basis_label = st.radio("Spawn rate is given as", ["% of the whole mix", "% of wet substrate"])
    basis = "of_total_mix" if basis_label.startswith("% of the whole") else "of_hydrated"
    spawn_pct = st.slider("Spawn rate (%)", 1, 40, 10)
    st.divider()
    species = st.selectbox("Species (sets a typical BE)", list(SPECIES_BE) + ["Custom"])
    be = st.number_input("Biological efficiency, BE (%)", 0, 300,
                         SPECIES_BE.get(species, 85))
    st.divider()
    cost_sub = st.number_input(f"Substrate cost per {unit}", 0.0, 100.0, 0.5, 0.05)
    cost_spawn = st.number_input(f"Spawn cost per {unit}", 0.0, 100.0, 2.0, 0.1)
    extra = st.number_input("Other cost per block (bag, filter, labour)", 0.0, 100.0, 0.5, 0.1)
    price = st.number_input(f"Sale price per {unit} of mushrooms", 0.0, 200.0, 10.0, 0.5)

# The original only knows "% of wet substrate", so convert if needed.
if basis == "of_total_mix":
    spawn_for_original = spawn_pct / (1 - spawn_pct / 100)
else:
    spawn_for_original = spawn_pct

orig = grow_plan(weighed, blocks, target_m, spawn_for_original, be,
                 cost_sub, cost_spawn, extra, price)
corr = grow_plan_corrected(weighed, blocks, start_moisture=start_m, moisture=target_m,
                           spawn_pct=spawn_pct, spawn_basis=basis, be=be,
                           cost_sub=cost_sub, cost_spawn=cost_spawn,
                           extra_per_block=extra, sale_price=price)

# --- MycoDemo warnings (first, simple version of M4) -------------------------------
warnings = list(corr["warnings"])
mix_pct = corr["per_block"]["spawn_pct_of_total_mix"]
if not 5 <= mix_pct <= 20:
    warnings.append(f"Spawn is {mix_pct:.1f} % of the whole mix, outside the 5–20 % range in our data.")
if start_m == 0:
    warnings.append("Starting moisture 0 % assumes oven-dried material; most bagged material holds about 10 %.")
if species not in ("Blue / Pearl Oyster", "Pink / Yellow Oyster"):
    warnings.append("MycoDemo's growth data is for oyster mushrooms; other species use the original rules of thumb only.")

# --- results --------------------------------------------------------------------------
ob, cb = orig["per_block"], corr["per_block"]
rows = [
    ("Dry matter per block", weighed, cb["dry"]),
    ("Water to add per block", ob["water"], cb["water"]),
    ("Wet substrate per block", ob["hydrated"], cb["hydrated"]),
    ("Spawn per block", ob["spawn"], cb["spawn"]),
    ("Spawn, % of wet substrate", 100 * ob["spawn"] / ob["hydrated"], cb["spawn_pct_of_hydrated"]),
    ("Spawn, % of whole mix", 100 * ob["spawn"] / (ob["hydrated"] + ob["spawn"]), cb["spawn_pct_of_total_mix"]),
    ("Expected yield per block", ob["yield"], cb["yield"]),
    ("Expected yield, batch", orig["totals"]["yield"], corr["totals"]["yield"]),
    ("Batch cost", orig["economics"]["total_cost"], corr["economics"]["total_cost"]),
    ("Profit", orig["economics"]["profit"], corr["economics"]["profit"]),
]
table = pd.DataFrame(rows, columns=["", "Original", "Corrected"]).set_index("")
table["MycoDemo"] = table["Corrected"]

col1, col2, col3 = st.columns(3)
for col, name, plan in [(col1, "Original", orig), (col2, "Corrected", corr), (col3, "MycoDemo", corr)]:
    with col:
        st.subheader(name)
        st.metric(f"Water per block ({unit})", f"{plan['per_block']['water']:.2f}")
        st.metric(f"Spawn per block ({unit})", f"{plan['per_block']['spawn']:.2f}")
        st.metric(f"Yield, batch ({unit})", f"{plan['totals']['yield']:.1f}")
        st.metric("Profit", f"{plan['economics']['profit']:.2f}")
        if name == "MycoDemo":
            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("No warnings for these inputs.")
            st.caption("Colonisation forecast: coming with the Monitor page.")

st.subheader("All numbers")
st.dataframe(table.style.format("{:.2f}"))

with st.expander("Why do the columns differ?"):
    st.markdown(
        f"""
- **Starting moisture.** The original treats {weighed} {unit} as bone dry. With {start_m} % water already in it,
  the dry matter is {cb['dry']:.2f} {unit}, so less water is added.
- **Yield.** BE is defined on dry substrate, so the corrected yield uses the dry matter.
- **Spawn basis.** The original counts spawn as % of wet substrate; MycoDemo counts % of the whole mix.
  Both are shown in the table.
- **BE is an assumption** ({be} %), not a prediction. It will become a range from real bag data later.
"""
    )
