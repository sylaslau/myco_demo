"""Build the property table for M3: one row per sample with density and/or stress.

Run from the project root:  python -m mycodemo.property_data
"""

from pathlib import Path

import pandas as pd

from mycodemo.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

EXTRACTED_DIR = INTERIM_DATA_DIR / "extracted"
OUTPUT_PATH = PROCESSED_DATA_DIR / "property_table.csv"

# d005 PFL specimens LF_1/LF_2 sit in the pre-compression sheet but were left out of
# the PFL mean, and their treatment is unknown (flag CF7)
PRE_COMPRESSION_UNKNOWN = {"d005-S-PFL-C-4", "d005-S-PFL-C-5"}

# Broad substrate groups, so different labs can share a category
SUBSTRATE_GROUP = {
    "hemp": "hemp",
    "flax": "flax",
    "flax waste": "flax",
    "softwood": "softwood",
    "beech sawdust": "hardwood",
    "birch sanding dust": "hardwood",
    "birch sanding dust + hemp shives": "hardwood",
    "birch sanding dust + birch sawdust": "hardwood",
    "wheat straw": "straw",
    "straw": "straw",
    "wheat straw + hemp shives": "straw",
    "wheat straw + birch sawdust": "straw",
    "arboloco pith": "pith",
    "arboloco pith + kikuyu grass": "pith",
}


def load_tables(folder: Path = EXTRACTED_DIR) -> dict[str, pd.DataFrame]:
    """Read the extracted CSVs that M3 needs."""
    names = ["measurement", "sample", "fermentation", "substrate"]
    return {n: pd.read_csv(folder / f"{n}.csv", encoding="utf-8-sig") for n in names}


def build_property_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join measurement -> sample -> fermentation and substrate."""
    m = tables["measurement"].rename(
        columns={
            "compressive_stress_at_10pct_strain_MPa": "stress_mpa",
            "dry_density_kg_m3": "density_kg_m3",
        }
    )
    # Keep rows with strength and/or density
    m = m[m["stress_mpa"].notna() | m["density_kg_m3"].notna()]
    if m["sample_id"].duplicated().any():
        raise ValueError("A sample has more than one strength/density row")

    # n_specimens: how many blocks a row stands for (a group mean counts its group)
    s = tables["sample"][["sample_id", "n_specimens", "substrate_id", "fermentation_id"]]
    f = tables["fermentation"][
        ["fermentation_id", "genus", "species", "spawn_pct_wt",
         "incubation_temp_c", "total_growth_days", "in_process_treatment",
         "post_treatment"]
    ]
    sub = tables["substrate"][["substrate_id", "plant_name", "particle_size_class"]]

    t = m.merge(s, on="sample_id", how="left", validate="one_to_one")
    t = t.merge(f, on="fermentation_id", how="left", validate="many_to_one")
    t = t.merge(sub, on="substrate_id", how="left", validate="many_to_one")

    t["species_name"] = t["genus"] + " " + t["species"]
    # Species is a model input: blocks with no recorded fungus (d006 density blocks, flag HF7) stay out
    t = t[t["species_name"].notna()]
    t["substrate_group"] = t["plant_name"].map(SUBSTRATE_GROUP)
    t["coated"] = t["post_treatment"].str.contains("chitosan", na=False)
    # 1 = pressed while growing (d005 P samples, 100 -> 80 mm), 0 = not, empty = unknown
    t["pre_compressed"] = (
        t["in_process_treatment"].str.contains("compress", na=False).astype(float)
    )
    # 1 = compacted after growth, before drying (d023 compact_50pct, to half thickness).
    # Kept separate: growth after pressing (d005) did not raise density, compaction did.
    t["compacted_after_growth"] = (
        t["post_treatment"].str.contains("compact", na=False).astype(float)
    )
    t.loc[t["sample_id"].isin(PRE_COMPRESSION_UNKNOWN), "pre_compressed"] = float("nan")
    cols = [
        "dataset_id", "sample_id", "measurement_type", "n_specimens",
        "species_name", "plant_name", "substrate_group", "particle_size_class",
        "spawn_pct_wt", "incubation_temp_c", "total_growth_days",
        "pre_compressed", "compacted_after_growth", "post_treatment", "coated", "density_kg_m3", "stress_mpa",
    ]
    return t[cols].reset_index(drop=True)


def main(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    table = build_property_table(load_tables())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved {len(table)} rows to {output_path}")
    return table


if __name__ == "__main__":
    main()
