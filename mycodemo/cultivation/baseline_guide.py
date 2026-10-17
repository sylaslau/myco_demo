"""Lookup of the cultivation baseline (NotebookLM consensus table, ref_2).

Two stages are kept apart on purpose:
- colonisation(): stage 1, from inoculation until the substrate is fully colonised.
- fruiting(): stage 2, every day count is from the START OF INDUCTION; no colonisation
  days are included.
"""

from functools import lru_cache

import pandas as pd

from mycodemo.config import PROCESSED_DATA_DIR

BASELINE_CSV = PROCESSED_DATA_DIR / "cultivation_baseline.csv"
TEXT_CSV = PROCESSED_DATA_DIR / "cultivation_guide_text.csv"

FALLBACK_SUBSTRATE = "sawdust"


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    return pd.read_csv(BASELINE_CSV)


@lru_cache(maxsize=1)
def load_text() -> pd.DataFrame:
    return pd.read_csv(TEXT_CSV)


def species_list() -> list[str]:
    return list(dict.fromkeys(load()["species"]))


def substrates(species: str) -> list[str]:
    df = load()
    return df.loc[df["species"] == species, "substrate_code"].tolist()


def row(species_latin: str, substrate_code: str) -> tuple[pd.Series, str]:
    """Row for this species and substrate, with a note saying which row was used and why."""
    df = load()
    sub = df[df["species"] == species_latin]
    if sub.empty:
        raise KeyError(f"unknown species: {species_latin}")
    hit = sub[sub["substrate_code"] == substrate_code]
    if not hit.empty:
        r = hit.iloc[0]
        return r, f"{r['ref_id']}: {species_latin} on {substrate_code}"
    fallback = sub[sub["substrate_code"] == FALLBACK_SUBSTRATE]
    if not fallback.empty:
        r, used = fallback.iloc[0], FALLBACK_SUBSTRATE
        why = f"{FALLBACK_SUBSTRATE} row"
    else:
        r, used = sub.iloc[0], sub.iloc[0]["substrate_code"]
        why = f"first row ({used}), no {FALLBACK_SUBSTRATE} row"
    return r, (
        f"{r['ref_id']}: no {substrate_code} row for {species_latin}; used the {why}"
    )


def colonisation(r: pd.Series) -> dict:
    """Stage 1 only."""
    return {
        "colonisation_days": {
            "min": int(r["colonisation_days_min"]),
            "typical": int(r["colonisation_days_typical"]),
            "max": int(r["colonisation_days_max"]),
        },
        "incubation_temp_C": {
            "min": int(r["incubation_temp_min_C"]),
            "opt": int(r["incubation_temp_opt_C"]),
            "max": int(r["incubation_temp_max_C"]),
        },
        "incubation_rh_pct": int(r["incubation_RH_pct"]),
        "source": r["source_reference"],
        "confidence": r["confidence"],
    }


def fruiting(r: pd.Series) -> dict:
    """Stage 2 only; days are counted from the start of induction."""
    t = load_text()
    text = t[t["ref_id"] == r["ref_id"]].iloc[0]
    return {
        "days_to_pinning": int(r["days_to_pinning_typical"]),
        "pinning_to_harvest": int(r["days_pinning_to_harvest_typical"]),
        "first_harvest": int(r["days_to_first_harvest_typical"]),
        "fruiting_temp_C": {
            "min": int(r["fruiting_temp_min_C"]),
            "opt": int(r["fruiting_temp_opt_C"]),
            "max": int(r["fruiting_temp_max_C"]),
        },
        "fruiting_rh_pct": {
            "min": int(r["fruiting_RH_min_pct"]),
            "max": int(r["fruiting_RH_max_pct"]),
        },
        "co2_max_ppm": int(r["fruiting_CO2_max_ppm"]),
        "induction_actions": {
            "overview": text["induction_actions"],
            "temperature": text["induction_temperature_action"],
            "light": text["induction_light_action"],
            "fresh_air": text["induction_fresh_air_action"],
            "humidity": text["induction_humidity_action"],
            "other": text["induction_other_actions"],
        },
        "induction_evidence_status": r["induction_evidence_status"],
    }


def temp_check(temp: float, lo: float, hi: float) -> str:
    if temp < lo:
        return "below"
    if temp > hi:
        return "above"
    return "inside"


def spawn_on_our_basis(pct_wet: float) -> float:
    """Spawn as % of wet substrate -> % of (wet substrate + spawn)."""
    return pct_wet / (100 + pct_wet) * 100
