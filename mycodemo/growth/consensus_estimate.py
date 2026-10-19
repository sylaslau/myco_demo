"""Colonisation time estimate from the 2.2 consensus regression (notebook 2.2).

The maths of notebook 2.2 lives here; the notebook fits the model and imports this module, the app loads the saved
bundle (model.json, settings.json, prepared_consensus.csv in models/colonisation/consensus_demo_*) and predicts.

Endpoint (project assumption, 10-18): the consensus typical colonisation day is taken as 90 % visible coverage; the
induction planning marker is that day + `post_90_days` (2). Temperature and low-humidity rules are provisional demo
assumptions, not validated biology. Nothing here touches streamlit.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mycodemo.config import MODELS_DIR

BUNDLE_PARENT = MODELS_DIR / "colonisation"
BUNDLE_FILES = ("model.json", "settings.json", "prepared_consensus.csv")
FEATURE_FIELDS = ("species", "substrate_code", "species_regime")

# Values transcribed from the supplied review summary, not independently verified.
REVIEW_OPTIMA = {
    "Pleurotus ostreatus": {"substrate_core": (23., 25.), "air": (18., 20.)},
    "Pleurotus pulmonarius": {"substrate_core": (24., 28.), "air": (20., 24.)},
    "Pleurotus eryngii": {"substrate_core": (24., 24.), "air": (18., 22.)},
    "Lentinula edodes": {"substrate_core": (23., 25.), "air": (18., 22.)},
    "Ganoderma lucidum": {"substrate_core": (27., 30.), "air": (24., 26.)},
}
SPECIES_ALIASES = {
    "pearl oyster": "Pleurotus ostreatus", "blue oyster": "Pleurotus ostreatus",
    "phoenix oyster": "Pleurotus pulmonarius", "pink oyster": "Pleurotus djamor",
    "king oyster": "Pleurotus eryngii", "shiitake": "Lentinula edodes",
    "lion's mane": "Hericium erinaceus", "reishi": "Ganoderma lucidum",
    "enoki": "Flammulina velutipes",
}
SUBSTRATE_ALIASES = {"cotton waste": "cotton", "natural logs": "logs", "master's mix": "masters_mix"}


class ModelUnavailable(FileNotFoundError):
    """The 2.2 bundle is missing or incomplete; callers fall back to the old range."""


def make_settings(ridge_alpha=0.10, post_90_days=2.0, temp_rate_loss_per_c=0.05, min_relative_rate=0.25,
                  max_temp_distance_c=10.0, low_rh_threshold=60.0, low_rh_delay_days=(5.0, 10.0),
                  harvest_spread=0.25, review_optima=None):
    """The settings the maths reads, with the same keys as settings.json of a 2.2 bundle."""
    return {
        "ridge_alpha": ridge_alpha, "post_90_days": post_90_days,
        "temperature_rate_loss_per_degree_assumed": temp_rate_loss_per_c,
        "temperature_min_relative_rate_assumed": min_relative_rate,
        "temperature_max_distance_degrees_assumed": max_temp_distance_c,
        "low_RH_threshold_assumed": low_rh_threshold,
        "low_RH_drying_exposure_delay_scenario": list(low_rh_delay_days),
        "harvest_relative_spread_assumed": harvest_spread,
        "review_temperature_optima": REVIEW_OPTIMA if review_optima is None else review_optima,
    }


def finite_number(value, name, lower=None, upper=None):
    value = float(value)
    if not np.isfinite(value) or (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def find_bundle(parent=None):
    """Newest consensus_demo_* folder under `parent` that holds all BUNDLE_FILES; ModelUnavailable if none."""
    parent = Path(parent) if parent is not None else BUNDLE_PARENT
    found = sorted(p for p in parent.glob("consensus_demo_*") if p.is_dir() and all((p / f).is_file() for f in BUNDLE_FILES))
    if not found:
        raise ModelUnavailable(f"No complete consensus_demo_* bundle in {parent}")
    return found[-1]


def load_bundle(path=None):
    """Load the saved 2.2 model, settings and prepared consensus rows. `path` = a bundle folder (default: newest)."""
    folder = find_bundle() if path is None else Path(path)
    for name in BUNDLE_FILES:
        if not (folder / name).is_file():
            raise ModelUnavailable(f"Missing {folder / name}")
    try:
        return ConsensusModel(json.loads((folder / "model.json").read_text()),
                              json.loads((folder / "settings.json").read_text()),
                              pd.read_csv(folder / "prepared_consensus.csv").fillna(""), folder)
    except (ValueError, KeyError) as exc:
        raise ModelUnavailable(f"Unreadable bundle {folder}: {exc}") from exc


@dataclass
class ConsensusModel:
    """model: levels + coefficients (list or array); settings: see make_settings; references: the 22 consensus rows."""
    model: dict
    settings: dict
    references: pd.DataFrame
    path: Path | None = None

    @property
    def post_90_days(self):
        return finite_number(self.settings["post_90_days"], "post-90 allowance", 0)

    def predict_reference(self, frame):
        x = np.column_stack([np.ones(len(frame))] + [frame[field].eq(value).to_numpy(float)
                                                     for field in FEATURE_FIELDS for value in self.model["levels"][field]])
        pred = x @ np.asarray(self.model["coefficients"], float)
        typ = np.exp(pred[:, 0])
        return pd.DataFrame({
            "t90_lower": typ * np.exp(-np.maximum(pred[:, 1], 0)), "t90_typical": typ,
            "t90_upper": typ * np.exp(np.maximum(pred[:, 2], 0)),
            "pinning_typical": np.exp(pred[:, 3]), "development_typical": np.exp(pred[:, 4]),
        }, index=frame.index)

    def selected_reference(self, species, substrate):
        refs = self.references
        species = SPECIES_ALIASES.get(str(species).strip().lower(), str(species).strip())
        exact = {s.lower(): s for s in refs.species.unique()}
        species = exact.get(species.lower(), species)
        substrate = SUBSTRATE_ALIASES.get(str(substrate).strip().lower(), str(substrate).strip().lower())
        selected = refs[(refs.species == species) & (refs.substrate_code == substrate)]
        if len(selected) != 1:
            choices = refs.loc[refs.species.eq(species), "substrate_code"].tolist()
            raise ValueError(f"Unsupported species/substrate pair: {species} / {substrate}. Available substrates: {choices}")
        return selected

    def environment_rules(self, row, temperature_c, temperature_basis, humidity_pct, drying_exposed):
        s = self.settings
        notes, factor, available = [], 1.0, True
        if temperature_basis not in ("air", "substrate_core", "unknown"):
            raise ValueError("temperature_basis must be air, substrate_core or unknown")
        band = s["review_temperature_optima"].get(row.species, {}).get(temperature_basis)
        band_source = "review optimum band"
        if band is None and temperature_basis in ("air", "substrate_core"):
            band = (float(row.incubation_temp_opt_C),) * 2
            band_source = "consensus optimum (ref_2), basis not stated in source: demo assumption"
        if temperature_c is None:
            notes.append("Temperature omitted: reference scenario, no adjustment.")
        else:
            temp = finite_number(temperature_c, "temperature", -20, 60)
            if band is None:
                available = False
                notes.append("No comparable temperature basis/recommendation; numeric adjusted forecast unavailable.")
            else:
                distance = max(band[0] - temp, 0., temp - band[1])
                if distance > s["temperature_max_distance_degrees_assumed"] or temp >= 35.:
                    available = False
                    notes.append("Temperature outside the provisional quantitative domain; numeric adjusted forecast unavailable.")
                else:
                    rate = max(s["temperature_min_relative_rate_assumed"],
                               1. - s["temperature_rate_loss_per_degree_assumed"] * distance)
                    factor = 1. / rate
            if band is not None:
                notes.append(f"{temperature_basis} optimum {tuple(band)}; {band_source}; no air/core conversion.")
        delay = (0., 0., 0.)
        if humidity_pct is None:
            notes.append("Ambient RH not supplied; no humidity adjustment.")
        else:
            rh = finite_number(humidity_pct, "humidity", 0, 100)
            if rh < s["low_RH_threshold_assumed"] and drying_exposed:
                low, high = s["low_RH_drying_exposure_delay_scenario"]
                delay = (low, (low + high) / 2, high)
                notes.append("Low-RH AND drying-exposure scenario: assumed 5-10 day delay; not measured substrate dryness.")
            elif rh < s["low_RH_threshold_assumed"]:
                notes.append("Low room RH, but drying exposure disabled: no duration penalty assigned.")
            if rh < float(row.incubation_RH_pct):
                notes.append("RH below this reference recommendation; the table alone gives no calibrated RH-duration slope.")
            if rh > 95:
                notes.append("High ambient RH does not establish excess substrate moisture; no automatic duration penalty.")
        return {"available": available, "temperature_duration_factor": factor,
                "humidity_delay_days": delay, "optimum_band": band, "notes": notes}

    def estimate_days(self, species, substrate, temperature_c=None, humidity_pct=None, temperature_basis="unknown",
                      drying_exposed=True, elapsed_days=0., induction_day=None, induction_confirmed=False):
        """Typical day and range (days since inoculation) at which 90 % is visible, the +2-day planning marker and
        the conditional first-harvest day. Raises ValueError for unsupported pairs or invalid inputs."""
        post_90, spread = self.post_90_days, self.settings["harvest_relative_spread_assumed"]
        elapsed = finite_number(elapsed_days, "elapsed days", 0)
        frame = self.selected_reference(species, substrate)
        row = frame.iloc[0]
        fit = self.predict_reference(frame).iloc[0]
        rules = self.environment_rules(row, temperature_c, temperature_basis, humidity_pct, drying_exposed)
        source_range = [float(row.colonisation_days_min), float(row.colonisation_days_typical), float(row.colonisation_days_max)]
        reference_t90 = [float(fit.t90_lower), float(fit.t90_typical), float(fit.t90_upper)]
        adjusted = None
        if rules["available"]:
            adjusted = [v * rules["temperature_duration_factor"] + d
                        for v, d in zip(reference_t90, rules["humidity_delay_days"])]
        marker = None if adjusted is None else [v + post_90 for v in adjusted]
        special = row.species == "Lentinula edodes" or row.regime == "natural_logs"
        notes = list(rules["notes"])
        if special:
            notes.append("Maturation/log regime: 90% + 2 days is a planning marker, not verified induction readiness. "
                         "Review guide actions.")
        harvest_typ = float(fit.pinning_typical + fit.development_typical)
        harvest_stage = [harvest_typ * (1 - spread), harvest_typ, harvest_typ * (1 + spread)]
        harvest = None
        if induction_day is not None:
            actual_induction = finite_number(induction_day, "induction day", 0)
            harvest = [actual_induction + v for v in harvest_stage]
            notes.append("Harvest clock anchored to entered induction day; incubation adjustment does not alter fruiting duration.")
        elif marker is not None and (not special or induction_confirmed):
            harvest = [a + b for a, b in zip(marker, harvest_stage)]
            notes.append("Harvest assumes induction starts at the planning marker and suitable fruiting conditions follow.")
        else:
            notes.append("Harvest date unavailable until an induction day or appropriate maturation confirmation is provided.")
        remaining = lambda days: None if days is None else [max(0., v - elapsed) for v in days]
        return {
            "species": row.species, "substrate": row.substrate_code, "ref_id": row.ref_id,
            "source_colonisation_days_min_typ_max": source_range,
            "source_first_harvest_stage_typical_days": float(row.days_to_first_harvest_typical),
            "consensus_regression_t90_min_typ_max": reference_t90,
            "adjusted_t90_min_typ_max": adjusted,
            "planning_90_plus_2_day_min_typ_max": marker,
            "planning_90_plus_2_days_remaining": remaining(marker),
            "harvest_day_min_typ_max": harvest,
            "harvest_days_remaining": remaining(harvest),
            "harvest_stage_min_typ_max": harvest_stage,
            "harvest_range_basis": "assumed +/-25% stage envelope; not sourced bounds",
            "elapsed_days": elapsed, "conditions": {"temperature_c": temperature_c, "temperature_basis": temperature_basis,
                                                   "humidity_pct": humidity_pct, "drying_exposed": drying_exposed},
            "environment_rules": rules, "needs_maturation_review": special,
            "induction_actions": row.induction_actions, "colonisation_cautions": row.colonisation_cautions,
            "source_reference": row.source_reference, "source_confidence_as_recorded": row.confidence,
            "endpoint_basis": "typical consensus duration treated as 90%; +2 days once; range envelope assumed",
            "model_basis": "consensus regression plus provisional environmental rules; not experimentally validated",
            "notes": notes,
        }
