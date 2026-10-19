"""Colonisation forecast update (notebook 2.5): the 2.2 consensus curve, updated by dated photo coverage and the
condition history.

The curve is anchored to the 2.2 regression day (typical day = 90 % visible, +2 days = induction planning marker).
Conditions are integrated into an effective clock (past intervals are never rewritten); dated coverage moves the
90 % day inside a prior-penalised grid fit. With no coverage the forecast equals the 2.2 estimate for the same
conditions. Planning scenario, not validated biology; scenario ranges are not confidence intervals.
Nothing here touches streamlit or the disk.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mycodemo.growth.consensus_estimate import ConsensusModel, finite_number

LOG9 = float(np.log(9.))
CONDITION_FIELDS = ("temperature_c", "temperature_basis", "humidity_pct", "drying_exposed")

DEFAULT_UPDATE_RULES = {
    "curve_midpoint_fraction": 0.60,  # assumed 50% day / 90% day at reference conditions
    "photo_uncertainty_pct": 10.0,    # measurement tolerance; NOT a calibrated standard deviation
    "t90_log_prior_scale": 0.45,
    "slope_log_prior_scale": 0.50,
    "scenario_loss_tolerance": 2.0,   # plausible-grid envelope, NOT a confidence level
    "duration_scale_min": 0.20, "duration_scale_max": 5.0,
    "stale_photo_days": 3.0,
    "stall_span_days": 3.0, "stall_progress_fraction": 0.03,
    "large_decline_fraction": 0.15,
    "poor_fit_rmse_fraction": 0.15,
}


def logistic_coverage(effective_time, t90, slope):
    argument = np.clip(np.asarray(slope) * (np.asarray(effective_time) - np.asarray(t90)) + LOG9, -60, 60)
    return 1. / (1. + np.exp(-argument))


# --- records (pure) ---------------------------------------------------------------------------------
def record_day(row, inoculation_date):
    day = None if row.get("day") is None else finite_number(row["day"], "observation day", 0)
    if row.get("date") is not None:
        if inoculation_date is None:
            raise ValueError("Date inputs need inoculation_date.")
        observed, inoculated = pd.Timestamp(row["date"]), pd.Timestamp(inoculation_date)
        if pd.isna(observed) or pd.isna(inoculated):
            raise ValueError("Invalid observation/inoculation date.")
        try:
            date_day = (observed - inoculated).total_seconds() / 86400.
        except TypeError as exc:
            raise ValueError("Use matching timezone conventions for dates.") from exc
        date_day = finite_number(date_day, "date-derived day", 0)
        if day is not None and not np.isclose(day, date_day):
            raise ValueError("day and date disagree.")
        day = date_day
    if day is None:
        raise ValueError("Each observation needs day or date.")
    return day


def prepare_coverage(records, current_day, inoculation_date=None, rules=None):
    """Validated, day-sorted coverage records (quality_ok False ones are returned separately, never dropped silently)."""
    rules = DEFAULT_UPDATE_RULES if rules is None else rules
    rows, rejected, ids, days = [], [], set(), set()
    for record in records:
        row = dict(record)
        day = record_day(row, inoculation_date)
        if day > current_day:
            raise ValueError("Future coverage observation is not allowed.")
        if day in days:
            raise ValueError("Duplicate coverage time; select one observation or consolidate explicitly.")
        days.add(day)
        photo_id = row.get("photo_id")
        if photo_id is not None:
            if str(photo_id) in ids:
                raise ValueError("Repeated photo_id.")
            ids.add(str(photo_id))
        if "coverage" in row:
            raise ValueError("Ambiguous coverage units: use coverage_pct or coverage_fraction.")
        pct, fraction = row.get("coverage_pct"), row.get("coverage_fraction")
        if pct is None and fraction is None:
            raise ValueError("Missing coverage_pct or coverage_fraction.")
        if pct is not None:
            c = finite_number(pct, "coverage percentage", 0, 100) / 100.
        else:
            c = finite_number(fraction, "coverage fraction", 0, 1)
        if pct is not None and fraction is not None and not np.isclose(c, finite_number(fraction, "coverage fraction", 0, 1)):
            raise ValueError("Coverage percentage/fraction disagree.")
        quality_ok = row.get("quality_ok", True)
        if not isinstance(quality_ok, (bool, np.bool_)):
            raise ValueError("quality_ok must be boolean, not text.")
        uncertainty_pct, uncertainty_fraction = row.get("uncertainty_pct"), row.get("uncertainty_fraction")
        if uncertainty_pct is not None:
            uncertainty = finite_number(uncertainty_pct, "coverage tolerance", .1, 100) / 100.
        elif uncertainty_fraction is not None:
            uncertainty = finite_number(uncertainty_fraction, "fractional coverage tolerance", .001, 1)
        else:
            uncertainty = rules["photo_uncertainty_pct"] / 100.
        if uncertainty_pct is not None and uncertainty_fraction is not None:
            if not np.isclose(uncertainty, finite_number(uncertainty_fraction, "fractional coverage tolerance", .001, 1)):
                raise ValueError("Coverage tolerance percentage/fraction disagree.")
        normal = {"day": day, "coverage_fraction": c, "coverage_pct": 100 * c,
                  "photo_id": photo_id, "quality_ok": bool(quality_ok), "uncertainty_fraction": uncertainty}
        if quality_ok:
            rows.append(normal)
        else:
            rejected.append({**normal, "reason": "quality_ok=False"})
    return sorted(rows, key=lambda r: r["day"]), rejected


def merged_conditions(previous, update):
    unknown = set(update) - set(CONDITION_FIELDS) - {"day", "date", "batch_id", "species", "substrate"}
    if unknown:
        raise ValueError(f"Unknown condition fields: {sorted(unknown)}")
    result = dict(previous)
    result.update({k: update[k] for k in CONDITION_FIELDS if k in update})
    if not isinstance(result.get("drying_exposed"), (bool, np.bool_)):
        raise ValueError("drying_exposed must be boolean.")
    return result


def prepare_conditions(records, current_day, initial, inoculation_date=None):
    """History [{'day', 'conditions', 'origin'}] starting with the 2.2 assumed conditions; changes apply from their day."""
    normalized, seen = [], set()
    for record in records:
        day = record_day(record, inoculation_date)
        if day > current_day:
            raise ValueError("Future condition record is not allowed; use FUTURE_CONDITIONS instead.")
        if day in seen:
            raise ValueError("Duplicate condition time.")
        seen.add(day)
        normalized.append((day, record))
    history = [{"day": 0., "conditions": dict(initial), "origin": "2.2 assumed starting conditions"}]
    for day, record in sorted(normalized, key=lambda item: item[0]):
        item = {"day": day, "conditions": merged_conditions(history[-1]["conditions"], record), "origin": "recorded update"}
        if day == 0:
            history[0] = item
        else:
            history.append(item)
    return history


def observed_crossing(observations, confirmed_day, current_day):
    if confirmed_day is not None:
        day = finite_number(confirmed_day, "confirmed 90% day", 0, current_day)
        return [day, day, day], "explicitly confirmed crossing day"
    hits = [r for r in observations if r["coverage_fraction"] >= .9]
    if not hits:
        return None, None
    first = hits[0]
    earlier = [r["day"] for r in observations if r["day"] < first["day"] and r["coverage_fraction"] < .9]
    low = max(earlier) if earlier else 0.
    high = first["day"]
    return [float(low), float((low + high) / 2), float(high)], "observed crossing bracket; centre is an assumed midpoint"


# --- the model of one batch -------------------------------------------------------------------------
@dataclass
class ForecastModel:
    """2.5 maths for one species/substrate pair of a 2.2 ConsensusModel."""
    consensus: ConsensusModel
    reference: pd.Series
    reference_t90: np.ndarray
    harvest_stage: np.ndarray
    needs_maturation: bool
    initial_conditions: dict
    rules: dict = field(default_factory=lambda: dict(DEFAULT_UPDATE_RULES))

    @classmethod
    def from_consensus(cls, consensus, species, substrate, initial_conditions, rules=None):
        frame = consensus.selected_reference(species, substrate)
        reference, raw = frame.iloc[0], consensus.predict_reference(frame).iloc[0]
        pred = np.array([raw.t90_lower, raw.t90_typical, raw.t90_upper], float)
        harvest_typical = float(raw.pinning_typical + raw.development_typical)
        spread = float(consensus.settings["harvest_relative_spread_assumed"])
        return cls(consensus, reference, pred,
                   np.array([harvest_typical * (1 - spread), harvest_typical, harvest_typical * (1 + spread)]),
                   bool(reference.species == "Lentinula edodes" or reference.regime == "natural_logs"),
                   dict(initial_conditions), dict(DEFAULT_UPDATE_RULES if rules is None else rules))

    @property
    def species(self):
        return self.reference.species

    @property
    def post_90_days(self):
        return self.consensus.post_90_days

    # exposure -----------------------------------------------------------------------------------
    def condition_effects(self, conditions):
        settings = self.consensus.settings
        temp, basis = conditions.get("temperature_c"), conditions.get("temperature_basis", "unknown")
        if basis not in ("air", "substrate_core", "unknown"):
            raise ValueError("Unknown temperature basis.")
        if not isinstance(conditions.get("drying_exposed"), (bool, np.bool_)):
            raise ValueError("drying_exposed must be boolean.")
        band = settings["review_temperature_optima"].get(self.species, {}).get(basis)
        notes, factor, available = [], 1., True
        if band is None and basis == "substrate_core":
            band = [float(self.reference.incubation_temp_opt_C)] * 2
            notes.append("Reference optimum treated as core temperature: inherited demo basis assumption.")
        if temp is None:
            notes.append("Temperature missing: reference-condition scenario assumed; not a measured exposure.")
        else:
            temp = finite_number(temp, "temperature", -20, 60)
            if band is None:
                available = False
                notes.append("No comparable temperature recommendation/basis.")
            else:
                distance = max(float(band[0]) - temp, 0., temp - float(band[1]))
                if distance > settings["temperature_max_distance_degrees_assumed"] or temp >= 35.:
                    available = False
                    notes.append("Temperature outside provisional quantitative domain.")
                else:
                    relative = max(settings["temperature_min_relative_rate_assumed"],
                                   1 - settings["temperature_rate_loss_per_degree_assumed"] * distance)
                    factor = 1. / relative
        rh, delay = conditions.get("humidity_pct"), np.zeros(3)
        if rh is not None:
            rh = finite_number(rh, "ambient RH", 0, 100)
            if rh < settings["low_RH_threshold_assumed"] and conditions["drying_exposed"]:
                low, high = settings["low_RH_drying_exposure_delay_scenario"]
                delay = np.array([low, (low + high) / 2, high], float)
                notes.append("Conditional low-RH/drying-exposure scenario, not a measured substrate-moisture effect.")
            if rh < float(self.reference.incubation_RH_pct):
                notes.append("RH below reference recommendation.")
        else:
            notes.append("RH missing; no humidity penalty inferred.")
        return {"available": available, "factor": factor, "delay": delay, "optimum_band": band, "notes": notes}

    def effective_rate(self, conditions):
        effect = self.condition_effects(conditions)
        if not effect["available"]:
            return None
        return self.reference_t90 / (self.reference_t90 * effect["factor"] + effect["delay"])

    def effective_clock(self, day, history):
        day = finite_number(day, "elapsed day", 0)
        total = np.zeros(3)
        for i, record in enumerate(history):
            start = float(record["day"])
            end = min(day, history[i + 1]["day"] if i + 1 < len(history) else day)
            if end <= start:
                continue
            rate = self.effective_rate(record["conditions"])
            if rate is None:
                return None
            total += (end - start) * rate
        return total

    def crossing_calendar_day(self, target, quantile, history, future, current_day):
        target = finite_number(target, "effective crossing target", 0)
        used = 0.
        for i, record in enumerate(history):
            start = float(record["day"])
            end = min(current_day, history[i + 1]["day"] if i + 1 < len(history) else current_day)
            if end <= start:
                continue
            rate = self.effective_rate(record["conditions"])
            if rate is None:
                return None
            increment = (end - start) * rate[quantile]
            if target <= used + increment:
                return start + (target - used) / rate[quantile]
            used += increment
        rate = self.effective_rate(future)
        if rate is None:
            return None
        return current_day + max(0., target - used) / rate[quantile]

    # curve --------------------------------------------------------------------------------------
    def fit_progress(self, observations, history, current_day):
        rules, ref = self.rules, self.reference_t90
        base_slope = LOG9 / ((1 - rules["curve_midpoint_fraction"]) * ref[1])
        result = {"status": "baseline_only", "scale_scenarios": [1., 1., 1.], "t90_effective": float(ref[1]),
                  "slope": float(base_slope), "rmse": None, "notes": [], "profile": None}
        if not observations:
            return result
        times = []
        for obs in observations:
            tau = self.effective_clock(obs["day"], history)
            if tau is None:
                result.update(status="conditions_unavailable", scale_scenarios=None)
                result["notes"].append("Historical conditions cannot be mapped to an effective time.")
                return result
            times.append(tau[1])
        days = np.array([r["day"] for r in observations])
        c = np.array([r["coverage_fraction"] for r in observations])
        if len(c) >= 2 and np.any(np.diff(c) < -rules["large_decline_fraction"]):
            result.update(status="declining_coverage_review", scale_scenarios=None)
            result["notes"].append("Large coverage decline: review image masks/contamination and progression before extrapolating.")
            return result
        if len(c) >= 3 and days[-1] - days[0] >= rules["stall_span_days"] \
                and c[-1] - c[0] <= rules["stall_progress_fraction"] and c.max() < .90:
            result.update(status="stalled_coverage_review", scale_scenarios=None)
            result["notes"].append("Little net progress across dated observations: no automatic finite completion forecast.")
            return result
        scales = np.geomspace(rules["duration_scale_min"], rules["duration_scale_max"], 161)
        slopes = base_slope * np.geomspace(.35, 3., 61)
        theta = ref[1] * scales
        pred = logistic_coverage(np.array(times)[None, None, :], theta[:, None, None], slopes[None, :, None])
        tolerance = np.array([r["uncertainty_fraction"] for r in observations])
        residual = (pred - c[None, None, :]) / tolerance[None, None, :]
        # Huber-like loss reduces the influence of a single noisy mask without hiding residual diagnostics.
        abs_res = np.abs(residual)
        photo_loss = np.sum(np.where(abs_res <= 2, residual ** 2, 4 * abs_res - 4), axis=2)
        prior = (np.log(scales) / rules["t90_log_prior_scale"]) ** 2
        slope_prior = (np.log(slopes / base_slope) / rules["slope_log_prior_scale"]) ** 2
        loss = photo_loss + prior[:, None] + slope_prior[None, :]
        best = np.unravel_index(np.argmin(loss), loss.shape)
        profile = loss.min(axis=1)
        plausible = scales[profile <= loss[best] + rules["scenario_loss_tolerance"]]
        chosen = pred[best[0], best[1], :]
        rmse = float(np.sqrt(np.mean((chosen - c) ** 2)))
        result.update(status="coverage_updated",
                      scale_scenarios=[float(plausible.min()), float(scales[best[0]]), float(plausible.max())],
                      t90_effective=float(theta[best[0]]), slope=float(slopes[best[1]]), rmse=rmse,
                      profile={"scale": scales.tolist(), "loss": profile.tolist()})
        if best[0] in (0, len(scales) - 1) or best[1] in (0, len(slopes) - 1):
            result.update(status="fit_boundary_review", scale_scenarios=None)
            result["notes"].append("Best curve touches the provisional search boundary; numeric update withheld.")
        if rmse > rules["poor_fit_rmse_fraction"]:
            result.update(status="poor_curve_agreement_review", scale_scenarios=None)
            result["notes"].append("Curve disagrees materially with coverage observations; numeric update withheld.")
        if len(c) < 3:
            result["notes"].append("Fewer than three photos: update remains strongly dependent on the baseline/shape assumptions.")
        if current_day - days[-1] > rules["stale_photo_days"]:
            result["notes"].append("Latest photo is stale; current progress is unobserved.")
        return result

    def forecast_batch(self, observations, history, future, current_day, induction_day=None,
                       induction_confirmed=False, confirmed_t90_day=None):
        """The updated 90 % day, the +2-day planning marker and a conditional first harvest, all in days since
        inoculation as [lower, typical, upper] scenario values (None when not computable)."""
        post_90 = self.post_90_days
        current_day = finite_number(current_day, "current day", 0)
        if not isinstance(induction_confirmed, (bool, np.bool_)):
            raise ValueError("induction_confirmed must be boolean.")
        fit = self.fit_progress(observations, history, current_day)
        crossing, basis = observed_crossing(observations, confirmed_t90_day, current_day)
        notes, t90, baseline_calendar = list(fit["notes"]), None, None
        baseline_raw = [self.crossing_calendar_day(float(d), q, history, future, current_day)
                        for q, d in enumerate(self.reference_t90)]
        if all(v is not None for v in baseline_raw):
            baseline_calendar = sorted(float(v) for v in baseline_raw)
        if crossing is not None:
            # Historical observed crossing can remain known even when later conditions are out of domain.
            t90, status = crossing, "observed_crossing"
            if fit["status"] == "declining_coverage_review":
                t90, status = None, fit["status"]
                notes.append("Historical crossing does not establish continued readiness after a large decline.")
        elif fit["scale_scenarios"] is not None:
            effective_targets = self.reference_t90 * np.array(fit["scale_scenarios"])
            projected = [self.crossing_calendar_day(float(d), q, history, future, current_day)
                         for q, d in enumerate(effective_targets)]
            if all(v is not None for v in projected):
                lower_bound = max((r["day"] for r in observations if r["coverage_fraction"] < .9), default=0.)
                t90, status = sorted(max(float(v), lower_bound) for v in projected), fit["status"]
                basis = "baseline-anchored effective-time curve" if observations else "conditions-only consensus baseline"
            else:
                status = "conditions_unavailable"
                notes.append("Historical/future conditions outside quantitative rule support.")
        else:
            status = fit["status"]
        if basis is not None:
            notes.append(basis)
        if not any(r["origin"] == "recorded update" and r["day"] == 0 for r in history):
            notes.append("Unrecorded initial exposure uses exported 2.2 conditions as an assumption.")
        notes.append("Future exposure uses selected future conditions held constant; no future observations were used.")
        marker = None if t90 is None else [v + post_90 for v in t90]
        remaining = lambda values: None if values is None else [max(0., v - current_day) for v in values]
        if marker is not None and marker[1] < current_day and crossing is None:
            notes.append("Predicted marker is already in the past but crossing is unconfirmed; zero remaining is not evidence of readiness.")
        harvest = None
        if induction_day is not None:
            start = finite_number(induction_day, "induction day", 0)
            harvest = [start + float(v) for v in self.harvest_stage]
            notes.append("Harvest uses entered induction day and the separate 2.2 fruiting scenario.")
        elif marker is not None and (not self.needs_maturation or induction_confirmed):
            harvest = [a + float(b) for a, b in zip(marker, self.harvest_stage)]
            notes.append("Harvest assumes induction at the planning marker, followed by suitable fruiting conditions.")
        else:
            notes.append("Harvest unavailable until a valid induction planning date or actual induction day is supplied.")
        if self.needs_maturation:
            notes.append("Special maturation/log regime: review species-specific readiness; 90% + 2 days alone is insufficient.")
        return {"status": status, "t90_day_min_typ_max": t90, "crossing_basis": basis,
                "conditions_only_t90_day_min_typ_max": baseline_calendar,
                "planning_90_plus_2_day_min_typ_max": marker, "planning_days_remaining": remaining(marker),
                "first_harvest_day_min_typ_max": harvest, "first_harvest_days_remaining": remaining(harvest),
                "current_day": current_day, "accepted_coverage_count": len(observations),
                "curve_fit": fit, "notes": notes}
