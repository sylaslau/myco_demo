"""One coordinator for the grower pages: a batch record and run(batch), which calls every step in order.

The pages only display what run() returns. Nothing here touches streamlit or the disk (the batch lives in
st.session_state and the JSON backup).

Every step returns a Step: a value, a source tag and a short note. When a step is missing or unsure it falls
back to the step below and says so in the note; it never returns nothing.
"""

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from mycodemo.cultivation import FRUITING_CONDITIONS, SPECIES_BE, substrate_recipe
from mycodemo.cultivation.corrected import spawn_corrected
from mycodemo.growth.forecast import forecast, forecast_warnings

# --- constants ------------------------------------------------------------------------------
TYPICAL_OYSTER_DAYS = (10, 14)        # usemycro.com guidance, the same range the Grow Log uses
MATERIAL_SPECIES = "Blue Oyster"      # the species in our material data (Pleurotus ostreatus)
PRODUCTS = ["Foam", "Acoustic board", "Game pieces"]
# environment: label -> (containers, dry substrate per container in kg); sizes are our pretend defaults
ENVIRONMENTS = {
    "Small plastic container (about 1 L)": (1, 0.2),
    "Grow bag (about 3 kg)": (1, 0.5),
    "Bucket or tub (about 10 L)": (1, 1.0),
    "Shelf or grow tent": (10, 1.0),
    "Room": (100, 1.0),
}
# common name in FRUITING_CONDITIONS -> key in SPECIES_BE (species without a typical BE are not listed)
BE_KEY = {"Blue Oyster": "Blue / Pearl Oyster", "Pearl Oyster": "Blue / Pearl Oyster",
          "Pink Oyster": "Pink / Yellow Oyster", "Golden / Yellow Oyster": "Pink / Yellow Oyster",
          "King Oyster": "King Oyster (Trumpet)", "Lion's Mane": "Lion's Mane", "Shiitake": "Shiitake",
          "Chestnut": "Chestnut", "Pioppino": "Pioppino", "Nameko": "Nameko", "Reishi": "Reishi"}
# the growing space the user describes once (the Grow Log's Environment box); not the recipe scale above
ENVIRONMENT_SIZES = ["Small plastic box", "Tub", "Tent", "Shelf", "Room"]
MEASURED_BY = ["Thermometer", "Sensor device", "Guess"]
# stage that ends colonisation -> where "Next stage" goes, per goal (Food: Fruiting, Materials: Post-treatment)
COLONISING_STAGES = {"Food": ("Inoculated", "Colonising"), "Materials": ("Colonisation",)}
AFTER_COLONISATION = {"Food": "Fruiting", "Materials": "Post-treatment"}
BASELINE, SPAWN_PCT, MOISTURE = "masters", 10, 60   # baseline recipe (50/50 hardwood sawdust + soy hull, 1 % gypsum)
HUMIDITY_RANGE = (80.0, 95.0)         # % RH wanted while colonising; out of range is only ever a Warning
CONSENSUS_CSV = Path.home() / "Desktop/MycoMaterials/germini/colonisation_consensus.csv"   # outside the repo
# Temperature rule (our assumption, see temperature_factor): the typical days hold at the optimum, the growth
# rate falls in straight lines to zero at the biological limits. The 10-35 °C limits are the "biological range"
# stated for Pleurotus in the consensus table; the optimum is the median of the table's optima.
BIOLOGICAL_RANGE = {"Pleurotus": (10.0, 35.0)}
FALLBACK_OPTIMUM = {"Pleurotus": 25.0}   # used when the consensus table is not on this machine
RATE_FLOOR = 0.15                     # slowest relative growth rate we allow, so days never run away
MAX_NUDGE = 1.3                       # 1-2 photos move the estimate by at most 30 %
SOURCES = ("typical range", "our model", "your photos", "preview")
MIN_PHOTOS = 3                        # the Colonisation forecast takes over from here
STAGE_TARGET = 0.90                   # "colonised" = 90 %


# --- batch record ---------------------------------------------------------------------------
@dataclass
class Photo:
    day: int
    file: str
    coverage: float                   # 0-1
    method: str = "HSV threshold"
    quality: str = ""                 # '' = passed, else the reason(s)


@dataclass
class Condition:
    date: date
    temp_c: float | None = None
    rh_pct: float | None = None


@dataclass
class StageEntry:
    stage: str
    date: date
    note: str = ""


@dataclass
class Environment:
    """The growing space, described once. Kept in st.session_state and the JSON backup."""
    name: str
    size: str = "Tub"
    temp_c: float = 21.0              # ambient temperature
    rh_pct: float | None = None       # optional humidity
    measured_by: str = "Guess"

    def __post_init__(self):
        if self.size not in ENVIRONMENT_SIZES:
            raise ValueError(f"unknown environment size {self.size!r}")
        if self.measured_by not in MEASURED_BY:
            raise ValueError(f"unknown 'measured by' {self.measured_by!r}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def next_stage(goal, latest_stage):
    """Where "Next stage" goes from the batch's latest logged stage, or None when it is not time yet.

    Only the step after colonisation is defined: Food goes to Fruiting, Materials to Post-treatment (drying/baking).
    """
    goal = goal if goal in AFTER_COLONISATION else "Food"
    return AFTER_COLONISATION[goal] if latest_stage in COLONISING_STAGES[goal] else None


@dataclass
class Batch:
    """Everything the pages know about one grow. Kept in st.session_state, never written to disk."""
    goal: str | None = None           # 'Food' or 'Materials'
    choice: str | None = None         # species name (Food; 'Custom' = own mushroom) or product (Materials)
    custom_name: str = ""
    environment: str | None = None    # recipe scale (page 3), see ENVIRONMENTS; not the Environment record
    env_name: str | None = None       # name of the Environment this batch grows in (Grow Log)
    temp_c: float | None = None
    recipe: dict | None = None        # {'preset', 'moisture', 'spawn_pct'}; None = the baseline
    inoc_date: date = field(default_factory=date.today)
    photos: list[Photo] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)
    stages: list[StageEntry] = field(default_factory=list)
    fruiting_photos: list[str] = field(default_factory=list)   # for Harvest readiness (Food)

    def to_dict(self):
        """Plain dict for the JSON backup (dates as ISO text)."""
        d = asdict(self)
        d["inoc_date"] = self.inoc_date.isoformat()
        d["conditions"] = [{**c, "date": c["date"].isoformat()} for c in d["conditions"]]
        d["stages"] = [{**s, "date": s["date"].isoformat()} for s in d["stages"]]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["inoc_date"] = date.fromisoformat(d["inoc_date"])
        d["photos"] = [Photo(**p) for p in d.get("photos", [])]
        d["conditions"] = [Condition(**{**c, "date": date.fromisoformat(c["date"])}) for c in d.get("conditions", [])]
        d["stages"] = [StageEntry(**{**s, "date": date.fromisoformat(s["date"])}) for s in d.get("stages", [])]
        return cls(**d)


@dataclass
class Step:
    """One step's answer. source is one of SOURCES; 'preview' means pretend and inactive."""
    name: str
    value: object
    source: str
    note: str = ""

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}")

    @property
    def is_preview(self):
        return self.source == "preview"


@dataclass
class Result:
    recipe: Step
    colonisation_estimate: Step           # Colonisation time estimate
    colonisation_forecast: Step           # Colonisation forecast (or the estimate it fell back to)
    dates: Step
    harvest_readiness: Step | None        # Food only
    warnings: Step                        # value = list of messages
    mould_check: Step
    outcome: Step                         # yield estimate (Food) or Material strength (Materials)


# --- species ----------------------------------------------------------------------------------
def species_row(batch):
    """The FRUITING_CONDITIONS entry for the chosen species (materials use MATERIAL_SPECIES); None if custom."""
    name = MATERIAL_SPECIES if batch.goal == "Materials" else batch.choice
    return next((s for s in FRUITING_CONDITIONS if s["name"] == name), None)


def choice_label(batch):
    if batch.goal == "Food" and batch.choice == "Custom":
        return batch.custom_name or "your own mushroom"
    return batch.choice or ""


def colonisation_days(row):
    """(min, max) days to colonise at the best temperature, or None when we have no range for that species.
    Only the oyster (Pleurotus) range is known."""
    if row is not None and row["latin"].startswith("Pleurotus"):
        return TYPICAL_OYSTER_DAYS
    return None


def days_to_fruit(row):
    """(min, max) days from the species' ``timeToFruit`` text, e.g. '10–14 days' -> (10, 14)."""
    if row is None:
        return None
    nums = [int(n) for n in re.findall(r"\d+", row["timeToFruit"])[:2]]
    return (nums[0], nums[-1]) if nums else None


def be_percent(row):
    key = BE_KEY.get(row["name"]) if row else None
    return SPECIES_BE.get(key)


# --- temperature rule (parsing moved here from notebook 2.1; the notebook itself is unchanged) -----
_NUM = r"(\d+(?:\.\d+)?)"
_RANGE = re.compile(_NUM + r"\s*-\s*" + _NUM)
_SINGLE = re.compile(_NUM)


def mid_of(text):
    """Middle of 'a-b', or the value itself for a single number; NaN if no number."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    m = _RANGE.search(text)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = _SINGLE.search(text)
    return float(m.group(1)) if m else np.nan


def parse_temp(text):
    """Return (temp_min, temp_opt, temp_max) in °C from the Incubation_Temp_C text of the consensus table."""
    nan3 = (np.nan, np.nan, np.nan)
    if not isinstance(text, str):
        return nan3
    opt = np.nan   # stated optimum, looked for in the full text (it can sit inside brackets)
    m = re.search(r"optimum\s+" + _NUM + r"(?:\s*-\s*" + _NUM + r")?\s*°C", text)
    if m:
        a = float(m.group(1))
        b = float(m.group(2)) if m.group(2) else a
        opt = (a + b) / 2
    else:
        m = re.search(_NUM + r"\s*°C\s*ideal", text)
        if m:
            opt = float(m.group(1))
    s = re.sub(r"\([^)]*\)", "", text)                              # drop all brackets
    s = re.sub(_NUM + r"(\s*-\s*" + _NUM + r")?\s*°F", "", s)        # drop F values
    clauses = [c.strip() for c in s.split(";")]
    stop = re.compile(r"injury|damage|cease|threshold|stop", re.IGNORECASE)
    clauses = [c for c in clauses if c and not stop.search(c) and not c.lower().startswith("optimum")]
    if not clauses:
        return nan3
    main = next((c for c in clauses if re.match(r"(Substrate|Core)\b", c, re.IGNORECASE)), clauses[0])
    lo = hi = mid = mid_of(main)
    m = _RANGE.search(main)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
    rng = next((c for c in clauses if c.lower().startswith("range")), None)
    if rng and _RANGE.search(rng):
        m = _RANGE.search(rng)
        lo, hi = float(m.group(1)), float(m.group(2))
    return lo, (opt if not np.isnan(opt) else mid), hi


@cache
def genus_optimum(genus="Pleurotus", path=CONSENSUS_CSV):
    """(optimum °C, where it came from): median of the table's optima for that genus, else the built-in value."""
    try:
        raw = pd.read_csv(path)
        opts = [parse_temp(t)[1] for sp, t in zip(raw["Species"], raw["Incubation_Temp_C"])
                if str(sp).startswith(genus)]
        opts = [o for o in opts if not np.isnan(o)]
        if opts:
            return float(np.median(opts)), "consensus table"
    except (OSError, KeyError, ValueError):
        pass
    return FALLBACK_OPTIMUM[genus], "built-in value (consensus table not found)"


def temperature_factor(temp_c, genus="Pleurotus"):
    """Relative growth rate at temp_c: 1 at the optimum, falling in straight lines to 0 at the biological limits.
    Returns (factor, inside); outside the limits the factor is RATE_FLOOR and inside is False."""
    lo, hi = BIOLOGICAL_RANGE[genus]
    opt = genus_optimum(genus)[0]
    if temp_c <= lo or temp_c >= hi:
        return RATE_FLOOR, False
    rate = (temp_c - lo) / (opt - lo) if temp_c < opt else (hi - temp_c) / (hi - opt)
    return max(min(rate, 1.0), RATE_FLOOR), True


# --- typical colonisation curve (the shape drawn in notebook 2.1: 50 % at 0.6 of the time to the target) ---
_K = np.log(9) / 0.4          # 90 % at relative time 1


def progress_of(cover):
    """Relative time u (1 = the target day) at which the typical curve shows this coverage (0-1)."""
    c = float(np.clip(cover, 0.02, 0.98))
    return 0.6 + np.log(c / (1 - c)) / _K


# --- steps ------------------------------------------------------------------------------------
def _plan_size(batch):
    return ENVIRONMENTS.get(batch.environment, (1, 0.5))


def recipe_step(batch):
    n, kg = _plan_size(batch)
    r = batch.recipe or {}
    preset, moisture, spawn_pct = r.get("preset", BASELINE), r.get("moisture", MOISTURE), r.get("spawn_pct", SPAWN_PCT)
    rec = substrate_recipe(kg, n, preset, moisture)
    spawn_kg = spawn_corrected(rec["per_block"]["hydrated"], spawn_pct, "of_total_mix")["spawn"]
    note = "Baseline recipe, not yet tuned to your product or environment."
    if batch.environment not in ENVIRONMENTS:
        note += " No environment chosen yet, so one 0.5 kg container is assumed."
    if batch.goal == "Food" and species_row(batch) is None:
        note += " For your own mushroom the baseline oyster recipe is used; it may not suit it."
    return Step("Recipe", {"rec": rec, "spawn_kg": spawn_kg, "n": n, "moisture": moisture, "spawn_pct": spawn_pct},
                "typical range", note)


def colonisation_estimate(batch, temp_c=None):
    """Colonisation time estimate: the typical range, adjusted for temperature (planned one unless given)."""
    row = species_row(batch)
    base = colonisation_days(row)
    if base is None:
        return Step("Colonisation time estimate", None, "preview",
                    "We have no colonisation range for this mushroom yet, so no estimate is shown.")
    temp_c = batch.temp_c if temp_c is None else temp_c
    if temp_c is None:
        return Step("Colonisation time estimate", {"days": base, "factor": 1.0, "temp_c": None}, "typical range",
                    "Typical oyster range (usemycro.com guidance); no temperature chosen yet.")
    factor, inside = temperature_factor(temp_c)
    days = (base[0] / factor, base[1] / factor)
    opt, where = genus_optimum()
    note = (f"Typical oyster range {base[0]}–{base[1]} days (usemycro.com guidance), assumed to hold at the "
            f"optimum ({opt:g} °C, {where}). At {temp_c:g} °C the growth rate is {factor:.0%} of that, so "
            "days = typical ÷ rate. A rough rule of ours; the sources give ranges, not days at each temperature.")
    if not inside:
        note += f" {temp_c:g} °C is outside the biological limits ({BIOLOGICAL_RANGE['Pleurotus'][0]:g}–" \
                f"{BIOLOGICAL_RANGE['Pleurotus'][1]:g} °C): the slowest rate is used and growth may stall."
    return Step("Colonisation time estimate", {"days": days, "factor": factor, "temp_c": temp_c}, "typical range", note)


def _photo_series(batch):
    ph = sorted((p for p in batch.photos if p.day >= 0), key=lambda p: p.day)   # a photo before inoculation is a date slip
    return np.array([p.day for p in ph], float), np.array([p.coverage for p in ph], float)


def colonisation_forecast(batch, est):
    """Colonisation forecast: our S-curve fit from 3 photos; 1-2 photos nudge the estimate; none = the estimate.
    value: {'day90', 'day90_range', 'method', ...} (days since inoculation)."""
    t, c = _photo_series(batch)
    n = len(t)
    name = "Colonisation forecast"
    if n >= MIN_PHOTOS:
        res = forecast(t, c)
        if res["params"] is not None and not np.isnan(res["day90"]):
            d90 = float(res["day90"])
            return Step(name, {"day90": d90, "day95": float(res["day95"]), "day90_range": (d90, d90),
                               "method": "curve fit", "n_photos": n}, "your photos",
                        f"S-curve fitted to your {n} photos.")
        why = "the curve could not be fitted" if res["params"] is None else "90% is not reached within 60 days"
        fallback = f"With {n} photos {why}, so the step below is shown."
    elif n > 0:
        fallback = None
    else:
        fallback = "No photos yet, so this is the Colonisation time estimate."
    if est.value is None:
        return Step(name, None, "preview", (fallback or "") + " " + est.note)
    lo, hi = est.value["days"]
    mid = (lo + hi) / 2
    if n == 0 or fallback:
        return Step(name, {"day90": mid, "day90_range": (lo, hi), "method": "estimate", "n_photos": n},
                    est.source, fallback + (" " + est.note if n else ""))
    t_last, c_last = t[-1], c[-1]
    if t_last <= 0:
        return Step(name, {"day90": mid, "day90_range": (lo, hi), "method": "estimate", "n_photos": n},
                    est.source, "The photo was taken on day 0, which tells us nothing about pace yet.")
    speed = float(np.clip(progress_of(c_last) * mid / t_last, 1 / MAX_NUDGE, MAX_NUDGE))
    pace = "ahead of" if speed > 1.02 else "behind" if speed < 0.98 else "in line with"
    return Step(name, {"day90": mid / speed, "day90_range": (lo / speed, hi / speed), "method": "nudged",
                       "speed": speed, "n_photos": n}, "your photos",
                f"{n} photo(s): {100 * c_last:.0f}% on day {t_last:.0f} is {pace} a typical curve, so the estimate "
                f"is nudged (at most {MAX_NUDGE - 1:.0%}). {MIN_PHOTOS} photos are needed for the S-curve forecast.")


def dates_step(batch, fc):
    """Colonised (90%), then the harvest window (Food) or the start of post-treatment (Materials)."""
    if fc.value is None:
        return Step("Dates", [], "preview", "No estimate yet for this mushroom, so there are no dates.")
    start, row = batch.inoc_date, species_row(batch)
    lo, hi = (round(d) for d in fc.value["day90_range"])
    events = [("Inoculation", start, start),
              ("Colonised (90%)", start + timedelta(days=lo), start + timedelta(days=hi))]
    note = f"Dates follow the {fc.name} ({fc.source})."
    if batch.goal == "Food":
        fruit = days_to_fruit(row)
        if fruit:
            events.append(("Harvest window", start + timedelta(days=lo + fruit[0]),
                           start + timedelta(days=hi + fruit[1])))
        else:
            note += " No fruiting time is known for this mushroom, so there is no harvest window."
    else:
        events.append(("Post-treatment can start", start + timedelta(days=hi), start + timedelta(days=hi)))
    return Step("Dates", events, fc.source, note)


def harvest_readiness(photos):
    """Harvest readiness from fruiting photos. Fixed shape, ready for the model of notebook 3.2 (still in
    training): input = list of photo paths; output = {'label': 'young' | 'ready' | None, 'score': 0-1 | None,
    'n_photos': int}. Until the model is ready the label and score stay None."""
    return {"label": None, "score": None, "n_photos": len(photos)}


def harvest_readiness_step(batch):
    if batch.goal != "Food":
        return None
    return Step("Harvest readiness", harvest_readiness(batch.fruiting_photos), "preview",
                "Planned: fruiting photos tell you young or ready to pick. The model is still in training. "
                "Until then use the harvest window from the dates.")


def mould_check(photos):
    """Mould check (part of Warnings): an empty step. Input = list of photo paths; output = {'mould': bool | None,
    'n_photos': int}. None until the model of notebook 2.6 is ready."""
    return {"mould": None, "n_photos": len(photos)}


def warnings_step(batch, est, fc):
    """Recipe warnings, forecast warnings, and temperature or humidity out of range (humidity never moves dates)."""
    msgs = list(recipe_step(batch).value["rec"]["warnings"])
    t, c = _photo_series(batch)
    if len(t):
        res = forecast(t, c) if len(t) >= 2 else {"params": None, "day90": np.nan, "day95": np.nan}
        msgs += forecast_warnings(t, c, res) if len(t) >= 2 else \
            [f"Only {len(t)} photo(s): at least {MIN_PHOTOS} are needed for a forecast you can trust."]
    row = species_row(batch)
    if row is not None:
        lo, hi = row["colonizeC"]
        temps = ([("planned", batch.temp_c)] if batch.temp_c is not None else []) + \
                [(k.date.isoformat(), k.temp_c) for k in batch.conditions if k.temp_c is not None]
        for when, tc in temps:
            if not lo <= tc <= hi:
                msgs.append(f"Temperature {tc:g} °C ({when}) is outside the colonisation range of {row['name']} "
                            f"({lo}–{hi} °C): colonisation will be slower, or may stall.")
    for k in batch.conditions:
        if k.rh_pct is not None and not HUMIDITY_RANGE[0] <= k.rh_pct <= HUMIDITY_RANGE[1]:
            msgs.append(f"Humidity {k.rh_pct:g}% ({k.date.isoformat()}) is outside {HUMIDITY_RANGE[0]:g}–"
                        f"{HUMIDITY_RANGE[1]:g}%. This does not change the dates.")
    return Step("Warnings", msgs, "your photos" if len(t) else "typical range",
                "Recipe, forecast, temperature and humidity checks.")


def mould_check_step(batch):
    return Step("Mould check", mould_check([p.file for p in batch.photos]), "preview",
                "Planned: each photo is checked for mould and you get a warning. Not active yet.")


@lru_cache(maxsize=1)
def _strength_model():
    from mycodemo.property_data import OUTPUT_PATH
    from mycodemo.property_model import TwoStepModel, training_rows
    return TwoStepModel().fit(training_rows(pd.read_csv(OUTPUT_PATH)))


def outcome_step(batch):
    """Yield estimate (Food) or Material strength (Materials)."""
    if batch.goal != "Materials":
        row = species_row(batch)
        be = be_percent(row)
        if be is None:
            return Step("Yield estimate", None, "preview", "No typical biological efficiency (BE) is known for this mushroom.")
        dry = recipe_step(batch).value["rec"]["totals"]["total_dry"]
        return Step("Yield estimate", {"kg": dry * be / 100, "be": be, "dry_kg": dry}, "typical range",
                    f"Estimate, not a prediction: an assumed BE of {be} % of the dry substrate ({dry:.1f} kg dry). "
                    "No dataset yet links growth to harvest weight.")
    try:
        frame = pd.DataFrame({"species_name": ["Pleurotus ostreatus"], "substrate_group": ["hardwood"]})
        model = _strength_model()
        stress, dens = float(model.predict_stress(frame)[0]), float(model.predict_density(frame)[0])
    except (FileNotFoundError, ValueError, KeyError):
        return Step("Material strength", None, "preview", "The material model could not be loaded.")
    return Step("Material strength", {"stress_mpa": stress, "density": dens}, "our model",
                f"Estimate for hardwood sawdust and P. ostreatus. The model has no input for the product, so it is "
                f"the same for {', '.join(PRODUCTS).lower()}; it is rough (tested by leaving one lab out).")


def days_to_next_stage(batch, cover, temp_c):
    """Per-photo card: days left until 90 % colonised at temp_c, from this photo's coverage (0-1).
    Always a rough range: value (low, high) days, or None without an estimate."""
    est = colonisation_estimate(batch, temp_c)
    name = "Days to 90% colonised (rough)"
    if est.value is None:
        return Step(name, None, "preview", est.note)
    if cover >= STAGE_TARGET:
        return Step(name, (0.0, 0.0), "your photos", "This photo is already at 90% or more.")
    left = 1 - progress_of(cover)
    lo, hi = est.value["days"]
    return Step(name, (max(0.0, left * lo), max(0.0, left * hi)), "your photos",
                f"Rough: where this photo sits on a typical curve, at {temp_c:g} °C. Real growth varies.")


# --- the coordinator ---------------------------------------------------------------------------
def run(batch):
    """Run every step in order and return one Result. Call again after every new photo or logged condition."""
    recipe = recipe_step(batch)
    est = colonisation_estimate(batch)
    fc = colonisation_forecast(batch, est)
    dates = dates_step(batch, fc)
    ready = harvest_readiness_step(batch)
    warns = warnings_step(batch, est, fc)
    mould = mould_check_step(batch)
    outcome = outcome_step(batch)
    return Result(recipe, est, fc, dates, ready, warns, mould, outcome)
