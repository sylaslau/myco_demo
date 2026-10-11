"""Corrected grow maths: the "corrected" column of the comparison.

Same idea as ``calculations.py`` (the untouched mycro-grow port), with three
intentional changes. Each one is listed in ``docs/docs/cultivation-port.md``.

1. Starting moisture. Material as weighed (pellets, straw, sawdust) already holds
   some water, often about 10 %. The original treats the weighed mass as bone dry,
   so it adds too much water. Here:
       dry matter      = as_weighed * (1 - m0)
       water already   = as_weighed * m0
       water to add    = M * dry / (1 - M) - water already
   where m0 is the starting moisture and M the target, both as fractions.
   Check: 5 lb at 10 % to 60 % gives 6.25 water and 11.25 hydrated, as on the
   hosted calculator (usemycro.com/calculator.html).

2. Yield from true dry matter. BE is fresh yield as a percent of DRY substrate,
   so yield = dry matter * BE, not as-weighed mass * BE.

3. Spawn basis. The original gives spawn as a percent of the hydrated substrate.
   MycoDemo uses the percent of the whole mix: spawn / (hydrated + spawn).
   The user picks a basis; both percentages are returned so they can be compared.
   20 % of hydrated substrate = 16.7 % of the whole mix.

Units: as in the original, use one mass unit throughout (kg or lb) and prices
per that unit. Substrate cost is charged on the as-weighed mass (what you buy).
"""

from __future__ import annotations

from .calculations import _clamp, _js_round, _or0, economics

SPAWN_BASES = ("of_hydrated", "of_total_mix")


def dry_matter(as_weighed: float, start_moisture_pct: float = 0) -> dict:
    """Split an as-weighed mass into dry matter and the water it already holds.

    ``start_moisture_pct`` is limited to 0-95 %. Returns ``{"dry", "water_in"}``.
    """
    w = max(0, _or0(as_weighed))
    m0 = _clamp(_or0(start_moisture_pct) / 100, 0, 0.95)
    return {"dry": w * (1 - m0), "water_in": w * m0}


def hydration_corrected(
    as_weighed: float, target_moisture_pct: float, start_moisture_pct: float = 0
) -> dict:
    """Water to add to as-weighed material to reach the target moisture.

    The target is limited to 40-75 % as in the original. If the material is
    already wetter than the target, no water is added and a warning is returned.
    Returns ``{"dry", "water_in", "water", "hydrated", "moisture", "warnings"}``;
    ``water`` is the water to ADD and ``moisture`` the final moisture (fraction).
    """
    d = dry_matter(as_weighed, start_moisture_pct)
    target = _clamp(_or0(target_moisture_pct) / 100, 0.40, 0.75)
    water_total = target * d["dry"] / (1 - target)
    water_add = water_total - d["water_in"]
    warnings = []
    if water_add < 0:
        warnings.append(
            "The material is already wetter than the target moisture; no water is added."
        )
        water_add = 0.0
    hydrated = d["dry"] + d["water_in"] + water_add
    moisture = (d["water_in"] + water_add) / hydrated if hydrated > 0 else 0
    return {
        "dry": d["dry"],
        "water_in": d["water_in"],
        "water": water_add,
        "hydrated": hydrated,
        "moisture": moisture,
        "warnings": warnings,
    }


def spawn_corrected(hydrated_mass: float, spawn_pct: float, basis: str = "of_total_mix") -> dict:
    """Grain spawn to add, with the percent given on either basis.

    ``basis="of_hydrated"``: spawn = hydrated * p (the original's definition).
    ``basis="of_total_mix"``: spawn = hydrated * p / (1 - p), so that
    spawn / (hydrated + spawn) = p (MycoDemo's definition).
    Returns ``{"spawn", "pct_of_hydrated", "pct_of_total_mix", "ratio"}``.
    """
    if basis not in SPAWN_BASES:
        raise ValueError(f"basis must be one of {SPAWN_BASES}, got {basis!r}")
    h = max(0, _or0(hydrated_mass))
    p = max(0, _or0(spawn_pct)) / 100
    if basis == "of_total_mix":
        if p >= 1:
            raise ValueError("spawn_pct must be below 100 % of the total mix")
        s = h * p / (1 - p)
    else:
        s = h * p
    pct_h = 100 * s / h if h > 0 else 0
    pct_t = 100 * s / (h + s) if h + s > 0 else 0
    ratio = f"1 : {h / s:.1f}" if s > 0 else "—"
    return {"spawn": s, "pct_of_hydrated": pct_h, "pct_of_total_mix": pct_t, "ratio": ratio}


def grow_plan_corrected(
    as_weighed_per_block: float,
    blocks: float,
    start_moisture: float = 10,
    moisture: float = 60,
    spawn_pct: float = 16.7,
    spawn_basis: str = "of_total_mix",
    be: float = 85,
    cost_sub: float = 0,
    cost_spawn: float = 0,
    extra_per_block: float = 0,
    sale_price: float = 0,
) -> dict:
    """Full corrected plan for a batch.

    Defaults: 10 % starting moisture (typical for bagged pellets, per the hosted
    calculator), 60 % target, 16.7 % spawn of the whole mix (= the original's
    20 % of hydrated substrate), 85 % BE (the original's oyster default; still a
    fixed assumption, not a prediction).
    Returns ``{"per_block", "totals", "economics", "warnings"}``.
    """
    n = max(0, _js_round(_or0(blocks)))
    w = max(0, _or0(as_weighed_per_block))
    h = hydration_corrected(w, moisture, start_moisture)
    s = spawn_corrected(h["hydrated"], spawn_pct, spawn_basis)
    y = h["dry"] * max(0, _or0(be)) / 100

    per_block = {
        "as_weighed": w,
        "dry": h["dry"],
        "water_in": h["water_in"],
        "water": h["water"],
        "hydrated": h["hydrated"],
        "spawn": s["spawn"],
        "spawn_pct_of_hydrated": s["pct_of_hydrated"],
        "spawn_pct_of_total_mix": s["pct_of_total_mix"],
        "spawn_ratio": s["ratio"],
        "yield": y,
    }
    totals = {
        "as_weighed": w * n,
        "dry": h["dry"] * n,
        "water": h["water"] * n,
        "hydrated": h["hydrated"] * n,
        "spawn": s["spawn"] * n,
        "yield": y * n,
        "blocks": n,
    }
    econ = economics(
        total_dry=totals["as_weighed"],  # pay for what you weigh
        total_spawn=totals["spawn"],
        total_yield=totals["yield"],
        blocks=n,
        cost_sub=cost_sub,
        cost_spawn=cost_spawn,
        extra_per_block=extra_per_block,
        sale_price=sale_price,
    )
    return {"per_block": per_block, "totals": totals, "economics": econ, "warnings": h["warnings"]}
