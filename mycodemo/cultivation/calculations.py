"""Core grow maths: hydration, spawn, yield from biological efficiency, economics.

Ported from mycro-grow ``src/grow.js`` (https://github.com/SaschaHeyer/mycro-grow,
commit 7cb8de13b602035f9bcc7eecb4b5d421b9de76a6, MIT licence, (c) 2026 Mycro).
See THIRD_PARTY_NOTICES.md. This is the ORIGINAL behaviour, unchanged.

Naming: JavaScript camelCase became snake_case (``dryMass`` -> ``dry_mass``,
``yieldFromBE`` -> ``yield_from_be``, ``growPlan`` -> ``grow_plan``). Result
dictionaries use snake_case keys too (``costPerBlock`` -> ``cost_per_block``).

Unit policy (from the original): the functions are unit-agnostic but internally
consistent. Pass masses in one unit (kg or lb) and prices per that same unit.

Important property of the original: ``hydration`` expects the DRY mass (zero
water). It has no input for the water already in the material as weighed.
"""

from __future__ import annotations

import math
from typing import Any


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def _or0(v: Any) -> float:
    """JavaScript ``v || 0``: None/0/NaN become 0."""
    return 0 if v is None or v != v else v


def _js_round(x: float) -> int:
    """JavaScript ``Math.round`` (halves round up), not Python's banker's rounding."""
    return math.floor(x + 0.5)


def hydration(dry_mass: float, moisture_pct: float) -> dict:
    """Water needed to hydrate a dry substrate to a target moisture content.

    water = M * dry / (1 - M); hydrated = dry + water. Moisture is clamped to
    40-75 % (original field range).

    Returns ``{"water", "hydrated", "moisture"}`` with moisture as a fraction.
    """
    dry = max(0, dry_mass)
    m = _clamp(moisture_pct / 100, 0.40, 0.75)
    water = (m * dry) / (1 - m) if 1 - m > 0 else 0
    return {"water": water, "hydrated": dry + water, "moisture": m}


def spawn(hydrated_mass: float, spawn_rate_pct: float) -> dict:
    """Grain spawn to add, as a percent of the HYDRATED substrate mass.

    Returns ``{"spawn", "rate", "ratio"}``; ``ratio`` is the text "1 : N".
    Note: this basis (spawn / hydrated substrate) differs from MycoDemo's
    (spawn / (substrate + spawn)). Conversion is handled outside this module.
    """
    r = max(0, spawn_rate_pct) / 100
    spawn_mass = max(0, hydrated_mass) * r
    ratio = f"1 : {1 / r:.1f}" if r > 0 else "—"
    return {"spawn": spawn_mass, "rate": r, "ratio": ratio}


def yield_from_be(dry_mass: float, be_pct: float) -> float:
    """Expected fresh yield = dry substrate * (BE / 100).

    BE (biological efficiency) is fresh harvest weight as a percent of the dry
    substrate weight. A fixed assumption in the original, not a prediction.
    """
    return max(0, dry_mass) * (max(0, be_pct) / 100)


def economics(
    total_dry: float = 0,
    total_spawn: float = 0,
    total_yield: float = 0,
    blocks: float = 0,
    cost_sub: float = 0,
    cost_spawn: float = 0,
    extra_per_block: float = 0,
    sale_price: float = 0,
) -> dict:
    """Per-batch economics (all masses in one unit, prices per that unit).

    Returns ``total_cost``, ``cost_per_block``, ``revenue``, ``profit``,
    ``margin`` (percent) and ``cost_per_unit_yield`` (None if no yield).
    """
    n = max(0, _js_round(_or0(blocks)))
    c_sub = _or0(total_dry) * _or0(cost_sub)
    c_spawn = _or0(total_spawn) * _or0(cost_spawn)
    extra = _or0(extra_per_block) * n
    total_cost = c_sub + c_spawn + extra
    revenue = _or0(total_yield) * _or0(sale_price)
    profit = revenue - total_cost
    return {
        "total_cost": total_cost,
        "cost_per_block": total_cost / n if n > 0 else 0,
        "revenue": revenue,
        "profit": profit,
        "margin": (profit / revenue) * 100 if revenue > 0 else 0,
        "cost_per_unit_yield": total_cost / total_yield if _or0(total_yield) > 0 else None,
    }


def grow_plan(
    dry_per_block: float,
    blocks: float,
    moisture: float | None = 60,
    spawn_rate: float | None = 20,
    be: float | None = 85,
    cost_sub: float = 0,
    cost_spawn: float = 0,
    extra_per_block: float = 0,
    sale_price: float = 0,
) -> dict:
    """Full plan for a batch: hydration + spawn + yield + economics.

    Defaults match the original: 60 % moisture, 20 % spawn, 85 % BE. Passing
    ``None`` for moisture, spawn_rate or be uses the default (as JS ``??``).
    Returns ``{"per_block", "totals", "economics"}``.
    """
    n = max(0, _js_round(_or0(blocks)))
    dry = max(0, _or0(dry_per_block))
    h = hydration(dry, 60 if moisture is None else moisture)
    s = spawn(h["hydrated"], 20 if spawn_rate is None else spawn_rate)
    y = yield_from_be(dry, 85 if be is None else be)

    per_block = {
        "dry": dry,
        "water": h["water"],
        "hydrated": h["hydrated"],
        "spawn": s["spawn"],
        "spawn_ratio": s["ratio"],
        "yield": y,
    }
    totals = {
        "dry": dry * n,
        "water": h["water"] * n,
        "hydrated": h["hydrated"] * n,
        "spawn": s["spawn"] * n,
        "yield": y * n,
        "blocks": n,
    }
    econ = economics(
        total_dry=totals["dry"],
        total_spawn=totals["spawn"],
        total_yield=totals["yield"],
        blocks=n,
        cost_sub=cost_sub,
        cost_spawn=cost_spawn,
        extra_per_block=extra_per_block,
        sale_price=sale_price,
    )
    return {"per_block": per_block, "totals": totals, "economics": econ}
