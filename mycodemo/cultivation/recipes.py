"""Substrate recipe maths: sawdust / soy hull / bran / gypsum split and water.

Ported from mycro-grow ``src/substrate.js`` (https://github.com/SaschaHeyer/mycro-grow,
commit 7cb8de13b602035f9bcc7eecb4b5d421b9de76a6, MIT licence, (c) 2026 Mycro).
See THIRD_PARTY_NOTICES.md. This is the ORIGINAL behaviour, unchanged.

A recipe splits a dry "base" into sawdust + soy hull by weight percent, adds bran
and gypsum as a percent of the base, then adds water to a target moisture
(clamped to 40-75 %). The ``straw`` preset treats the whole base as straw pellets.
"""

from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    # 50/50 hardwood sawdust + soy hull pellets, 1 % gypsum (the original's "workhorse").
    "masters": {"pSaw": 50, "pSoy": 50, "pBran": 0, "pGyp": 1},
    # 100 % sawdust, supplemented with 20 % bran, 1 % gypsum.
    "suppsaw": {"pSaw": 100, "pSoy": 0, "pBran": 20, "pGyp": 1},
    # 100 % soy hull pellets, 1 % gypsum.
    "soyhull": {"pSaw": 0, "pSoy": 100, "pBran": 0, "pGyp": 1},
    # 100 % straw pellets, no supplementation.
    "straw": {"pSaw": 0, "pSoy": 0, "pBran": 0, "pGyp": 0, "straw": True},
}
# Preset keys keep the original camelCase (pSaw, pSoy, pBran, pGyp) for easy comparison.


def _or0(v: Any) -> float:
    """JavaScript ``v || 0``."""
    return 0 if v is None or v != v else v


def _js_round(x: float) -> int:
    import math

    return math.floor(x + 0.5)


def _num(n: float) -> str:
    """Format a number like JavaScript's template literal (120.0 -> '120')."""
    return str(int(n)) if float(n).is_integer() else repr(float(n))


def substrate_recipe(
    base_per_block: float,
    blocks: float,
    preset: str | None = None,
    moisture: float | None = 60,
    p_saw: float | None = None,
    p_soy: float | None = None,
    p_bran: float | None = None,
    p_gyp: float | None = None,
    straw: bool = False,
) -> dict:
    """Compute a recipe for a batch of blocks.

    ``preset`` is one of ``PRESETS``; if given, the p_* arguments are ignored.
    Returns ``{"per_block", "totals", "ingredients", "warnings"}`` (snake_case
    keys; ingredient names as in the original).
    """
    p = None
    if preset:
        p = PRESETS.get(preset)
        if p is None:
            raise ValueError(f'Unknown preset "{preset}". Use one of: {", ".join(PRESETS)}')

    n = max(0, _js_round(_or0(blocks)))
    base = max(0, _or0(base_per_block))
    is_straw = bool(p.get("straw") if p else straw)
    p_saw_v = max(0, _or0(p["pSaw"] if p else p_saw))
    p_soy_v = max(0, _or0(p["pSoy"] if p else p_soy))
    p_bran_v = max(0, _or0(p["pBran"] if p else p_bran))
    p_gyp_v = max(0, _or0(p["pGyp"] if p else p_gyp))
    m = min(0.75, max(0.40, (60 if moisture is None else moisture) / 100))

    warnings: list[str] = []
    total_pct = p_saw_v + p_soy_v
    if not is_straw and total_pct > 0 and abs(total_pct - 100) > 0.5:
        warnings.append(
            f"sawdust % + soy hull % = {_num(total_pct)}% (not 100%). The base is split by "
            "these weights as entered — adjust so they total 100% for a standard recipe."
        )

    saw = base * (p_saw_v / 100)
    soy = base * (p_soy_v / 100)
    straw_pb = base if is_straw else 0
    base_dry = straw_pb if is_straw else saw + soy
    bran = base_dry * (p_bran_v / 100)
    gyp = base_dry * (p_gyp_v / 100)
    total_dry = base_dry + bran + gyp
    water = (m * total_dry) / (1 - m) if 1 - m > 0 else 0
    hydrated = total_dry + water

    ingredients: list[dict] = []

    def add(name: str, per_block: float) -> None:
        if per_block > 0:
            ingredients.append({"name": name, "per_block": per_block, "total": per_block * n})

    if is_straw:
        add("Straw pellets", straw_pb)
    else:
        add("Hardwood sawdust", saw)
        add("Soy hull pellets", soy)
    add("Bran (wheat/oat)", bran)
    add("Gypsum", gyp)

    return {
        "per_block": {"total_dry": total_dry, "water": water, "hydrated": hydrated},
        "totals": {
            "total_dry": total_dry * n,
            "water": water * n,
            "hydrated": hydrated * n,
            "blocks": n,
        },
        "ingredients": ingredients,
        "warnings": warnings,
    }
