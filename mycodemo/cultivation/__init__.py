"""Cultivation maths ported from mycro-grow (MIT). See THIRD_PARTY_NOTICES.md.

This package is a faithful port: same formulas, same clamps, same defaults as the
original JavaScript. Intentional changes (moisture handling, spawn basis) do NOT
go here; they belong in a separate module so this one stays the "original" column
of the comparison.
"""

from .calculations import economics, grow_plan, hydration, spawn, yield_from_be
from .recipes import PRESETS, substrate_recipe
from .species import FRUITING_CONDITIONS, SPECIES_BE

__all__ = [
    "hydration",
    "spawn",
    "yield_from_be",
    "economics",
    "grow_plan",
    "substrate_recipe",
    "PRESETS",
    "SPECIES_BE",
    "FRUITING_CONDITIONS",
]
