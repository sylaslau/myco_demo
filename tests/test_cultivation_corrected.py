"""Tests for the corrected grow maths (mycodemo.cultivation.corrected).

Each test checks one intentional change against a hand calculation, and that
the corrected maths falls back to the original when the change does not apply.
"""

import pytest

from mycodemo.cultivation import grow_plan, hydration, spawn
from mycodemo.cultivation.corrected import (
    dry_matter,
    grow_plan_corrected,
    hydration_corrected,
    spawn_corrected,
)


def close(a, b, eps=1e-9):
    return abs(a - b) < eps


# --- change 1: starting moisture -----------------------------------------------------


def test_matches_hosted_calculator_example():
    # usemycro.com: 5 lb at 10 % moisture, target 60 % -> add 6.25, hydrated 11.25
    h = hydration_corrected(5, 60, 10)
    assert close(h["dry"], 4.5)
    assert close(h["water"], 6.25)
    assert close(h["hydrated"], 11.25)
    assert close(h["moisture"], 0.60)


def test_zero_start_moisture_equals_the_original():
    for w, m in [(5, 60), (2.3, 45), (10, 75)]:
        h = hydration_corrected(w, m, 0)
        o = hydration(w, m)
        assert close(h["water"], o["water"])
        assert close(h["hydrated"], o["hydrated"])


def test_final_moisture_hits_target_for_any_start():
    for m0 in [0, 5, 10, 25, 50]:
        assert close(hydration_corrected(3, 62, m0)["moisture"], 0.62)


def test_too_wet_material_adds_no_water_and_warns():
    h = hydration_corrected(5, 60, 70)
    assert h["water"] == 0
    assert close(h["moisture"], 0.70)
    assert h["warnings"]


def test_dry_matter_split():
    d = dry_matter(10, 12)
    assert close(d["dry"], 8.8)
    assert close(d["water_in"], 1.2)


# --- change 2: yield from true dry matter --------------------------------------------


def test_yield_uses_dry_matter():
    p = grow_plan_corrected(5, 1, start_moisture=10, be=100)
    assert close(p["per_block"]["yield"], 4.5)


# --- change 3: spawn basis -----------------------------------------------------------


def test_spawn_of_hydrated_equals_the_original():
    s = spawn_corrected(12.5, 20, "of_hydrated")
    assert close(s["spawn"], spawn(12.5, 20)["spawn"])
    assert s["ratio"] == "1 : 5.0"


def test_spawn_of_total_mix_definition():
    s = spawn_corrected(12.5, 10, "of_total_mix")
    assert close(s["spawn"] / (12.5 + s["spawn"]), 0.10)
    assert close(s["pct_of_total_mix"], 10)


def test_twenty_percent_of_hydrated_is_16_7_of_mix():
    s = spawn_corrected(12.5, 20, "of_hydrated")
    assert round(s["pct_of_total_mix"], 1) == 16.7


def test_unknown_basis_is_refused():
    with pytest.raises(ValueError):
        spawn_corrected(10, 10, "of_dry")


# --- whole plan ----------------------------------------------------------------------


def test_plan_falls_back_to_original_when_no_change_applies():
    o = grow_plan(5, 100, 60, 20, 85, 0.5, 1.0, 0.5, 8)
    c = grow_plan_corrected(
        5, 100, start_moisture=0, moisture=60, spawn_pct=20, spawn_basis="of_hydrated",
        be=85, cost_sub=0.5, cost_spawn=1.0, extra_per_block=0.5, sale_price=8,
    )
    for k in ["water", "hydrated", "spawn", "yield"]:
        assert close(c["totals"][k], o["totals"][k])
    for k in ["total_cost", "revenue", "profit"]:
        assert close(c["economics"][k], o["economics"][k])


def test_plan_with_ten_percent_moisture():
    # 100 blocks of 5 lb at 10 %: hydrated 11.25 each; spawn 20 % of hydrated = 2.25
    c = grow_plan_corrected(5, 100, start_moisture=10, spawn_pct=20, spawn_basis="of_hydrated")
    assert close(c["totals"]["water"], 625)
    assert close(c["totals"]["spawn"], 225)
    assert close(c["totals"]["yield"], 382.5)  # 450 dry x 85 %
