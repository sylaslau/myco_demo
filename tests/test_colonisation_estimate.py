import numpy as np
import pandas as pd
import pytest

from mycodemo.growth import colonisation_estimate as ce


def synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"plot_group": rng.choice(ce.GROUPS, n), "substrate_group": rng.choice(["straw", "compost", "other"], n),
                       "temp_c": rng.uniform(18, 30, n)})
    log_days = (np.log(15) + 0.3 * (df.plot_group == "King oyster") + 0.2 * (df.substrate_group == "compost")
                + 1.0 * -np.log(ce.cardinal_rate(df.temp_c, 25, 10, 35)))
    df["days_typ"] = np.exp(log_days)
    return df


def test_fit_recovers_known_effects():
    m = ce.fit_model(synthetic(), "cardinal")
    assert m["coef"]["group=King oyster"] == pytest.approx(0.3)
    assert m["coef"]["substrate=compost"] == pytest.approx(0.2)
    assert m["coef"]["temp"] == pytest.approx(1.0)
    assert m["intercept"] == pytest.approx(np.log(15))
    assert not m["rank_deficient"]


def test_percent_effects_are_against_reference():
    m = ce.fit_model(synthetic(), "cardinal")
    assert ce.percent_effects(m)["substrate=compost"] == pytest.approx(np.expm1(0.2) * 100)
    assert "temp" not in ce.percent_effects(m)


def test_predict_matches_data_and_is_json_ready():
    import json
    df = synthetic()
    m = json.loads(json.dumps(ce.fit_model(df, "cardinal")))
    assert ce.predict_typical(m, df) == pytest.approx(df.days_typ.to_numpy())


def test_distance_form_uses_group_optimum():
    df = synthetic()
    m = ce.fit_model(df, "distance")
    assert set(m["group_opt"]) == set(ce.GROUPS) | {"all"}
    assert "temp" in m["coef"]


def test_absent_level_gets_no_term_and_rank_flag():
    df = synthetic()
    df = df[df.plot_group != "Tropical oyster"]
    assert "group=Tropical oyster" not in ce.fit_model(df)["coef"]
    twin = pd.DataFrame({"plot_group": ["Cool oyster"] * 3 + ["King oyster"], "substrate_group": ["straw"] * 3 + ["supplemented sawdust"],
                         "temp_c": [24.0] * 4, "days_typ": [15.0, 15.0, 15.0, 20.0]})
    assert ce.fit_model(twin, "none")["rank_deficient"]


def test_cardinal_rate_shape():
    assert ce.cardinal_rate(25, 25, 10, 35) == 1
    assert ce.cardinal_rate(17.5, 25, 10, 35) == pytest.approx(0.5)
    assert ce.cardinal_rate(5, 25, 10, 35) == ce.RATE_FLOOR


def test_band_ratios_and_baseline():
    df = pd.DataFrame({"plot_group": ["A", "A", "B"], "days_min": [10, 12, 8], "days_typ": [15, 16, np.nan], "days_max": [20, 24, 12]})
    r = ce.band_ratios(df)
    assert r["n_rows"] == 2 and r["min_over_typ"] == pytest.approx(np.median([10 / 15, 12 / 16]))
    b = ce.baseline_band(df)
    assert b["A"] == {"min": 11.0, "typ": 15.5, "max": 22.0}
    assert np.isnan(b["B"]["typ"]) and b["all"]["max"] == 20.0


@pytest.mark.parametrize("pct, warns", [(None, False), (5, False), (12, False), (20, False), (4.9, True), (0.9, True), (21, True)])
def test_spawn_warning_bounds(pct, warns):
    assert (ce.spawn_warning(pct) is not None) is warns


def coefficients(use_model):
    df = synthetic()
    return {"use_model": use_model, "model": ce.fit_model(df, "cardinal"),
            "band_ratios": {"min_over_typ": 0.8, "max_over_typ": 1.5},
            "baseline": {"Cool oyster": {"min": 12, "typ": 16, "max": 24}, "all": {"min": 11, "typ": 15, "max": 22}}}


def test_estimate_model_mode():
    est = ce.estimate_days("Cool oyster", "straw", 25, 10, coefficients(True))
    assert est.source == "our model" and est.spawn_warning is None
    assert est.min_days == pytest.approx(0.8 * est.typical_days) and est.max_days == pytest.approx(1.5 * est.typical_days)
    assert est.typical_days == pytest.approx(15)
    slower = ce.estimate_days("Cool oyster", "compost", 20, None, coefficients(True))
    assert slower.typical_days > est.typical_days
    assert ce.estimate_days("Cool oyster", "straw", None, None, coefficients(True)).typical_days == pytest.approx(15)


def test_estimate_baseline_mode_ignores_inputs():
    a = ce.estimate_days("Cool oyster", "straw", 20, 3, coefficients(False))
    b = ce.estimate_days("Cool oyster", "compost", 30, None, coefficients(False))
    assert a[:3] == b[:3] == (12, 16, 24) and a.source == "typical range" and a.spawn_warning and not b.spawn_warning
    assert ce.estimate_days("King oyster", "straw", 25, None, coefficients(False))[:3] == (11, 15, 22)


def test_estimate_without_file_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "COEFFICIENTS_PATH", tmp_path / "missing.json")
    assert ce.load_coefficients() is None
    est = ce.estimate_days("Cool oyster", "straw", 25)
    assert est[:3] == ce.FALLBACK_BAND and est.source == "typical range"
