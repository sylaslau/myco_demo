"""The 2.2 estimate and 2.5 forecast in the pipeline, their modules, and the fallback when the model is missing."""

import shutil
from datetime import date, timedelta

import numpy as np
import pytest

from mycodemo import pipeline
from mycodemo.growth import consensus_estimate as ce
from mycodemo.growth import forecast_update as fu
from mycodemo.growth.forecast import logistic
from mycodemo.pipeline import Batch, Condition, Photo, run

START = date(2026, 10, 8)
TODAY = date(2026, 10, 8)            # forecast "today" = day 0 unless a test says otherwise


@pytest.fixture(scope="module")
def cm():
    try:
        return ce.load_bundle()
    except ce.ModelUnavailable:
        pytest.skip("no models/colonisation/consensus_demo_* bundle in this checkout")


def food(**kw):
    kw.setdefault("temp_c", 19.0)
    return Batch(goal="Food", choice="Blue Oyster", environment="Grow bag (about 3 kg)", inoc_date=START, **kw)


def photos(days, k=0.8, t50=6.0):
    return [Photo(day=d, file=f"p{d}.jpg", coverage=float(logistic(d, k, t50))) for d in days]


# --- module 2.2 -------------------------------------------------------------------------------
def test_estimate_reproduces_the_saved_bundle_prediction(cm):
    saved = ce.find_bundle() / "prediction.json"
    if not saved.is_file():
        pytest.skip("prediction.json not in this checkout")
    import json
    p = json.loads(saved.read_text())
    again = cm.estimate_days(**json.loads((ce.find_bundle() / "settings.json").read_text())["user_input"])
    assert json.loads(json.dumps(again)) == p


def test_temperature_rule_and_plus_two_marker(cm):
    opt = cm.estimate_days("Pleurotus ostreatus", "straw", 19.0, temperature_basis="air")
    cool = cm.estimate_days("Pleurotus ostreatus", "straw", 14.0, temperature_basis="air")   # 4 °C below the 18-20 band
    assert np.isclose(cool["adjusted_t90_min_typ_max"][1] / opt["adjusted_t90_min_typ_max"][1], 1 / 0.8)
    assert np.allclose(np.array(opt["planning_90_plus_2_day_min_typ_max"]) - opt["adjusted_t90_min_typ_max"], 2.0)


def test_low_humidity_delay_only_when_drying_exposed(cm):
    base = cm.estimate_days("Pleurotus ostreatus", "straw", 19.0, temperature_basis="air")
    dry = cm.estimate_days("Pleurotus ostreatus", "straw", 19.0, 50.0, "air", drying_exposed=True)
    sealed = cm.estimate_days("Pleurotus ostreatus", "straw", 19.0, 50.0, "air", drying_exposed=False)
    assert np.allclose(np.array(dry["adjusted_t90_min_typ_max"]) - base["adjusted_t90_min_typ_max"], [5, 7.5, 10])
    assert sealed["adjusted_t90_min_typ_max"] == base["adjusted_t90_min_typ_max"]


def test_out_of_domain_and_unsupported_inputs(cm):
    assert cm.estimate_days("Pleurotus ostreatus", "straw", 38.0, temperature_basis="air")["adjusted_t90_min_typ_max"] is None
    with pytest.raises(ValueError):
        cm.estimate_days("Pleurotus ostreatus", "logs")
    with pytest.raises(ValueError):
        cm.estimate_days("Pleurotus ostreatus", "straw", humidity_pct=120)


def test_missing_bundle_is_reported(tmp_path):
    with pytest.raises(ce.ModelUnavailable):
        ce.find_bundle(tmp_path)
    broken = tmp_path / "consensus_demo_x"
    broken.mkdir()
    (broken / "model.json").write_text("{}")
    with pytest.raises(ce.ModelUnavailable):
        ce.load_bundle(broken)


# --- module 2.5 -------------------------------------------------------------------------------
def fm_for(cm, temp=19.0):
    cond = {"temperature_c": temp, "temperature_basis": "air", "humidity_pct": None, "drying_exposed": False}
    return fu.ForecastModel.from_consensus(cm, "Pleurotus ostreatus", "sawdust", cond), cond


def test_no_coverage_forecast_equals_the_22_estimate(cm):
    fm, cond = fm_for(cm)
    f = fm.forecast_batch([], fu.prepare_conditions([], 0., cond), cond, 0.)
    est = cm.estimate_days("Pleurotus ostreatus", "sawdust", 19.0, temperature_basis="air")
    assert f["status"] == "baseline_only"
    assert np.allclose(f["t90_day_min_typ_max"], est["adjusted_t90_min_typ_max"])
    assert np.allclose(f["planning_90_plus_2_day_min_typ_max"], est["planning_90_plus_2_day_min_typ_max"])


def test_faster_photos_move_the_day_earlier_and_conditions_are_dated(cm):
    fm, cond = fm_for(cm)
    hist = fu.prepare_conditions([], 5., cond)
    base = fm.forecast_batch([], hist, cond, 5.)["t90_day_min_typ_max"][1]
    obs, _ = fu.prepare_coverage([{"day": d, "coverage_fraction": float(logistic(d, 1.2, 3.0))} for d in (1, 3, 5)], 5.)
    assert fm.forecast_batch(obs, hist, cond, 5.)["t90_day_min_typ_max"][1] < base
    # a colder spell from day 2 slows only the days after it
    cold = fu.prepare_conditions([{"day": 2, "temperature_c": 12.0}], 5., cond)
    assert np.allclose(fm.effective_clock(2., cold), fm.effective_clock(2., hist))
    assert fm.forecast_batch([], cold, cold[-1]["conditions"], 5.)["t90_day_min_typ_max"][1] > base


def test_bad_records_are_rejected():
    for records, day in [([{"day": 3, "coverage_pct": 20}], 2), ([{"day": 1, "coverage_pct": 120}], 2),
                         ([{"day": 1, "coverage_pct": 20}, {"day": 1, "coverage_pct": 30}], 2)]:
        with pytest.raises(ValueError):
            fu.prepare_coverage(records, day)


# --- pipeline wiring --------------------------------------------------------------------------
def test_pipeline_uses_the_consensus_model_when_present(cm):
    r = run(food())
    est, fc = r.colonisation_estimate, r.colonisation_forecast
    assert est.source == "consensus model" and est.value["engine"] == "consensus"
    assert fc.source == "consensus model" and fc.value["method"] == "estimate"
    names = [n for n, _, _ in r.dates.value]
    assert names == ["Inoculation", pipeline.ENDPOINT_LABEL, pipeline.MARKER_LABEL, "Pinning", "First harvest"]
    ev = {n: (a, b) for n, a, b in r.dates.value}
    assert ev[pipeline.MARKER_LABEL] == tuple(d + timedelta(days=2) for d in ev[pipeline.ENDPOINT_LABEL])
    assert "project assumption" in r.dates.note


def test_no_photos_forecast_is_the_22_estimate(cm):
    b = food()
    est = pipeline.colonisation_estimate(b)
    fc = pipeline.colonisation_forecast(b, est, today=TODAY)
    assert np.allclose(fc.value["day90_range"], est.value["days"]) and np.isclose(fc.value["day90"], est.value["typical"])
    # ... also when the batch is days old and a temperature spell is logged but no photo was taken
    old = pipeline.colonisation_forecast(b, est, today=START + timedelta(days=30))
    assert np.allclose(old.value["day90_range"], est.value["days"])


def test_photos_update_the_forecast_and_the_source_tag(cm):
    b = food(photos=photos((1, 3, 5), t50=3.0))
    est = pipeline.colonisation_estimate(b)
    fc = pipeline.colonisation_forecast(b, est, today=START + timedelta(days=5))
    assert fc.source == "photos + consensus model" and fc.value["method"] == "consensus update"
    assert fc.value["day90"] < est.value["typical"]


def test_flagged_photos_are_not_used(cm):
    flagged = [Photo(day=d, file=f"f{d}", coverage=0.9, quality="blurry") for d in (1, 3, 5)]
    b = food(photos=flagged)
    fc = pipeline.colonisation_forecast(b, pipeline.colonisation_estimate(b), today=START + timedelta(days=5))
    assert fc.value["method"] == "estimate"


def test_two_photos_on_one_day_are_averaged_not_rejected(cm):
    b = food(photos=photos((2, 2, 4)))
    fc = pipeline.colonisation_forecast(b, pipeline.colonisation_estimate(b), today=START + timedelta(days=4))
    assert fc.source == "photos + consensus model"


def test_missing_model_falls_back_and_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "BUNDLE_PARENT", tmp_path)          # no bundle here
    r = run(food(photos=photos((1, 3, 5, 7))))
    assert r.colonisation_estimate.source == "fallback: typical range"
    assert "Fallback used" in r.colonisation_estimate.note and "2.2 model" in r.colonisation_estimate.note
    assert r.colonisation_forecast.source == "fallback: photo curve"
    assert r.colonisation_forecast.value["method"] == "curve fit"
    assert r.dates.value and r.dates.value[1][0] == pipeline.ENDPOINT_LABEL        # nothing goes blank
    none = run(food())
    assert none.colonisation_estimate.source == none.colonisation_forecast.source == "fallback: typical range"


def test_broken_model_file_falls_back(monkeypatch, tmp_path, cm):
    shutil.copytree(ce.find_bundle(), tmp_path / ce.find_bundle().name)
    (tmp_path / ce.find_bundle().name / "model.json").write_text("not json")
    monkeypatch.setattr(ce, "BUNDLE_PARENT", tmp_path)
    assert run(food()).colonisation_estimate.source == "fallback: typical range"


def test_a_failing_call_falls_back(monkeypatch, cm):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(ce.ConsensusModel, "estimate_days", boom)
    est = pipeline.colonisation_estimate(food())
    assert est.source == "fallback: typical range" and "boom" in est.note and est.value["days"]


def test_failed_update_falls_back_to_the_logistic_fit(monkeypatch, cm):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(fu.ForecastModel, "forecast_batch", boom)
    b = food(photos=photos((1, 3, 5, 7)))
    fc = pipeline.colonisation_forecast(b, pipeline.colonisation_estimate(b), today=START + timedelta(days=7))
    assert fc.source == "fallback: photo curve" and fc.value["method"] == "curve fit" and "2.5 model" in fc.note


def test_temperature_with_no_rule_falls_back_for_the_old_range(cm):
    est = pipeline.colonisation_estimate(food(temp_c=38.0))        # outside the 2.2 quantitative domain
    assert est.source == "fallback: typical range" and est.value["days"]
    assert pipeline.colonisation_forecast(food(temp_c=38.0), est).value["day90_range"]


def test_logged_conditions_move_the_forecast_but_future_ones_do_not_break_it(cm):
    base = food(photos=photos((1, 3)))
    cold = food(photos=photos((1, 3)), conditions=[Condition(START + timedelta(days=2), 13.0, 90.0)])
    today = START + timedelta(days=4)
    a = pipeline.colonisation_forecast(base, pipeline.colonisation_estimate(base), today=today)
    b = pipeline.colonisation_forecast(cold, pipeline.colonisation_estimate(cold), today=today)
    assert b.value["day90"] > a.value["day90"]
    # a reading dated before inoculation is ignored, not an error
    early = food(conditions=[Condition(START - timedelta(days=3), 20.0, None)])
    assert pipeline.colonisation_forecast(early, pipeline.colonisation_estimate(early), today=today).source == "consensus model"


def test_materials_batch_still_works(cm):
    b = Batch(goal="Materials", choice="Foam", environment="Grow bag (about 3 kg)", inoc_date=START, temp_c=19.0)
    r = run(b)
    assert [n for n, _, _ in r.dates.value] == ["Inoculation", pipeline.ENDPOINT_LABEL, "Post-treatment can start"]


@pytest.mark.parametrize("species, substrate", [("Pleurotus djamor", "straw"), ("Hericium erinaceus", "sawdust"),
                                                 ("Flammulina velutipes", "sawdust")])
def test_air_temperature_uses_consensus_optimum_when_review_has_no_band(cm, species, substrate):
    # Pink Oyster, Lion's Mane, Enoki: no review air band, so the ref_2 optimum is the band
    r = cm.estimate_days(species, substrate, 20.0, temperature_basis="air")
    assert r["environment_rules"]["available"]
    assert r["adjusted_t90_min_typ_max"] is not None
    row = cm.references[(cm.references.species == species) & (cm.references.substrate_code == substrate)].iloc[0]
    opt = float(row.incubation_temp_opt_C)
    at_opt = cm.estimate_days(species, substrate, opt, temperature_basis="air")
    assert at_opt["environment_rules"]["temperature_duration_factor"] == 1.0
    assert "consensus optimum" in " ".join(at_opt["environment_rules"]["notes"])
    assert not cm.estimate_days(species, substrate, 20.0, temperature_basis="unknown")["environment_rules"]["optimum_band"]
