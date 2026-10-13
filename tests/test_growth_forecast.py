"""Tests for Colonisation forecast and Warnings (mycodemo.growth.forecast), using made-up curves."""

from datetime import date

import numpy as np
import pytest

from mycodemo.growth.forecast import (
    backup_from_json,
    backup_to_json,
    build_labels_html,
    csv_to_payload,
    restore_from_text,
    biological_efficiency,
    cost_summary,
    day_reaching,
    forecast,
    forecast_by_photo,
    forecast_warnings,
    logistic,
    loss_rate_text,
    pace_vs_typical,
)

T = np.arange(0, 15, 1.0)
K, T50 = 0.8, 6.0
C = logistic(T, K, T50)


def test_logistic_midpoint():
    assert logistic(T50, K, T50) == pytest.approx(0.5)


def test_day_reaching_matches_formula():
    # 90% day of a logistic: t50 + ln(9) / k
    assert day_reaching(logistic, [K, T50], 0.90) == pytest.approx(T50 + np.log(9) / K, abs=0.01)


def test_day_reaching_never():
    assert np.isnan(day_reaching(lambda t, a: 0.5 * np.ones_like(t), [1.0], 0.90))


def test_forecast_recovers_true_days():
    r = forecast(T, C)
    assert r["day90"] == pytest.approx(T50 + np.log(9) / K, abs=0.1)
    assert r["day95"] == pytest.approx(T50 + np.log(19) / K, abs=0.1)
    assert r["day95"] > r["day90"]


def test_forecast_too_few_points_fails_gracefully():
    r = forecast([1.0], [0.1])
    assert r["params"] is None and np.isnan(r["day90"])


def test_forecast_by_photo_has_one_row_per_photo():
    h = forecast_by_photo(T, C)
    assert list(h["n_photos"]) == list(range(2, len(T) + 1))


def test_warning_few_photos():
    r = forecast(T[:2], C[:2])
    assert any("photo" in w for w in forecast_warnings(T[:2], C[:2], r))


def test_warning_never_reaches_90():
    c = np.full(len(T), 0.05)  # dish that stays almost empty
    r = forecast(T, c)
    w = forecast_warnings(T, c, r)
    assert w  # something is flagged


def test_no_warning_for_good_series():
    assert forecast_warnings(T, C, forecast(T, C)) == []


def test_pace_ahead():
    assert pace_vs_typical(8.0) == "ahead"


def test_pace_within():
    assert pace_vs_typical(10) == "within"
    assert pace_vs_typical(12.5) == "within"
    assert pace_vs_typical(14) == "within"


def test_pace_behind():
    assert pace_vs_typical(20.0) == "behind"


def test_pace_unknown_for_nan():
    assert pace_vs_typical(np.nan) == "unknown"


def test_biological_efficiency():
    assert biological_efficiency(300, 1000) == pytest.approx(30.0)
    assert np.isnan(biological_efficiency(300, None))
    assert np.isnan(biological_efficiency(0, 1000))
    assert np.isnan(biological_efficiency(300, 0))


def test_loss_rate_text():
    assert loss_rate_text(2, 13) == "2 of 13 containers lost"


def test_cost_summary_with_harvest():
    costs = {"substrate": 10, "spawn": 5, "supplies": 3, "labour": 2}
    r = cost_summary(costs, 2000, price_per_kg=15)
    assert r["total"] == 20
    assert r["per_kg"] == pytest.approx(10)
    assert r["margin_per_kg"] == pytest.approx(5)


def test_cost_summary_n_a_without_harvest_or_price():
    costs = {"substrate": 10, "spawn": None}
    r = cost_summary(costs, 0, price_per_kg=15)
    assert r["total"] == 10
    assert np.isnan(r["per_kg"]) and np.isnan(r["margin_per_kg"])
    assert np.isnan(cost_summary(costs, 500)["margin_per_kg"])


def _log():
    d = date(2026, 10, 8)
    return {
        "dataset": "syldd", "dish": "SYLDD-CS-D1",
        "batch": {"batch_label": "syldd", "species": "P. ostreatus", "inoc_date": d, "particle_mm": None,
                  "n_containers": 13, "room_temp": True, "temp_c": None, "rh_pct": 85.0},
        "dish_fields": {"SYLDD-CS-D1": {"substrate": "stem", "spawn_pct": 10.0}},
        "stage_log": {"SYLDD-CS-D1": [{"stage": "Colonising", "date": date(2026, 10, 10), "note": ""}],
                      "SYLDD-CS-D2": []},
        "photos": [{"series_id": "SYLDD-CS-D1", "day": 1, "coverage": 0.05, "file": "day_01/a.jpg"},
                   {"series_id": "SYLDD-CS-D1", "day": 3, "coverage": 0.2, "file": "upload: b.jpg"}],
        "harvest": {"SYLDD-CS-D1": [{"date": date(2026, 11, 1), "g": 250.0}]},
        "dry_g": {"SYLDD-CS-D1": 1000.0},
        "losses": {"SYLDD-CS-D2": {"stage": "Colonising", "cause": "unknown", "date": date(2026, 10, 12)}},
        "costs": {"B-CS": {"currency": "CHF", "price": 20.0, "substrate": 4.0, "spawn": 3.0,
                           "supplies": None, "labour": 2.0}},
    }


def test_backup_round_trip_gives_same_log():
    log = _log()
    back = backup_from_json(backup_to_json(log))
    assert back.pop("replace") is True
    assert back == log
    assert restore_from_text(backup_to_json(log))["harvest"] == log["harvest"]


@pytest.mark.parametrize("text, message", [
    ("not json at all {", "not valid JSON"),
    ('{"hello": 1}', "not a Grow Log backup"),
    ('{"kind": "mycodemo-grow-log", "version": 99}', "not supported"),
    ('{"kind": "mycodemo-grow-log", "version": 1}', "missing"),
])
def test_backup_wrong_file_gives_clear_error(text, message):
    with pytest.raises(ValueError, match=message):
        backup_from_json(text)


def test_backup_bad_date_gives_clear_error():
    import json
    doc = json.loads(backup_to_json(_log()))
    doc["batch"]["inoc_date"] = "yesterday"
    with pytest.raises(ValueError, match="bad date"):
        backup_from_json(json.dumps(doc))


def test_csv_restore():
    csv = ("batch_label,species,substrate,spawn_pct,inoculation_date,containers_n,container_id,day,"
           "coverage_pct,stage,stage_note,harvest_flushes,dry_substrate_g\n"
           "syldd,P. ostreatus,stem,10,2026-10-08,13,SYLDD-CS-D1,0,2,Inoculated,,2026-11-01:250,1000\n"
           "syldd,P. ostreatus,stem,10,2026-10-08,13,SYLDD-CS-D1,3,20,Colonising,,2026-11-01:250,1000\n")
    p = csv_to_payload(csv)
    assert p["replace"] is False and p["dataset"] == "syldd" and p["dish"] == "SYLDD-CS-D1"
    assert p["harvest"]["SYLDD-CS-D1"] == [{"date": date(2026, 11, 1), "g": 250.0}]
    assert [e["stage"] for e in p["stage_log"]["SYLDD-CS-D1"]] == ["Inoculated", "Colonising"]
    assert p["stage_log"]["SYLDD-CS-D1"][1]["date"] == date(2026, 10, 11)


def test_csv_wrong_columns_gives_clear_error():
    with pytest.raises(ValueError, match="missing columns"):
        csv_to_payload("a,b\n1,2\n")


def test_labels_eight_per_page():
    lab = {"batch": "syldd", "container": "C", "species": "P. ostreatus", "inoculation": "2026-10-08",
           "expected": "2026-10-22", "expected_note": "typical, usemycro.com guidance"}
    out = build_labels_html([lab] * 9)
    assert out.count('class="page"') == 2 and out.count('class="label"') == 9
    assert out.count("flush: ____") == 9
