import pytest

from mycodemo.cultivation import baseline_guide as bg

NUMERIC = [
    c for c in bg.load().columns
    if bg.load()[c].dtype.kind in "if"
]


def test_shape():
    df = bg.load()
    assert len(df) == 22
    assert len(bg.species_list()) == 8
    assert len(bg.load_text()) == 22


def test_ref_ids_unique_and_match_text():
    df = bg.load()
    assert df["ref_id"].is_unique
    assert df["ref_id"].tolist() == [f"cb{i:02d}" for i in range(1, 23)]
    assert bg.load_text()["ref_id"].tolist() == df["ref_id"].tolist()


def test_species_substrate_pairs_unique():
    assert not bg.load().duplicated(["species", "substrate_code"]).any()


def test_numeric_columns_have_no_blanks():
    assert NUMERIC
    assert not bg.load()[NUMERIC].isna().any().any()


def test_first_harvest_after_pinning():
    df = bg.load()
    assert (df["days_to_first_harvest_typical"] >= df["days_to_pinning_typical"]).all()


def test_colonisation_ordered():
    df = bg.load()
    assert (df["colonisation_days_min"] <= df["colonisation_days_typical"]).all()
    assert (df["colonisation_days_typical"] <= df["colonisation_days_max"]).all()


def test_row_exact():
    r, note = bg.row("Pleurotus ostreatus", "straw")
    assert r["substrate_code"] == "straw"
    assert "no straw row" not in note


def test_row_falls_back_to_sawdust():
    # P. ostreatus has straw, cotton, sawdust but no logs row
    r, note = bg.row("Pleurotus ostreatus", "logs")
    assert r["substrate_code"] == "sawdust"
    assert "no logs row" in note and "sawdust" in note and r["ref_id"] in note


def test_row_falls_back_to_first_when_no_sawdust():
    df = bg.load()
    no_sawdust = [
        s for s in bg.species_list() if "sawdust" not in bg.substrates(s)
    ]
    if not no_sawdust:
        pytest.skip("every species has a sawdust row")
    r, note = bg.row(no_sawdust[0], "logs")
    assert r["ref_id"] == df[df["species"] == no_sawdust[0]].iloc[0]["ref_id"]


def test_row_unknown_species():
    with pytest.raises(KeyError):
        bg.row("Not a mushroom", "straw")


def test_temp_check_edges():
    assert bg.temp_check(10, 10, 20) == "inside"
    assert bg.temp_check(20, 10, 20) == "inside"
    assert bg.temp_check(9.9, 10, 20) == "below"
    assert bg.temp_check(20.1, 10, 20) == "above"


def test_spawn_on_our_basis():
    assert bg.spawn_on_our_basis(20) == pytest.approx(16.6667, abs=1e-3)
    assert bg.spawn_on_our_basis(0) == 0


def test_stages_do_not_share_keys():
    for ref in bg.load()["ref_id"]:
        r = bg.load().set_index("ref_id", drop=False).loc[ref]
        assert set(bg.colonisation(r)).isdisjoint(bg.fruiting(r))


def test_fruiting_has_induction_text():
    r, _ = bg.row("Pleurotus ostreatus", "straw")
    f = bg.fruiting(r)
    assert f["induction_actions"]["overview"]
    assert f["first_harvest"] >= f["days_to_pinning"]
    assert f["induction_evidence_status"] == "source_supported"


def test_matches_filters_by_substrate_and_incubation_range():
    hit = bg.matches("sawdust", 24)
    assert not hit.empty and set(hit["substrate_code"]) == {"sawdust"}
    assert ((hit["incubation_temp_min_C"] <= 24) & (24 <= hit["incubation_temp_max_C"])).all()
    assert bg.matches("sawdust", -5).empty


def test_stages_agree_with_pipeline_dates_step_for_blue_oyster(monkeypatch):
    """The ref_2 stage table against the dates step of the fallback maths. The pipeline's induction planning marker,
    Pinning and First harvest sit POST_90_DAYS (the project's 2-day assumption) later than the table's days."""
    from datetime import date, timedelta

    from mycodemo import pipeline
    from mycodemo.growth.consensus_estimate import ModelUnavailable

    def missing():
        raise ModelUnavailable("test")
    monkeypatch.setattr(pipeline, "_consensus", missing)
    batch = pipeline.Batch(goal="Food", choice="Blue Oyster", inoc_date=date(2026, 1, 1))
    row, _ = pipeline.baseline_row(batch)
    st_ = bg.stages(row).set_index("stage")
    est = pipeline.colonisation_estimate(batch)
    fc = pipeline.colonisation_forecast(batch, est)
    events = pipeline.dates_step(batch, fc).value
    table = {"Inoculation": ("Inoculation", 0), pipeline.ENDPOINT_LABEL: ("Fully colonised", 0),
             pipeline.MARKER_LABEL: ("Start induction", 2), "Pinning": ("Pinning", 2),
             "First harvest": ("First harvest", 2)}
    assert [e[0] for e in events] == list(table)
    for label, start, end in events:
        stage, shift = table[label]
        assert start == batch.inoc_date + timedelta(days=int(st_.loc[stage, "start_day"]) + shift)
        assert end == batch.inoc_date + timedelta(days=int(st_.loc[stage, "end_day"]) + shift)
