"""Check image: traced_coverage (moved from notebook 2.4), checked photos in the Colonisation forecast, the feedback
texts, and the Check image page in AppTest with a fake Save value."""

import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mycodemo import pipeline
from mycodemo.growth import consensus_estimate as ce
from mycodemo.growth.traced_coverage import circle, coverage_from_marks, segment
from mycodemo.pipeline import Batch, Photo, run

ROOT = Path(__file__).parents[1]
PHOTOS = ROOT / "data/raw/syldd_2026"
START = date(2026, 10, 8)


def food(photos=(), **kw):
    kw.setdefault("temp_c", 19.0)
    return Batch(goal="Food", choice="Blue Oyster", environment="Grow bag (about 3 kg)", inoc_date=START,
                 photos=list(photos), **kw)


def png(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def half_white():
    """80 x 100 software pattern: left half bright (mycelium), right half dark (substrate)."""
    a = np.full((80, 100, 3), 55, np.uint8)
    a[:, :50] = 235
    return a


SEEDS = [{"label": "mycelium", "x": .2, "y": .5, "radius": .03}, {"label": "substrate", "x": .8, "y": .5, "radius": .03}]


@pytest.fixture(scope="module")
def consensus():
    try:
        return ce.load_bundle()
    except ce.ModelUnavailable:
        pytest.skip("no models/colonisation/consensus_demo_* bundle in this checkout")


# --- traced_coverage: the notebook's numbers ---------------------------------------------------------
# PHOTO_RECORDS of notebook 2.4 and the numbers its own copy of the code gave before the move (2026-10-20)
NOTEBOOK_RECORDS = [
    ("day_04/IMG_20261012_154516.jpg", circle(.51, .51, .42), [(.255, .365, "mycelium"), (.65, .35, "substrate")],
     47.193363761501125, 41.50166630442657, 53.76258784322249),
    ("day_08/IMG_20261016_154903.jpg", circle(.52, .49, .41), [(.50, .60, "mycelium"), (.70, .77, "substrate")],
     76.6825579671301, 73.10150283040029, 80.83995787664853),
]


@pytest.mark.parametrize("path, outline, clicks, pct, lower, upper", NOTEBOOK_RECORDS)
def test_traced_coverage_gives_the_notebook_numbers(path, outline, clicks, pct, lower, upper):
    if not (PHOTOS / path).is_file():
        pytest.skip("syldd photos are not in this checkout")
    marks = {"roi": {"kind": "polygon", "points": outline},
             "seeds": [{"label": lab, "x": x, "y": y, "radius": .015} for x, y, lab in clicks]}
    res = coverage_from_marks(str(PHOTOS / path), marks)
    assert res["coverage_pct"] == pytest.approx(pct, abs=1e-9)
    assert (res["lower_pct"], res["upper_pct"]) == pytest.approx((lower, upper), abs=1e-9)
    assert res["uncertainty"] == 10.0 and res["method"] == "per-image colour/texture examples"
    # bytes give the same as the file path (the app passes the uploaded bytes)
    assert coverage_from_marks((PHOTOS / path).read_bytes(), marks)["coverage_pct"] == res["coverage_pct"]


def test_coverage_from_marks_on_a_software_pattern():
    res = coverage_from_marks(png(half_white()), {"full_frame_substrate_confirmed": True, "seeds": SEEDS})
    assert res["coverage_pct"] == pytest.approx(50.0, abs=1.0)
    brushed = coverage_from_marks(png(half_white()), {"full_frame_substrate_confirmed": True, "seeds": SEEDS,
                                                      "strokes": [{"label": "mycelium", "size": .3,
                                                                   "points": [[.7, .3], [.8, .7]]}]})
    assert brushed["coverage_pct"] > res["coverage_pct"]


def test_coverage_from_marks_needs_an_outline():
    with pytest.raises(ValueError, match="boundary"):
        coverage_from_marks(png(half_white()), {"seeds": SEEDS})


def test_segment_keeps_the_brush_after_polygons():
    full = np.ones((80, 100), bool)
    m, u, *_ = segment(half_white(), full, {"seeds": SEEDS, "strokes": [{"label": "substrate", "size": .3,
                                                                       "points": [[.25, .5]]}]})
    assert not m[40, 25] and not u[40, 25]


# --- the forecast: a checked photo replaces the HSV one ------------------------------------------------
def test_checked_photo_replaces_hsv_of_the_same_sample_and_day():
    hsv_a = Photo(day=6, file="a.jpg", coverage=0.05, sample="A")
    checked_a = Photo(day=6, file="a.jpg", coverage=0.40, method=pipeline.MARKED_METHOD, sample="A", checked=True)
    hsv_b = Photo(day=6, file="b.jpg", coverage=0.10, sample="B")              # other sample: kept
    hsv_a8 = Photo(day=8, file="a8.jpg", coverage=0.20, sample="A")            # other day: kept
    used = pipeline.forecast_photos(food([hsv_a, checked_a, hsv_b, hsv_a8]))
    assert used == [checked_a, hsv_b, hsv_a8]


def test_checked_photo_moves_the_consensus_forecast_and_the_note_says_so(consensus):
    hsv = Photo(day=8, file="a.jpg", coverage=0.05, sample="A")
    checked = Photo(day=8, file="a.jpg", coverage=0.60, method=pipeline.MARKED_METHOD, sample="A", checked=True)
    other = Photo(day=8, file="b.jpg", coverage=0.40, sample="B")
    only_hsv = run(food([hsv, other])).colonisation_forecast
    both = run(food([hsv, checked, other])).colonisation_forecast
    no_hsv = run(food([checked, other])).colonisation_forecast
    assert both.source == "photos + consensus model"
    assert both.value["day90"] == no_hsv.value["day90"] < only_hsv.value["day90"]   # mean of 0.60 and 0.40, not 0.05
    assert both.value["n_photos"] == 2 and both.value["n_checked"] == 1
    assert "2 photo(s), 1 of them checked by you" in both.note
    assert "0 of them checked by you" in only_hsv.note


def test_checked_photo_counts_even_with_a_quality_note(consensus):
    flagged = Photo(day=8, file="a.jpg", coverage=0.6, quality="blurry", sample="A")
    assert run(food([flagged])).colonisation_forecast.value["method"] == "estimate"
    checked = Photo(day=8, file="a.jpg", coverage=0.6, quality="blurry", sample="A", checked=True)
    assert run(food([checked])).colonisation_forecast.value["method"] == "consensus update"


def test_old_backup_photos_without_sample_still_load():
    b = Batch.from_dict({**food().to_dict(), "photos": [{"day": 3, "file": "x.jpg", "coverage": 0.1}]})
    assert b.photos[0].sample == "" and b.photos[0].checked is False


# --- feedback texts --------------------------------------------------------------------------------
def test_typical_coverage_curve_hits_2_and_90_percent():
    assert pipeline.typical_coverage(0, 15) == pytest.approx(0.02)
    assert pipeline.typical_coverage(15, 15) == pytest.approx(0.90)
    assert pipeline.typical_coverage(6, 12) > pipeline.typical_coverage(6, 20)


@pytest.mark.parametrize("pct, verdict", [(38.2, "so in range"), (51, "so faster than typical"),
                                          (12, "so slower than typical"), (45.4, "so in range")])
def test_pace_text(pct, verdict):
    text = pipeline.pace_text(pct, 6, (0.20, 0.45))
    assert text.startswith(f"{round(pct)}% of the surface colonised on day 6. Typical for day 6: 20-45%, ")
    assert text.endswith(verdict + ".")


def test_pace_text_without_band_gives_only_the_percentage():
    assert pipeline.pace_text(38.4, 6, None) == "38% of the surface colonised on day 6."


def test_typical_band_uses_the_earliest_and_latest_day_of_the_estimate():
    b = food()
    lo, hi = pipeline.colonisation_estimate(b).value["days"]
    band = pipeline.typical_band(b, 6)
    assert band == (pipeline.typical_coverage(6, hi), pipeline.typical_coverage(6, lo))
    assert band[0] < band[1]
    assert pipeline.typical_band(Batch(goal="Food", choice="Custom", inoc_date=START), 6) is None


@pytest.mark.parametrize("new, old, text", [
    (date(2026, 10, 24), date(2026, 10, 26), "Estimated induction day: 24 Oct, 2 days earlier than before this photo."),
    (date(2026, 10, 27), date(2026, 10, 26), "Estimated induction day: 27 Oct, 1 day later than before this photo."),
    (date(2026, 10, 26), date(2026, 10, 26), "Estimated induction day: 26 Oct, unchanged by this photo."),
])
def test_induction_text(new, old, text):
    assert pipeline.induction_text(new, old) == text


def test_induction_date_is_forecast_90_day_plus_two():
    b = food()
    r = run(b)
    fc = r.colonisation_forecast.value
    assert pipeline.induction_date(b, r) == START + timedelta(days=round(fc["day90"] + fc.get("post_90_days", 2)))


# --- the page in AppTest, with a fake Save value from the component -----------------------------------
def page(monkeypatch, save_value, plan=True):
    """AppTest of 5_Check_image.py: one uploaded HSV photo of sample B1-01 on day 6; the component returns save_value."""
    from streamlit.testing.v1 import AppTest

    from mycodemo.app import components
    if save_value != "real":
        monkeypatch.setattr(components, "check_image", lambda **kw: save_value)
    at = AppTest.from_file(str(ROOT / "mycodemo/app/pages/5_Check_image.py"), default_timeout=60)
    at.session_state["goal"], at.session_state["choice"] = "Food", "Blue Oyster"
    if plan:
        at.session_state["batch"] = {"batch_label": "B1", "species": "Pleurotus ostreatus", "inoc_date": START,
                                     "n_containers": 2, "temp_c": 19.0, "substrate": "sawdust", "spawn_pct": 10.0}
    at.session_state["sample_photos"] = {"B1-01": [
        {"series_id": "B1-01", "day": 6, "file": "upload: d6.png", "coverage": 0.02, "method": "HSV threshold",
         "quality": "", "date": "2026-10-14", "data": png(half_white())}]}
    at.session_state["check_target"] = {"sample": "B1-01", "file": "upload: d6.png"}
    at.run()
    return at


def saved(reviewed=True, t=1):
    return {"photo_id": "B1-01|upload: d6.png", "saved_at": t, "reviewed": reviewed,
            "record": {"full_frame_substrate_confirmed": True, "seeds": SEEDS, "strokes": [], "view": "top",
                       "date": "2026-10-14"}}


def text_of(at):
    return " ".join(m.value for m in at.markdown) + " " + " ".join(w.value for w in at.warning) + " " + \
        " ".join(i.value for i in at.info)


def test_page_stores_the_checked_percentage_and_gives_feedback(monkeypatch):
    at = page(monkeypatch, saved())
    assert not at.exception
    row = at.session_state["sample_photos"]["B1-01"][0]
    assert row["method"] == pipeline.MARKED_METHOD and row["checked"] is True
    assert row["coverage"] == pytest.approx(0.50, abs=0.01) and row["hsv_coverage"] == 0.02
    assert row["marks"]["seeds"] == SEEDS
    text = text_of(at)
    assert "50% of the surface colonised on day 6. Typical for day 6:" in text
    assert "Estimated induction day:" in text and "than before this photo" in text
    assert "not used in the forecast" not in text
    photo = at.session_state["pipeline_batch"].photos[0]
    assert photo.checked and photo.coverage == row["coverage"]       # the HSV value is replaced in the forecast


def test_page_unticked_photo_is_not_used_in_the_forecast(monkeypatch):
    at = page(monkeypatch, saved(reviewed=False))
    assert not at.exception
    row = at.session_state["sample_photos"]["B1-01"][0]
    assert row["method"] == pipeline.MARKED_METHOD and row["checked"] is False
    text = text_of(at)
    assert "not used in the forecast until you tick" in text
    assert "would be" in text
    assert at.session_state["pipeline_batch"].photos == []


def test_page_without_a_plan_batch_says_what_is_missing(monkeypatch):
    at = page(monkeypatch, saved(), plan=False)
    assert not at.exception
    text = text_of(at)
    assert "no plan batch yet" in text and "Estimated induction day" not in text


def test_page_before_save_shows_no_result(monkeypatch):
    at = page(monkeypatch, None)
    assert not at.exception
    assert not [s for s in at.subheader if s.value == "Result"]
    assert at.session_state["sample_photos"]["B1-01"][0]["method"] == "HSV threshold"


def test_page_runs_with_the_real_component(monkeypatch):
    at = page(monkeypatch, "real")           # no fake: the declared component itself (AppTest gets its default None)
    assert not at.exception


def test_page_with_no_photos_points_to_monitor():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "mycodemo/app/pages/5_Check_image.py"), default_timeout=60)
    at.run()
    assert "Upload a photo" in at.info[0].value
    # st.page_link cannot find pages in a single-page AppTest (AppTest limit, not the app)
    assert all("Could not find page" in e.message for e in at.exception)
