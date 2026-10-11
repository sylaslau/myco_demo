"""Parity tests: the Python port must give the same numbers as mycro-grow.

Two layers:
1. The 12 tests of the original ``test/grow.test.js`` (mycro-grow, commit
   7cb8de13b602035f9bcc7eecb4b5d421b9de76a6, MIT), rewritten in Python.
2. Reference values: ``_ORIGINAL`` below was produced ONCE by running the original
   JavaScript (``src/index.js``) on 25 calls and 3 data tables, then pasted here.
   Each Python result is compared with it. If the port ever drifts, these fail.
"""

import json
import re

import pytest

from mycodemo.cultivation import (
    FRUITING_CONDITIONS,
    PRESETS,
    SPECIES_BE,
    economics,
    grow_plan,
    hydration,
    spawn,
    substrate_recipe,
    yield_from_be,
)

# --- layer 1: the original 12 tests -------------------------------------------------


def close(a, b, eps=1e-6):
    return abs(a - b) < eps


def test_hydration_hits_the_target_moisture_exactly():
    r = hydration(5, 60)
    assert close(r["water"], 7.5)
    assert close(r["hydrated"], 12.5)
    assert close(r["moisture"], 0.6)
    assert close(r["water"] / r["hydrated"], 0.6)


def test_hydration_clamps_moisture_to_40_75():
    assert close(hydration(10, 90)["moisture"], 0.75)
    assert close(hydration(10, 10)["moisture"], 0.40)


def test_spawn_rate_gives_the_right_mass_and_ratio():
    s = spawn(12.5, 20)
    assert close(s["spawn"], 2.5)
    assert s["ratio"] == "1 : 5.0"
    assert spawn(10, 0)["ratio"] == "—"


def test_yield_from_biological_efficiency():
    assert close(yield_from_be(5, 85), 4.25)
    assert close(yield_from_be(5, 100), 5)


def test_economics_profit_margin_and_per_block_cost():
    e = economics(
        total_dry=500, total_spawn=250, total_yield=425, blocks=100,
        cost_sub=0.5, cost_spawn=1, extra_per_block=0.5, sale_price=8,
    )
    assert close(e["total_cost"], 550)
    assert close(e["revenue"], 3400)
    assert close(e["profit"], 2850)
    assert close(e["cost_per_block"], 5.5)
    assert close(e["margin"], (2850 / 3400) * 100)
    assert close(e["cost_per_unit_yield"], 550 / 425)


def test_grow_plan_ties_everything_together():
    p = grow_plan(
        dry_per_block=5, blocks=100, moisture=60, spawn_rate=20, be=85,
        cost_sub=0.5, cost_spawn=1, sale_price=8,
    )
    assert close(p["per_block"]["water"], 7.5)
    assert close(p["per_block"]["spawn"], 2.5)
    assert close(p["per_block"]["yield"], 4.25)
    assert close(p["totals"]["yield"], 425)
    assert p["totals"]["blocks"] == 100
    assert p["economics"]["profit"] > 0


def test_masters_mix_preset_splits_50_50_and_adds_1pct_gypsum():
    r = substrate_recipe(base_per_block=5, blocks=10, preset="masters", moisture=60)
    assert [i["name"] for i in r["ingredients"]] == [
        "Hardwood sawdust", "Soy hull pellets", "Gypsum"
    ]
    assert close(r["ingredients"][0]["per_block"], 2.5)
    assert close(r["ingredients"][1]["per_block"], 2.5)
    assert close(r["ingredients"][2]["per_block"], 0.05)
    assert close(r["totals"]["total_dry"], (5 + 0.05) * 10)
    assert len(r["warnings"]) == 0


def test_suppsaw_preset_adds_20pct_bran():
    r = substrate_recipe(base_per_block=10, blocks=1, preset="suppsaw")
    bran = next(i for i in r["ingredients"] if i["name"].startswith("Bran"))
    assert close(bran["per_block"], 2)


def test_straw_preset_treats_the_whole_base_as_straw_pellets():
    r = substrate_recipe(base_per_block=8, blocks=1, preset="straw")
    assert r["ingredients"][0]["name"] == "Straw pellets"
    assert close(r["ingredients"][0]["per_block"], 8)


def test_custom_recipe_not_summing_to_100_warns():
    r = substrate_recipe(base_per_block=5, blocks=1, p_saw=60, p_soy=60)
    assert len(r["warnings"]) == 1


def test_unknown_preset_raises_a_helpful_error():
    with pytest.raises(ValueError, match="Unknown preset"):
        substrate_recipe(base_per_block=5, blocks=1, preset="nope")


def test_data_exports_are_present_and_well_formed():
    assert len(FRUITING_CONDITIONS) == 14
    assert SPECIES_BE["Shiitake"] == 90
    for s in FRUITING_CONDITIONS:
        assert s["name"] and s["latin"] and len(s["fruitC"]) == 2
    assert "masters" in PRESETS


# --- layer 2: values produced by the original JavaScript ----------------------------

_ORIGINAL = json.loads(r'''{"data":{"SPECIES_BE":{"Blue / Pearl Oyster":85,"Pink / Yellow Oyster":75,"King Oyster (Trumpet)":60,"Lion's Mane":60,"Shiitake":90,"Chestnut":60,"Pioppino":60,"Nameko":60,"Reishi":45},"FRUITING_CONDITIONS":[{"name":"Blue Oyster","latin":"Pleurotus ostreatus","type":"Oyster","difficulty":"Beginner","colonizeC":[21,27],"fruitC":[10,21],"humidity":"85–95%","co2":"<800 ppm","fae":"High","light":"Moderate (12 h)","timeToFruit":"10–14 days","be":"75–100%+","note":"The most forgiving species and a great first grow — tolerates a wide temperature band."},{"name":"Pearl Oyster","latin":"Pleurotus ostreatus","type":"Oyster","difficulty":"Beginner","colonizeC":[21,27],"fruitC":[13,21],"humidity":"85–95%","co2":"<800 ppm","fae":"High","light":"Moderate","timeToFruit":"10–14 days","be":"75–100%","note":"The classic grey oyster; fruits a touch warmer than the blue strain."},{"name":"Pink Oyster","latin":"Pleurotus djamor","type":"Oyster","difficulty":"Beginner","colonizeC":[24,30],"fruitC":[18,30],"humidity":"85–95%","co2":"<800 ppm","fae":"High","light":"Moderate","timeToFruit":"10–14 days","be":"65–90%","note":"Loves heat and fruits fast, but very short shelf life — sell within a day or two."},{"name":"Golden / Yellow Oyster","latin":"Pleurotus citrinopileatus","type":"Oyster","difficulty":"Beginner","colonizeC":[24,30],"fruitC":[21,30],"humidity":"85–95%","co2":"<800 ppm","fae":"High","light":"Moderate","timeToFruit":"10–14 days","be":"65–90%","note":"Warm-season; delicate bright-yellow clusters that are fragile after harvest."},{"name":"King Oyster","latin":"Pleurotus eryngii","type":"Oyster","difficulty":"Intermediate","colonizeC":[22,25],"fruitC":[15,18],"humidity":"85–90%","co2":"500–800 ppm","fae":"Moderate","light":"Low–moderate","timeToFruit":"14–21 days","be":"50–75%","note":"Fruit few per block for big meaty stems; a brief cold shock helps trigger pinning."},{"name":"Lion's Mane","latin":"Hericium erinaceus","type":"Wood-lover","difficulty":"Intermediate","colonizeC":[21,24],"fruitC":[18,24],"humidity":"90–95%","co2":"<800 ppm","fae":"High","light":"Low","timeToFruit":"10–14 days","be":"50–75%","note":"High CO2 makes branchy 'coral' growth with no spines — keep fresh air high and humidity up."},{"name":"Shiitake","latin":"Lentinula edodes","type":"Wood-lover","difficulty":"Advanced","colonizeC":[21,27],"fruitC":[10,21],"humidity":"80–90%","co2":"<1000 ppm","fae":"Moderate–high","light":"Moderate","timeToFruit":"35–70 days + browning","be":"75–125%","note":"Long colonization, then the block must brown before fruiting. A cold-water soak triggers each flush."},{"name":"Chestnut","latin":"Pholiota adiposa","type":"Wood-lover","difficulty":"Intermediate","colonizeC":[21,24],"fruitC":[13,18],"humidity":"85–95%","co2":"<800 ppm","fae":"High","light":"Moderate","timeToFruit":"14–21 days","be":"50–75%","note":"Crunchy, nutty, chef-favorite clusters with a nicer shelf life than oysters."},{"name":"Pioppino","latin":"Cyclocybe aegerita","type":"Wood-lover","difficulty":"Intermediate","colonizeC":[24,27],"fruitC":[15,21],"humidity":"85–95%","co2":"<800 ppm","fae":"High","light":"Moderate","timeToFruit":"14–21 days","be":"50–75%","note":"Long elegant stems and a deep savory flavor; tolerant once fully colonized."},{"name":"Nameko","latin":"Pholiota microspora","type":"Wood-lover","difficulty":"Intermediate","colonizeC":[21,24],"fruitC":[10,16],"humidity":"90–95%","co2":"<800 ppm","fae":"High","light":"Moderate","timeToFruit":"14–30 days","be":"50–75%","note":"Cool-fruiting with a glossy gelatinous cap; a staple in miso soup."},{"name":"Enoki","latin":"Flammulina velutipes","type":"Wood-lover","difficulty":"Advanced","colonizeC":[21,24],"fruitC":[8,13],"humidity":"80–90%","co2":"High (long-stem form)","fae":"Low (restricted)","light":"Low","timeToFruit":"14–21 days","be":"50–75%","note":"The white long-stem market form is grown cold in CO2-rich tubes; with fresh air you get the wild brown form."},{"name":"Maitake (Hen of the Woods)","latin":"Grifola frondosa","type":"Wood-lover","difficulty":"Advanced","colonizeC":[21,24],"fruitC":[13,18],"humidity":"85–95%","co2":"<800 ppm","fae":"High","light":"Moderate","timeToFruit":"30–60 days + initiation","be":"~30–50%","note":"Demands extended incubation and a cool initiation — one of the trickier gourmets to fruit."},{"name":"Reishi","latin":"Ganoderma lingzhi","type":"Medicinal","difficulty":"Intermediate","colonizeC":[25,30],"fruitC":[24,29],"humidity":"90–95%","co2":"High → antlers / Low → conks","fae":"Varies by form","light":"Needed for conks","timeToFruit":"14–21 days","be":"~30–50%","note":"Steer the form with CO2: high CO2 + low light gives antlers; low CO2 + light gives flat conks. Grown for medicine, not weight."},{"name":"Turkey Tail","latin":"Trametes versicolor","type":"Medicinal","difficulty":"Beginner","colonizeC":[24,27],"fruitC":[18,24],"humidity":"85–95%","co2":"<1000 ppm","fae":"High","light":"Needed","timeToFruit":"14–30 days","be":"medicinal — not weight-based","note":"Vigorous and tolerant; grown for extracts. Light plus fresh air gives good banded rosettes."}],"PRESETS":{"masters":{"pSaw":50,"pSoy":50,"pBran":0,"pGyp":1},"suppsaw":{"pSaw":100,"pSoy":0,"pBran":20,"pGyp":1},"soyhull":{"pSaw":0,"pSoy":100,"pBran":0,"pGyp":1},"straw":{"pSaw":0,"pSoy":0,"pBran":0,"pGyp":0,"straw":true}}},"golden":[{"fn":"hydration","args":[5,60],"result":{"water":7.5,"hydrated":12.5,"moisture":0.6}},{"fn":"hydration","args":[5,10],"result":{"water":3.3333333333333335,"hydrated":8.333333333333334,"moisture":0.4}},{"fn":"hydration","args":[10,90],"result":{"water":30,"hydrated":40,"moisture":0.75}},{"fn":"hydration","args":[0,60],"result":{"water":0,"hydrated":0,"moisture":0.6}},{"fn":"hydration","args":[3.7,55.5],"result":{"water":4.614606741573034,"hydrated":8.314606741573034,"moisture":0.555}},{"fn":"hydration","args":[100,75],"result":{"water":300,"hydrated":400,"moisture":0.75}},{"fn":"spawn","args":[12.5,20],"result":{"spawn":2.5,"rate":0.2,"ratio":"1 : 5.0"}},{"fn":"spawn","args":[10,0],"result":{"spawn":0,"rate":0,"ratio":"—"}},{"fn":"spawn","args":[7.3,16.7],"result":{"spawn":1.2190999999999999,"rate":0.16699999999999998,"ratio":"1 : 6.0"}},{"fn":"spawn","args":[12.5,9.1],"result":{"spawn":1.1375,"rate":0.091,"ratio":"1 : 11.0"}},{"fn":"yieldFromBE","args":[5,85],"result":4.25},{"fn":"yieldFromBE","args":[5,100],"result":5},{"fn":"yieldFromBE","args":[2.2,60],"result":1.32},{"fn":"economics","args":[{"totalDry":500,"totalSpawn":250,"totalYield":425,"blocks":100,"costSub":0.5,"costSpawn":1,"extraPerBlock":0.5,"salePrice":8}],"result":{"totalCost":550,"costPerBlock":5.5,"revenue":3400,"profit":2850,"margin":83.82352941176471,"costPerUnitYield":1.2941176470588236}},{"fn":"economics","args":[{"totalDry":10,"totalSpawn":2,"totalYield":0,"blocks":3}],"result":{"totalCost":0,"costPerBlock":0,"revenue":0,"profit":0,"margin":0,"costPerUnitYield":null}},{"fn":"growPlan","args":[{"dryPerBlock":5,"blocks":100,"moisture":60,"spawnRate":20,"be":85,"costSub":0.5,"costSpawn":1,"extraPerBlock":0.5,"salePrice":8}],"result":{"perBlock":{"dry":5,"water":7.5,"hydrated":12.5,"spawn":2.5,"spawnRatio":"1 : 5.0","yield":4.25},"totals":{"dry":500,"water":750,"hydrated":1250,"spawn":250,"yield":425,"blocks":100},"economics":{"totalCost":550,"costPerBlock":5.5,"revenue":3400,"profit":2850,"margin":83.82352941176471,"costPerUnitYield":1.2941176470588236}}},{"fn":"growPlan","args":[{"dryPerBlock":2.5,"blocks":7}],"result":{"perBlock":{"dry":2.5,"water":3.75,"hydrated":6.25,"spawn":1.25,"spawnRatio":"1 : 5.0","yield":2.125},"totals":{"dry":17.5,"water":26.25,"hydrated":43.75,"spawn":8.75,"yield":14.875,"blocks":7},"economics":{"totalCost":0,"costPerBlock":0,"revenue":0,"profit":0,"margin":0,"costPerUnitYield":0}}},{"fn":"growPlan","args":[{"dryPerBlock":1.8,"blocks":12,"moisture":68,"spawnRate":10,"be":60,"costSub":0.3,"costSpawn":2,"salePrice":12}],"result":{"perBlock":{"dry":1.8,"water":3.825000000000001,"hydrated":5.625000000000001,"spawn":0.5625000000000001,"spawnRatio":"1 : 10.0","yield":1.08},"totals":{"dry":21.6,"water":45.90000000000001,"hydrated":67.50000000000001,"spawn":6.750000000000002,"yield":12.96,"blocks":12},"economics":{"totalCost":19.980000000000004,"costPerBlock":1.6650000000000003,"revenue":155.52,"profit":135.54000000000002,"margin":87.15277777777779,"costPerUnitYield":1.541666666666667}}},{"fn":"substrateRecipe","args":[{"preset":"masters","basePerBlock":5,"blocks":10,"moisture":60}],"result":{"perBlock":{"totalDry":5.05,"water":7.574999999999999,"hydrated":12.625},"totals":{"totalDry":50.5,"water":75.75,"hydrated":126.25,"blocks":10},"ingredients":[{"name":"Hardwood sawdust","perBlock":2.5,"total":25},{"name":"Soy hull pellets","perBlock":2.5,"total":25},{"name":"Gypsum","perBlock":0.05,"total":0.5}],"warnings":[]}},{"fn":"substrateRecipe","args":[{"preset":"suppsaw","basePerBlock":5,"blocks":10,"moisture":60}],"result":{"perBlock":{"totalDry":6.05,"water":9.075,"hydrated":15.125},"totals":{"totalDry":60.5,"water":90.75,"hydrated":151.25,"blocks":10},"ingredients":[{"name":"Hardwood sawdust","perBlock":5,"total":50},{"name":"Bran (wheat/oat)","perBlock":1,"total":10},{"name":"Gypsum","perBlock":0.05,"total":0.5}],"warnings":[]}},{"fn":"substrateRecipe","args":[{"preset":"soyhull","basePerBlock":5,"blocks":10,"moisture":60}],"result":{"perBlock":{"totalDry":5.05,"water":7.574999999999999,"hydrated":12.625},"totals":{"totalDry":50.5,"water":75.75,"hydrated":126.25,"blocks":10},"ingredients":[{"name":"Soy hull pellets","perBlock":5,"total":50},{"name":"Gypsum","perBlock":0.05,"total":0.5}],"warnings":[]}},{"fn":"substrateRecipe","args":[{"preset":"straw","basePerBlock":5,"blocks":10,"moisture":60}],"result":{"perBlock":{"totalDry":5,"water":7.5,"hydrated":12.5},"totals":{"totalDry":50,"water":75,"hydrated":125,"blocks":10},"ingredients":[{"name":"Straw pellets","perBlock":5,"total":50}],"warnings":[]}},{"fn":"substrateRecipe","args":[{"pSaw":60,"pSoy":60,"basePerBlock":5,"blocks":1}],"result":{"perBlock":{"totalDry":6,"water":8.999999999999998,"hydrated":14.999999999999998},"totals":{"totalDry":6,"water":8.999999999999998,"hydrated":14.999999999999998,"blocks":1},"ingredients":[{"name":"Hardwood sawdust","perBlock":3,"total":3},{"name":"Soy hull pellets","perBlock":3,"total":3}],"warnings":["sawdust % + soy hull % = 120% (not 100%). The base is split by these weights as entered — adjust so they total 100% for a standard recipe."]}},{"fn":"substrateRecipe","args":[{"pSaw":70,"pSoy":30,"pBran":10,"pGyp":2,"basePerBlock":3.3,"blocks":4,"moisture":66}],"result":{"perBlock":{"totalDry":3.6959999999999993,"water":7.1745882352941175,"hydrated":10.870588235294116},"totals":{"totalDry":14.783999999999997,"water":28.69835294117647,"hydrated":43.482352941176465,"blocks":4},"ingredients":[{"name":"Hardwood sawdust","perBlock":2.3099999999999996,"total":9.239999999999998},{"name":"Soy hull pellets","perBlock":0.9899999999999999,"total":3.9599999999999995},{"name":"Bran (wheat/oat)","perBlock":0.32999999999999996,"total":1.3199999999999998},{"name":"Gypsum","perBlock":0.06599999999999999,"total":0.26399999999999996}],"warnings":[]}},{"fn":"substrateRecipe","args":[{"preset":"nope","basePerBlock":5,"blocks":1}],"result":{"__error":"Unknown preset \"nope\". Use one of: masters, suppsaw, soyhull, straw"}}]}''')


def _snake(name):
    return re.sub(r"([A-Z])", lambda m: "_" + m.group(1).lower(), name)


def _snake_keys(obj):
    if isinstance(obj, dict):
        return {_snake(k): _snake_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_snake_keys(v) for v in obj]
    return obj


_FUNCS = {
    "hydration": hydration,
    "spawn": spawn,
    "yieldFromBE": yield_from_be,
    "economics": economics,
    "growPlan": grow_plan,
    "substrateRecipe": substrate_recipe,
}


def _assert_same(got, want, path=""):
    if isinstance(want, dict):
        assert set(got) == set(want), f"keys differ at {path}: {set(got) ^ set(want)}"
        for k in want:
            _assert_same(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert len(got) == len(want), f"length differs at {path}"
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_same(g, w, f"{path}[{i}]")
    elif isinstance(want, float) or isinstance(got, float):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12), path
    else:
        assert got == want, path


@pytest.mark.parametrize(
    "case", _ORIGINAL["golden"], ids=lambda c: f"{c['fn']}-{json.dumps(c['args'])[:40]}"
)
def test_port_matches_original_javascript(case):
    fn = _FUNCS[case["fn"]]
    args = case["args"]
    if len(args) == 1 and isinstance(args[0], dict):
        call = lambda: fn(**_snake_keys(args[0]))  # noqa: E731
    else:
        call = lambda: fn(*args)  # noqa: E731
    want = case["result"]
    if isinstance(want, dict) and "__error" in want:
        with pytest.raises(ValueError) as exc:
            call()
        assert str(exc.value) == want["__error"]
        return
    _assert_same(call(), _snake_keys(want))


def test_reference_tables_match_original():
    assert SPECIES_BE == _ORIGINAL["data"]["SPECIES_BE"]
    assert FRUITING_CONDITIONS == _ORIGINAL["data"]["FRUITING_CONDITIONS"]
    assert PRESETS == _ORIGINAL["data"]["PRESETS"]
