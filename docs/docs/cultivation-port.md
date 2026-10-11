# Cultivation maths (ported from mycro-grow)

This page explains the `mycodemo.cultivation` package: what it is, where it comes
from and what each function does. It is the **"original" column** of the comparison
between the plain grower calculator and the MycoDemo version.

**Source:** [mycro-grow](https://github.com/SaschaHeyer/mycro-grow), MIT licence,
commit `7cb8de1`. See `THIRD_PARTY_NOTICES.md` for the full licence and file mapping.

## What each function does

All masses use one unit that you choose (kg or lb); prices are per that same unit.

| Function | In plain words |
|---|---|
| `hydration(dry_mass, moisture_pct)` | How much water to add to dry material to reach a target moisture. Water = M x dry / (1 - M), where M is the moisture as a fraction. Moisture is limited to 40-75 %. |
| `spawn(hydrated_mass, spawn_rate_pct)` | How much grain spawn to add, as a percent of the wet (hydrated) substrate. Also returns the ratio, for example "1 : 5.0". |
| `yield_from_be(dry_mass, be_pct)` | Expected fresh mushroom weight = dry substrate x biological efficiency (BE). BE is a fixed number you assume, for example 85 % for oysters. It is not predicted from data. |
| `economics(...)` | Cost, revenue, profit and margin for a batch. |
| `grow_plan(...)` | Runs the four above for one block and for the whole batch. Defaults: 60 % moisture, 20 % spawn, 85 % BE. |
| `substrate_recipe(...)` | Splits a dry base into sawdust, soy hull, bran and gypsum, then adds water. Four presets: `masters`, `suppsaw`, `soyhull`, `straw`. |
| `SPECIES_BE`, `FRUITING_CONDITIONS` | Reference tables: BE for 9 species and temperature, humidity, CO2 and timing for 14 species. These are rules of thumb from the original, without citations. |

Names were changed from camelCase to snake_case, but the maths is the same.

## How we know the port is correct

`tests/test_cultivation_parity.py` has two layers:

1. The 12 tests of the original project, rewritten in Python.
2. 25 calls whose answers came from running the **original JavaScript** once. Our
   Python must give the same numbers (to 12 digits) and the same error messages, and
   the three reference tables must be identical.

Run: `pytest tests/test_cultivation_parity.py`

## Limits of the original (kept for now, not yet changed)

These are properties of the original code, noted so the comparison is honest:

- **Dry mass only.** `hydration` assumes the starting material has no water. Real
  material as weighed holds some moisture (for example 10 %), so the water to add and the
  spawn amount come out too high. (The project plan file reports that the hosted
  calculator handles this but the repository does not. We have not verified that.)
- **Spawn basis.** Spawn is a percent of the *hydrated substrate*. MycoDemo uses a
  percent of the *whole mix* (spawn / (substrate + spawn)). A 20 % rate on the
  original basis is 16.7 % of the whole mix.
- **Yield is a fixed assumption** (BE), with no uncertainty and no link to growth.
- **No time.** Nothing predicts how many days colonisation takes.

## Changes from the original

The corrected maths lives in `mycodemo/cultivation/corrected.py` (tested in
`tests/test_cultivation_corrected.py`). The files ported from mycro-grow stay
unchanged, so the app can show both columns side by side.

| Change | Original | Corrected | Why |
|---|---|---|---|
| 1. Starting moisture | Weighed mass treated as bone dry | Dry matter = weighed mass x (1 - starting moisture); only the missing water is added | Bagged pellets and straw hold about 10 % water. Example: 5 lb at 10 % to 60 % adds 6.25 lb, not 7.5 lb (same as the hosted calculator) |
| 2. Yield basis | Yield = weighed mass x BE | Yield = dry matter x BE | BE is defined on dry substrate |
| 3. Spawn basis | Spawn % of hydrated substrate | Choose either basis; both percentages shown | MycoDemo uses % of the whole mix. 20 % of hydrated substrate = 16.7 % of the whole mix |

If the starting moisture is 0 and the spawn basis is "of_hydrated", the corrected
maths gives exactly the original numbers (checked by a test).

Not changed yet: BE is still one fixed number. A range from real bag data and the
colonisation forecast will come as further modules.
