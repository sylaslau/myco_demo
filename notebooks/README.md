# Notebooks

Numbered by topic. The first digit is the topic (one app page per topic); inside a topic the
numbers follow the work steps: x.0 first look at the data, x.1 baseline maths, x.2 onwards models.
Notebooks stay thin: code the Streamlit app also needs lives in the `mycodemo` package.
Notebooks are saved without outputs, so use Run All.

| Group | Notebook | Question it answers | Needs first |
|---|---|---|---|
| 0 EDA | 0.1.dataset_composition | What is in the data? | |
| | 0.2.benchmarks | Where do commercial claims sit against lab data? | |
| | 0.3.growth_vs_temperature_plotly | Colonisation time and temperature, interactive | `pip install plotly`; reads the table in 0.2 |
| 1 Calculator | 1.0 (planned) | First look: mycro-grow inputs and outputs | |
| | 1.1 (planned) | Maths: mycro-grow original vs corrected | |
| | 1.2 (planned) | Calculator improved with ML | |
| 2 Colonisation | 2.0.colonisation_parameter_effects | First look: how did each changed recipe or condition change colonisation? | `pip install plotly`; 2.3 for chart 2 |
| | 2.1 (planned) | Maths: simple rules for days to colonise from the literature and the recipe | |
| | 2.2.colonisation_time_estimate | Colonisation time estimate from recipe and conditions (empty, planned) | |
| | 2.3.image_coverage | Photo coverage: photo to % colonised | writes `data/interim/coverage_auto.csv` |
| | 2.4.forecast_updater | Colonisation forecast: curve updated with each photo | 2.3 |
| 3 Fruiting bodies | 3.0, 3.1 (planned) | First look, maths | |
| | 3.2 Harvest readiness | Kaggle notebooks, for now still `kaggle_harvest_readiness*.ipynb` (renamed when the Kaggle work is done) | Kaggle GPU |
| 4 Yield | 4.0 (planned) | First look at the yield data | |
| | 4.1 (planned) | Maths: yield from biological efficiency (BE) | |
| | 4.2 (planned) | ML yield estimate | |
| | 4.3 (planned) | Yield updated from logged harvests | |
| 5 Material properties | 5.0.feature_screening_material | First look: do the recipe and conditions affect the final material properties? (strength now; acoustic, fire resistance later if data allow) | |
| | 5.1.material_baseline_maths | Baseline maths: average, foam rule, treatment multipliers, products | reads the table in 0.2 |
| | 5.2.property_model | Material strength model | |

Retired: 0.0.eda_1st_attempt (moved to MycoThingiverse/notebooks), templates/ (course examples, removed), 5.3.growth_strength_link (photo growth vs strength; moved to MycoMaterials/archive on 2026-10-13, its question is covered by 5.0).
