# Notebooks

Numbered by topic. The first digit is the topic (one app page per topic); inside a topic the
numbers follow the work steps: x.0 EDA, x.1 baseline maths, x.2 onwards models.
Every notebook has a number, and Kaggle notebooks are named `<number>.kaggle_<name>`.
Every notebook that trains or fits a model saves its history (per epoch, per fold, learning curve) and plots it.
Notebooks stay thin: code the Streamlit app also needs lives in the `mycodemo` package.
Notebooks are saved without outputs, so use Run All.

| Group | Notebook | Question it answers | Needs first |
|---|---|---|---|
| 0 EDA | 0.1.dataset_composition | What is in the data? | |
| | 0.2.benchmarks | Where do commercial claims sit against lab data? | |
| | 0.3.growth_vs_temperature_plotly | Colonisation time and temperature, interactive | `pip install plotly`; reads the table in 0.2 |
| 1 Calculator | 1.0 (planned) | EDA: mycro-grow inputs and outputs | |
| | 1.1 (planned) | Maths: mycro-grow original vs corrected | |
| | 1.2 (planned) | Calculator improved with ML | |
| 2 Colonisation | 2.0.colonisation_EDA (planned) | EDA: consensus table by genus and species (content being planned) | |
| | 2.1.colonisation_baseline | Baseline from the field's general consensus on spawn run (guides, textbooks, extension sources), across genera and species: tidy table, range bars, curves with a temperature slider, two heatmaps | `pip install plotly`; data in `MycoMaterials/germini/colonisation_consensus.csv` (outside the repo) |
| | 2.2.colonisation_time_estimate | Colonisation time estimate from recipe and conditions (placeholder) | |
| | 2.3.colonisation_image_EDA (planned) | EDA of the colonisation photos | runs on the PC |
| | 2.4.colonisation_image_coverage | Photo coverage: photo to % colonised | writes `data/interim/coverage_auto.csv` |
| | 2.5.colonisation_forecast_updater | Colonisation forecast: curve updated with each photo; saves the error against number of photos | 2.4 |
| | 2.6.kaggle_mould_detection | Warnings: mould detection from photos (its EDA is a section inside) | Kaggle GPU |
| 3 Fruiting bodies | 3.0.kaggle_fruiting_image_EDA | EDA: how do the photo sets behind Harvest readiness differ by farm in size, quality, light, colour and background? | Kaggle, CPU only; Duman input and `ROBOFLOW_API_KEY` |
| | 3.1 (planned) | Maths | |
| | 3.2.kaggle_harvest_readiness | Harvest readiness: leave one farm out over five farms | Kaggle GPU |
| 4 Yield | 4.0 (planned) | EDA of the yield data | |
| | 4.1 (planned) | Maths: yield from biological efficiency (BE) | |
| | 4.2 (planned) | ML yield estimate | |
| | 4.3 (planned) | Yield updated from logged harvests | |
| 5 Material properties | 5.0.feature_screening_material | EDA: do the recipe and conditions affect the final material properties? (strength now; acoustic, fire resistance later if data allow) | |
| | 5.0.colonisation_parameter_effects | The old cross-study colonisation charts: how did each changed recipe or condition change colonisation? Kept from our mostly mycomaterial studies, to revisit later | `pip install plotly`; 2.4 for chart 2 |
| | 5.1.material_baseline_maths | Baseline maths: average, foam rule, treatment multipliers, products | reads the table in 0.2 |
| | 5.2.property_model | Material strength model | |

Retired: 0.0.eda_1st_attempt (moved to MycoThingiverse/notebooks), templates/ (course examples, removed), 5.3.growth_strength_link (photo growth vs strength; moved to MycoMaterials/archive on 2026-10-13, its question is covered by 5.0).
