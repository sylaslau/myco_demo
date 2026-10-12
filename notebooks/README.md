# Notebooks

Numbered by topic. The first digit is the topic (one app page per topic); inside a topic the
numbers follow the work steps: look at the data, baseline maths, then ML.
Notebooks stay thin: code the Streamlit app also needs lives in the `mycodemo` package.
Notebooks are saved without outputs, so use Run All.

| Group | Notebook | Question it answers | Needs first |
|---|---|---|---|
| 0 Look at the data | 0.1.dataset_composition | What is in the data? | |
| | 0.2.benchmarks | Where do commercial claims sit against lab data? | |
| | 0.3.growth_vs_temperature_plotly | Growth time and temperature, interactive | `pip install plotly`; reads the table in 0.2 |
| | 0.4.growth_parameter_effects | How did each changed growth parameter change growth? | `pip install plotly`; 2.1 for chart 2 |
| | 0.5.feature_screening_material | Which inputs can the **Material strength** model learn? | |
| 1 Calculator | 1.0 (planned) | mycro-grow maths: original vs corrected | |
| | 1.1 (planned) | Calculator improved with ML | |
| 2 Growth | 2.0.colonisation_time_estimate | **Colonisation time estimate** from recipe and conditions (empty, planned) | |
| | 2.1.image_coverage | **Photo coverage**: photo to % colonised | writes `data/interim/coverage_auto.csv` |
| | 2.2.forecast_updater | **Growth forecast**: curve updated with each photo | 2.1 |
| 3 Fruiting bodies | (free) | | |
| 4 Yield | (free) | | |
| 5 Material properties | 5.1 (planned) | Baseline maths for material properties | |
| | 5.2.property_model | **Material strength** model | |
| | 5.3.growth_strength_link | Growth vs strength; to be reworked to: do growing conditions change material properties? | 2.1 |

Retired: 0.0.eda_1st_attempt (moved to MycoThingiverse/notebooks), templates/ (course examples, removed).
