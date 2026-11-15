# Saved product-analytics MVP data

The notebook has editable English constants. This folder holds the frozen multilingual evaluation suite and input PNGs; it needs no generation scripts.

## Synthetic A/B tasks

- `reports/`: 20 analyst notes in English, Spanish, French and German. Some values are inferred from totals or durations; some are unavailable.
- `charts/`: exact Plotly-rendered PNGs sent to models. Assignment timelines are line charts, dates are only axis labels, and acquisition mixes are horizontal stacked bars. `plotly/` retains source figures for audit; the notebook displays the PNGs.
- `chart_data.json`, `ground_truth.json`: underlying counts and numerical records for audit. These are not model inputs.

A/B assignment is at unique-visitor level, with one binary purchase outcome per eligible visitor. Synthetic dates are inclusive UTC and money is GBP. Absolute uplift is in percentage points; relative uplift is in percent. The better-option task chooses by the 95% difference interval: `equal` means inconclusive, not equivalence or a commercial launch decision.

## Controlled estimation example

`historical_experiments.json` has exactly ten noisy historical estimates across UI changes, performance improvements and bug fixes. `proposed_experiments.json` has 15 held-out feature combinations. The simulator's numerical coefficients, qualitative assumptions and expected effects are documented in `estimation_design.json`. Models receive all ten histories and descriptions of the proposed changes, without the simulator’s structural assumptions, coefficients or target truth. This is a toy-world prediction task, separate from the document portfolio.

## Real forecasting data and attribution

Fanaee-T, H. (2013). **Bike Sharing** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894

Dataset page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
Downloaded 14 November 2026.

`bike_sharing_daily_raw.csv` is the original `day.csv`, unchanged. `bike_sharing_daily.csv` selects its daily date, total, casual and registered counts, renames total counts to rentals, and adds English weekday labels. No count values are changed, no spikes are inserted, and no timezone conversion is performed. The casual and registered series sum to the total; they are real membership dimensions, not invented devices. `real_data_source.json` records provenance and a raw-file SHA-256.

Inputs contain the previous 365 daily observations. Rental forecasts have 30-day horizons. The summer walkthrough starts 1 August 2012; a second window starts 1 April 2012 during spring growth. Each has 60 rolling origins. The saved membership-dimension cases remain available in the full suite. Future counts, weather and other observed future covariates are not supplied. This familiar public dataset may have appeared in model training, so the exercise is not an uncontaminated capability benchmark.

NOAA Global Monitoring Laboratory: **Mauna Loa monthly mean atmospheric CO₂**, https://gml.noaa.gov/ccgg/trends/data.html. Downloaded 14 November 2026 from https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv. NOAA government data are public domain unless otherwise marked; source acknowledgment: https://gml.noaa.gov/about/disclaimer.html.

`noaa_co2_monthly_raw.csv` preserves the downloaded file. `noaa_co2_monthly.csv` selects positive monthly means for 2000–2019, unchanged, with month-start dates. `noaa_co2_source.json` records provenance. This series supplies a persistent trend plus annual seasonality. Models receive 120 monthly observations and forecast 12 months; the walkthrough starts January 2019, with 12 rolling origins in the suite. Monthly values are never interpolated into daily data.

## Evaluation files

`model_cases.json` has 367 cases: extraction 20, direct arithmetic 20, one-shot counts plus arithmetic 20, count extraction 20, staged arithmetic 20, A/B chart representations 40, fact-only charts 20, effect estimates 15, summer rental forecasts 60, spring rental forecasts 60, membership forecasts 60, and monthly CO₂ forecasts 12. `expected_results.json` contains separate scoring targets. `examples.json` points to corresponding reference cases; English walkthrough prompts live directly in the notebook.

A staged case has a `from_counts` dependency. Its second state is built at runtime from that provider's extracted counts, never from hidden truth. The response cache stores that state and its input hash. The same four conversion/uplift questions and targets are used for direct, one-shot and staged calls. The arithmetic comparison scores only the four shared conversion/uplift fields. Failed pipeline outputs are scored as failures, not silently discarded from scalar accuracy.

`offline_baselines/` contains precomputed training-only reference estimates for the original rental and estimation cases. `snapshot.json` checksums the frozen pack. Overlapping time windows and multiple views of the same experiment are correlated; no independent-replicate significance claim is implied.

Tables show native Clairic scalar confidence. Luna's regular Responses output has no equivalent native per-field confidence. Estimate/forecast p10–p90 coverage is evaluated separately. No model results are fabricated in this folder.

Forecast comparisons report MAE, MAPE (%) and WAPE (%), alongside interval coverage and provider elapsed seconds. MAPE averages absolute percentage error over nonzero actuals; WAPE divides total absolute error by total absolute actuals. Scores in the full suite use valid predictions and report their count. Expected rows use zero point error and 80% nominal p10–p90 coverage. Two-stage timing includes both requests.

`chart_facts` adds 20 fact-only image cases (13 fields each): numbers and interval endpoints from the result dashboard, assignment dates from the timeline x-axis. No dates/numbers are given in the text, and no winner judgment is requested. The active notebook suite has 120 cases: readouts 20, counts 20, one-shot arithmetic 20, mixed A/B charts 40 and chart facts 20. Older estimation/forecast/other arithmetic cases remain in the frozen pack but are excluded from the active run. Timing charts use successful calls and compare matched cases for paired differences.
