# Bluebikes Demand Prediction & Dock Capacity Optimization

A course project that predicts hourly Bluebikes demand at three busy Cambridge stations and uses an hourly inventory model to decide where to add or remove docks within a fixed budget.

**Stations studied:** MIT at Mass Ave / Amherst St · Harvard Square at Mass Ave / Dunster · Central Square at Mass Ave / Essex St

## Approach

1. **Data preparation.** Monthly Bluebikes trip data were cleaned and joined with daily weather (precipitation, snow, average temperature) and TD Garden event dates grouped by type (basketball, hockey, concert, comedy, fair).
2. **Demand prediction.** Hourly outbound trips per station were modeled with OLS, Linear/Ridge/Lasso regression, Random Forest, Gradient Boosting and XGBoost, using hour, day of week, week of year, weather and event features.
3. **Capacity optimization.** A mixed-integer program built in Pyomo tracks how many bikes are at each station every hour of the year and picks how many docks to add or remove to maximize completed rentals, subject to a $200,000 budget. It uses the historical 2024 hourly trip counts as demand.

## Results

**Prediction.** XGBoost after grid search performed best, with **test R² = 0.77** and test RMSE = 4.80 trips/hour. Hour of day dominated the XGBoost feature importance; in Random Forest and Gradient Boosting, average temperature ranked first.

| Model | Test R² | Test RMSE |
|---|---|---|
| XGBoost (grid search) | 0.769 | 4.80 |
| Random Forest | 0.748 | 5.02 |
| Gradient Boosting | 0.649 | 5.92 |
| Linear Regression | 0.643 | 5.97 |

Full comparison: [`modified data/model_results_report.md`](modified%20data/model_results_report.md)

**Optimization.** Under the default assumptions, the model recommends adding **8 docks at Central Square, 5 at Harvard Square and 4 at MIT**, costing $67,099 (34% of the budget). With these docks, 67.7% of 2024 outbound demand can be served.

![Results summary](modified%20data/results_summary_table.png)

### Model formulation

- **Decision variables:** docks added `add[s]` and removed `remove[s]` (integers), new capacity `capacity[s]`, bikes at the station `inv[s,t]`, rentals `x_out[s,t]` and returns `x_in[s,t]` for every hour t
- **Objective:** maximize Σ x_out[s,t] − λ · total cost, with λ = 0.2
- **Inventory:** inv[s,t+1] = inv[s,t] + x_in[s,t] − x_out[s,t]; every station starts half full; inv[s,t] ≤ capacity[s]
- **Rentals and returns:** x_out ≤ outbound demand and x_out ≤ bikes available; x_in ≤ inbound demand and x_in ≤ empty docks
- **Capacity and budget:** capacity[s] = old_capacity[s] + add[s] − remove[s]; total cost ≤ $200,000
- **Assumptions:** current docks MIT 31, Harvard 19, Central 19; $3,947 to add a dock ($75,000 per 19-dock station), removing a dock saves $200; returns are estimated as 80% of rentals because the data only records trip starts

## Repository layout

| File | Purpose |
|---|---|
| `add_event_type_features.py` | Adds TD Garden event-type features to the monthly data |
| `bluebikes_top3_opt.py` | Initial exploration and modeling (originally written in Google Colab) |
| `analyze_full_year_data.py` | Full-year feature engineering and model comparison; writes `full_year_agg_outbound.csv` |
| `bluebikes_optimization_model.py` | Pyomo capacity optimization and result charts |
| `weather data.csv`, `tdgarden_events_with_type.csv` | Raw weather and event data |
| `modified data/` | Cleaned monthly data (`clean_data_1`–`12.csv`), model outputs and figures |

See [`OPTIMIZATION_MODEL_README.md`](OPTIMIZATION_MODEL_README.md) for details on the optimization module.

## How to run

```bash
pip install pandas numpy scikit-learn xgboost statsmodels pyomo matplotlib seaborn tqdm
brew install glpk          # LP solver used by Pyomo
```

See [`INSTALL.md`](INSTALL.md) for other ways to install the solver.

`merged_all_months.csv` (105 MB) exceeds GitHub's file size limit and is not included. It is simply the 12 monthly files combined, so you can rebuild it:

```bash
cd "modified data"
head -1 clean_data_1.csv > merged_all_months.csv
for i in $(seq 1 12); do tail -n +2 clean_data_$i.csv >> merged_all_months.csv; done
```

Then run the two main steps:

```bash
python analyze_full_year_data.py          # prediction models
python bluebikes_optimization_model.py    # capacity optimization
```

## Data sources

- Trip data: [Bluebikes System Data](https://bluebikes.com/system-data), used under the Bluebikes data license agreement
- Weather: NOAA daily observations for Boston
- Events: TD Garden public event schedule
