"""
Full-year Bluebikes demand analysis, following the approach in bluebikes_top3_opt.py.

Aggregates hourly outbound trips for three stations, joins weather and event
features, then compares Linear/OLS/Ridge/Lasso, Random Forest, GBM and XGBoost.

Outputs (written to DATA_DIR):
- full_year_agg_outbound.csv    aggregated hourly data, used by the optimization model
- full_year_prediction_plot.png actual vs predicted trips per station
- model_results_summary.csv     metrics for every model
- model_results_report.md       the same results as a Markdown report
"""

import itertools

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

DATA_DIR = 'modified data'
RANDOM_STATE = 42

STATIONS = [
    "MIT at Mass Ave / Amherst St",
    "Harvard Square at Mass Ave/ Dunster",
    "Central Square at Mass Ave / Essex St",
]
WEATHER_COLS = ['PRCP', 'SNOW', 'TAVG']
EVENT_COLS = ['event_Basketball', 'event_Comedy', 'event_Concert', 'event_Fair', 'event_Hockey']


def section(title, char="="):
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def add_time_features(df, date_col):
    df['day_of_week'] = df[date_col].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['month'] = df[date_col].dt.month_name()
    return df


def evaluate(model, name, best_params='None', significant='N/A', top_features='N/A',
             clip=False, add_const=False):
    """Score a fitted model on the train/test split and record the result."""
    Xtr, Xte = (sm.add_constant(X_train), sm.add_constant(X_test)) if add_const else (X_train, X_test)
    y_train_pred = model.predict(Xtr)
    y_test_pred = model.predict(Xte)
    if clip:
        y_train_pred = np.maximum(y_train_pred, 0)
        y_test_pred = np.maximum(y_test_pred, 0)

    result = {
        'Model': name,
        'Train_R2': r2_score(y_train, y_train_pred),
        'Test_R2': r2_score(y_test, y_test_pred),
        'Train_RMSE': rmse(y_train, y_train_pred),
        'Test_RMSE': rmse(y_test, y_test_pred),
        'Best_Params': best_params,
        'Significant_Features': significant,
        'Top_Features': top_features,
    }
    result['OSR2'] = result['Test_R2']  # out-of-sample R²
    all_model_results.append(result)

    print(f"Train RMSE: {result['Train_RMSE']:.3f} | Train R²: {result['Train_R2']:.3f}")
    print(f"Test  RMSE: {result['Test_RMSE']:.3f} | Test  R²: {result['Test_R2']:.3f}")
    return result


def top_importances(model, k=10):
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return ', '.join(importance.nlargest(k).index)


def tune_and_evaluate(name, base_model, param_name, values, n_jobs=-1):
    """Pick one hyperparameter by 5-fold CV R², refit on the training set, then evaluate."""
    print(f"\n===== {name} =====")
    cv_scores = {}
    for val in values:
        model = clone(base_model).set_params(**{param_name: val})
        cv_scores[val] = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=n_jobs).mean()

    print(pd.DataFrame({'Parameter': [f"{param_name}={v}" for v in cv_scores],
                        'CV_R2': list(cv_scores.values())}).round(3))
    best_val = max(cv_scores, key=cv_scores.get)
    best_param = {param_name: best_val}
    print(f"\n{name} best parameter: {best_param} (CV R²={cv_scores[best_val]:.3f})")

    best_model = clone(base_model).set_params(**best_param).fit(X_train, y_train)
    if hasattr(best_model, 'feature_importances_'):
        significant = 'N/A (tree model - use feature importance)'
        top_features = top_importances(best_model)
    else:
        significant = top_features = 'N/A (regularized model)'
    evaluate(best_model, name, str(best_param), significant, top_features)


# ---------------------------------------------------------------------------
section("Step 1: Load data")

df = pd.read_csv(f'{DATA_DIR}/merged_all_months.csv')
print(f"Data shape: {df.shape}")

# ---------------------------------------------------------------------------
section("Step 2: Aggregate outbound trips by start station")

df['started_at'] = pd.to_datetime(df['started_at'], format='mixed', errors='coerce')
df['date'] = df['started_at'].dt.date

df_f = df[df['start_station_name'].isin(STATIONS)].copy()
print(f"Filtered data shape: {df_f.shape}")
print(f"Stations: {df_f['start_station_name'].unique()}")

# Weather is constant within an hour, so the mean just carries it through;
# event flags are 0/1, so max marks the hour if any trip fell on an event day.
agg = {'ride_id': 'count', **{c: 'mean' for c in WEATHER_COLS + ['extreme_weather']},
       **{c: 'max' for c in EVENT_COLS}}
outbound = (
    df_f.groupby(['start_station_name', 'date', 'hour'])
        .agg(agg)
        .reset_index()
        .rename(columns={'ride_id': 'trip_count'})
)
print(f"\nAggregated data shape: {outbound.shape}")
print(outbound.head(20))

# Build the full station × date × hour grid so hours with no trips count as 0
stations = outbound['start_station_name'].unique()
dates = outbound['date'].unique()
hours = sorted(outbound['hour'].unique())
print(f"\nStations: {len(stations)} | Dates: {len(dates)} | Hours: {len(hours)} | "
      f"Grid size: {len(stations) * len(dates) * len(hours)}")

full_grid = pd.MultiIndex.from_product(
    [stations, dates, hours],
    names=['start_station_name', 'date', 'hour']
).to_frame(index=False)

merged = full_grid.merge(outbound, on=['start_station_name', 'date', 'hour'], how='left')

merged['trip_count'] = merged['trip_count'].fillna(0)
# Hours without trips have no weather reading: carry the last known value forward
merged[WEATHER_COLS] = merged[WEATHER_COLS].ffill().fillna(0)
merged[['extreme_weather'] + EVENT_COLS] = merged[['extreme_weather'] + EVENT_COLS].fillna(0)

merged['date'] = pd.to_datetime(merged['date'])
merged = add_time_features(merged, 'date')

print(f"\nMerged data shape: {merged.shape}")
print(merged.head(10))

output_file = f'{DATA_DIR}/full_year_agg_outbound.csv'
merged.to_csv(output_file, index=False)
print(f"\nSaved aggregated data to: {output_file}")

# ---------------------------------------------------------------------------
section("Step 3: One-hot encode categorical variables")

complete_df = pd.get_dummies(
    merged,
    columns=['start_station_name', 'day_of_week', 'week_of_year', 'month', 'hour'],
    prefix=['station', 'day', 'week', 'month', 'hour'],
    dtype=int
)
print(f"Encoded data shape: {complete_df.shape}")
print(complete_df.head(10))

all_model_results = []

# ---------------------------------------------------------------------------
section("Step 4: Train and evaluate models")

y = complete_df['trip_count']
X = complete_df.drop(columns=['trip_count', 'date', 'is_weekend'], errors='ignore')

# Month dummies are collinear with week-of-year, so drop them
X = X.drop(columns=[c for c in X.columns if c.startswith('month_')])

# Drop one baseline level per categorical variable to avoid perfect collinearity
baseline_dummies = [
    'day_Monday',
    'station_MIT at Mass Ave / Amherst St',
    'week_1',
    'hour_0',
]
X = X.drop(columns=baseline_dummies, errors='ignore')
print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

section("4.1 Linear regression", "-")

lr = LinearRegression().fit(X_train, y_train)
coef_df = pd.Series(lr.coef_, index=X.columns).sort_values(ascending=False)
print("Top 10 positive coefficients:")
print(coef_df.head(10))
print("\nTop 10 negative coefficients:")
print(coef_df.tail(10))
evaluate(lr, 'Linear Regression',
         significant='N/A (no p-values from sklearn)',
         top_features=', '.join(coef_df.head(10).index))

print("\nOLS regression with statsmodels:")
ols_model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(ols_model.summary())

pvalues = ols_model.pvalues.drop('const')
significant_features = pvalues[pvalues < 0.05].sort_values()
if len(significant_features):
    significant_str = ', '.join(f"{feat} (p={pval:.4f})" for feat, pval in significant_features.items())
    top_str = ', '.join(significant_features.head(10).index)
else:
    significant_str, top_str = 'None', 'N/A'
evaluate(ols_model, 'OLS Regression', significant=significant_str, top_features=top_str, add_const=True)

section("4.2 Ridge and Lasso regression", "-")

tune_and_evaluate('Ridge', Ridge(), 'alpha', [0.0001, 0.001, 0.01, 0.1, 1, 5, 10])
tune_and_evaluate('Lasso', Lasso(max_iter=5000), 'alpha', [0.0001, 0.001, 0.01, 0.1])

section("4.3 Tree models (Random Forest, GBM, XGBoost)", "-")

tree_models = [
    ('RandomForest', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
     'n_estimators', [50, 100, 150]),
    ('GBM', GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=100),
     'learning_rate', [0.05, 0.1]),
    ('XGBoost', xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, n_estimators=100),
     'max_depth', [3, 5, 7]),
]
for name, model, param_name, values in tqdm(tree_models, desc="Tree models"):
    tune_and_evaluate(name, model, param_name, values)

section("4.4 XGBoost grid search", "-")

param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators': [100, 150, 200, 300],
}

results = []
for depth, lr_, n_est in tqdm(list(itertools.product(*param_grid_xgb.values())), desc="XGBoost grid search"):
    params = {'max_depth': depth, 'learning_rate': lr_, 'n_estimators': n_est}
    model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=1).mean()

    model.fit(X_train, y_train)
    results.append({
        **params,
        'Train_R2': r2_score(y_train, np.maximum(model.predict(X_train), 0)),
        'CV_R2': cv_r2,
        'Test_R2': r2_score(y_test, np.maximum(model.predict(X_test), 0)),
    })

results_df = pd.DataFrame(results).sort_values('CV_R2', ascending=False).reset_index(drop=True)
print("\n=== XGBoost grid search results ===")
print(results_df.round(5).to_string())

best = results_df.iloc[0]
best_xgb_params = {
    'max_depth': int(best['max_depth']),
    'learning_rate': float(best['learning_rate']),
    'n_estimators': int(best['n_estimators']),
}
print(f"\nBest parameters: {best_xgb_params}")

best_xgb = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_xgb_params).fit(X_train, y_train)
evaluate(best_xgb, 'XGBoost (Grid Search Best)', str(best_xgb_params),
         'N/A (tree model - use feature importance)', top_importances(best_xgb), clip=True)

# ---------------------------------------------------------------------------
section("Step 5: Plot predictions")

# Refit on all data for the in-sample visualization
final_model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_xgb_params).fit(X, y)

plot_df = complete_df.copy()
plot_df['predicted'] = np.maximum(final_model.predict(X), 0)
plot_df['actual'] = plot_df['trip_count']

# Recover the station name from the one-hot columns
station_cols = [c for c in plot_df.columns if c.startswith('station_')]
plot_df['station'] = plot_df[station_cols].idxmax(axis=1).str.replace('station_', '')

# Smooth over 24 hours to make the yearly trend readable
plot_df = plot_df.sort_values('date')
plot_df['actual_smooth'] = plot_df['actual'].rolling(24, center=True).mean()
plot_df['predicted_smooth'] = plot_df['predicted'].rolling(24, center=True).mean()

plot_stations = plot_df['station'].unique()
n = len(plot_stations)
plt.figure(figsize=(14, 3.5 * n))

for i, station in enumerate(plot_stations, 1):
    subset = plot_df[plot_df['station'] == station]
    plt.subplot(n, 1, i)
    sns.lineplot(data=subset, x='date', y='actual_smooth',
                 color='tab:blue', label='Actual', linewidth=1.5, errorbar=None)
    sns.lineplot(data=subset, x='date', y='predicted_smooth',
                 color='tab:orange', label='Predicted', linewidth=1.5, errorbar=None, alpha=0.7)

    plt.title(station)
    plt.ylabel("Trip Count")
    plt.xlabel(None)
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(rotation=45, ha='right')
    ax.grid(False)

plt.suptitle("Actual vs Predicted Trip Counts by Station (Full Year Data, Time Smoothed)",
             fontsize=14, y=0.95)
plt.xlabel("Time (by Month)")
plt.tight_layout(rect=[0, 0, 1, 0.94])

output_plot = f'{DATA_DIR}/full_year_prediction_plot.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"Saved visualization to: {output_plot}")
plt.close()

plot_df['residual'] = plot_df['actual'] - plot_df['predicted']
cols_to_show = ['date', 'hour', 'station', 'actual', 'predicted', 'residual']
cols_to_show = [c for c in cols_to_show if c in plot_df.columns]

print("\n=== Top 20 Underpredicted Cases (Actual >> Predicted) ===")
print(plot_df.nlargest(20, 'residual')[cols_to_show])
print("\n=== Top 20 Overpredicted Cases (Predicted >> Actual) ===")
print(plot_df.nsmallest(20, 'residual')[cols_to_show])

# ---------------------------------------------------------------------------
section("Step 6: Summarize model results")

results_summary_df = pd.DataFrame(all_model_results).sort_values('OSR2', ascending=False).reset_index(drop=True)
print("All models, sorted by OSR²:")
print(results_summary_df.to_string(index=False))

summary_csv_path = f'{DATA_DIR}/model_results_summary.csv'
results_summary_df.to_csv(summary_csv_path, index=False)
print(f"\nSaved model summary to: {summary_csv_path}")

best_row = results_summary_df.iloc[0]
lines = [
    "# Bluebikes Demand Prediction - Model Comparison Report",
    "",
    "## Summary",
    "",
    "This report compares the performance of different machine learning models for predicting Bluebikes trip counts.",
    "",
    f"**Best Model (by OSR²):** {best_row['Model']} (OSR² = {best_row['OSR2']:.4f})",
    "",
    "## Model Performance Comparison",
    "",
    "| Model | Train R² | Test R² (OSR²) | Train RMSE | Test RMSE | Best Parameters |",
    "|-------|----------|---------------|------------|----------|-----------------|",
]
for _, row in results_summary_df.iterrows():
    lines.append(f"| {row['Model']} | {row['Train_R2']:.4f} | {row['OSR2']:.4f} | "
                 f"{row['Train_RMSE']:.2f} | {row['Test_RMSE']:.2f} | {row['Best_Params']} |")

lines += ["", "## Significant Features / Top Features by Model", ""]
for _, row in results_summary_df.iterrows():
    lines += [f"### {row['Model']}", "",
              f"- **Significant Features:** {row['Significant_Features']}", "",
              f"- **Top 10 Features:** {row['Top_Features']}", "",
              "---", ""]

markdown_path = f'{DATA_DIR}/model_results_report.md'
with open(markdown_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print(f"Saved Markdown report to: {markdown_path}")

section("Analysis Complete!")
