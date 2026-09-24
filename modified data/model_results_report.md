# Bluebikes Demand Prediction - Model Comparison Report

## Summary

This report compares the performance of different machine learning models for predicting Bluebikes trip counts.

**Best Model (by OSR²):** XGBoost (Grid Search Best) (OSR² = 0.7693)

## Model Performance Comparison

| Model | Train R² | Test R² (OSR²) | Train RMSE | Test RMSE | Best Parameters |
|-------|----------|---------------|------------|----------|-----------------|
| XGBoost (Grid Search Best) | 0.8731 | 0.7693 | 3.55 | 4.80 | {'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 300} |
| XGBoost | 0.8725 | 0.7667 | 3.56 | 4.83 | {'max_depth': 7} |
| RandomForest | 0.9652 | 0.7480 | 1.86 | 5.02 | {'n_estimators': 150} |
| GBM | 0.6616 | 0.6486 | 5.80 | 5.92 | {'learning_rate': 0.1} |
| Lasso | 0.6487 | 0.6427 | 5.91 | 5.97 | {'alpha': 0.0001} |
| Ridge | 0.6487 | 0.6427 | 5.91 | 5.97 | {'alpha': 0.0001} |
| OLS Regression | 0.6487 | 0.6427 | 5.91 | 5.97 | None |
| Linear Regression | 0.6487 | 0.6427 | 5.91 | 5.97 | None |

## Significant Features / Top Features by Model

### XGBoost (Grid Search Best)

- **Significant Features:** N/A (tree model - use feature importance)

- **Top 10 Features:** hour_4, hour_3, hour_6, hour_17, hour_18, hour_5, hour_2, hour_1, hour_19, hour_16

---

### XGBoost

- **Significant Features:** N/A (tree model - use feature importance)

- **Top 10 Features:** hour_17, hour_6, hour_5, hour_2, hour_3, hour_18, hour_1, hour_4, hour_16, hour_19

---

### RandomForest

- **Significant Features:** N/A (tree model - use feature importance)

- **Top 10 Features:** TAVG, hour_17, hour_18, hour_16, hour_19, PRCP, hour_15, station_Harvard Square at Mass Ave/ Dunster, hour_14, hour_13

---

### GBM

- **Significant Features:** N/A (tree model - use feature importance)

- **Top 10 Features:** TAVG, hour_17, hour_18, hour_16, hour_19, hour_15, hour_4, hour_3, hour_5, hour_2

---

### Lasso

- **Significant Features:** N/A (regularized model)

- **Top 10 Features:** N/A (regularized model)

---

### Ridge

- **Significant Features:** N/A (regularized model)

- **Top 10 Features:** N/A (regularized model)

---

### OLS Regression

- **Significant Features:** hour_12 (p=0.0000), hour_20 (p=0.0000), hour_19 (p=0.0000), hour_18 (p=0.0000), hour_17 (p=0.0000), hour_16 (p=0.0000), hour_15 (p=0.0000), hour_14 (p=0.0000), hour_13 (p=0.0000), station_Harvard Square at Mass Ave/ Dunster (p=0.0000), hour_11 (p=0.0000), hour_21 (p=0.0000), station_Central Square at Mass Ave / Essex St (p=0.0000), hour_10 (p=0.0000), week_40 (p=0.0000), week_42 (p=0.0000), week_37 (p=0.0000), hour_9 (p=0.0000), hour_8 (p=0.0000), week_36 (p=0.0000), week_44 (p=0.0000), week_39 (p=0.0000), week_43 (p=0.0000), week_38 (p=0.0000), week_47 (p=0.0000), week_41 (p=0.0000), PRCP (p=0.0000), week_45 (p=0.0000), week_46 (p=0.0000), week_35 (p=0.0000), week_17 (p=0.0000), hour_22 (p=0.0000), week_30 (p=0.0000), week_19 (p=0.0000), week_16 (p=0.0000), week_18 (p=0.0000), week_20 (p=0.0000), week_15 (p=0.0000), TAVG (p=0.0000), week_6 (p=0.0000), week_10 (p=0.0000), week_9 (p=0.0000), week_21 (p=0.0000), week_34 (p=0.0000), week_22 (p=0.0000), week_8 (p=0.0000), week_29 (p=0.0000), week_49 (p=0.0000), week_11 (p=0.0000), week_14 (p=0.0000), week_50 (p=0.0000), week_5 (p=0.0000), week_7 (p=0.0000), week_32 (p=0.0000), week_4 (p=0.0000), week_33 (p=0.0000), hour_4 (p=0.0000), week_31 (p=0.0000), week_12 (p=0.0000), week_26 (p=0.0000), hour_3 (p=0.0000), week_13 (p=0.0000), day_Saturday (p=0.0000), week_24 (p=0.0000), week_28 (p=0.0000), hour_5 (p=0.0000), day_Friday (p=0.0000), week_27 (p=0.0000), week_23 (p=0.0000), hour_2 (p=0.0000), hour_23 (p=0.0000), week_2 (p=0.0000), day_Sunday (p=0.0000), week_48 (p=0.0000), extreme_weather (p=0.0000), week_3 (p=0.0000), hour_6 (p=0.0000), week_25 (p=0.0000), week_51 (p=0.0004), hour_1 (p=0.0021), hour_7 (p=0.0036), SNOW (p=0.0076), week_52 (p=0.0097), event_Hockey (p=0.0498)

- **Top 10 Features:** hour_12, hour_20, hour_19, hour_18, hour_17, hour_16, hour_15, hour_14, hour_13, station_Harvard Square at Mass Ave/ Dunster

---

### Linear Regression

- **Significant Features:** N/A (no p-values from sklearn)

- **Top 10 Features:** hour_17, hour_18, hour_16, hour_19, hour_15, hour_14, hour_13, hour_12, hour_20, week_37

---
