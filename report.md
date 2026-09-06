# Lahore AQI Predictor — Final Project Report

## 1. Objective

The system predicts **Air Quality Index (AQI) for Lahore, Pakistan for the current day and the next 72 hours** using a serverless-style MLOps architecture. The fixed location is 31.5204° N, 74.3587° E (`Asia/Karachi`). The model is intentionally trained on **no more than the most recent six calendar months** of Lahore hourly data.

The project uses Open-Meteo for weather and CAMS atmospheric-composition data, Hopsworks for the Feature Store/Feature View/Model Registry, GitHub Actions for scheduled pipelines, Streamlit for the interactive dashboard, and FastAPI for an optional JSON endpoint.

## 2. System architecture

1. **Rolling historical backfill:** Calculate the date exactly six calendar months before the run date. Fetch archive data for the older portion and recent/live Open-Meteo past hours for the newest days/current day.
2. **Raw Feature Group:** Store Lahore weather, pollutant concentrations and historical AQI in `lahore_aqi_raw_6m`.
3. **Feature engineering:** Create AQI lags, change rates, rolling means, Lahore-local cyclic time features, and next-hour exogenous inputs.
4. **Engineered Feature Group:** Store supervised one-hour-ahead rows in `lahore_aqi_engineered_6m`.
5. **Feature View:** Expose the supervised training data as `lahore_aqi_training_view_6m`, with next-hour AQI as the label.
6. **Training:** Every run filters the Feature View to the latest six months, performs a chronological train/validation/test split, compares five models, evaluates RMSE/MAE/R², generates SHAP explanations, and registers the selected model.
7. **Inference:** Load the best Hopsworks model, obtain current/recent AQI and future weather/pollutant forecasts from Open-Meteo, recursively predict each future hour, and present current-day plus 72-hour results.

## 3. Data sources

### Open-Meteo Weather

Variables used: 2 m temperature, 2 m relative humidity, precipitation, surface pressure, and 10 m wind speed.

### Open-Meteo Air Quality / CAMS

Variables used: PM2.5, PM10, carbon monoxide, nitrogen dioxide, sulphur dioxide, ozone, and historical AQI.

Historical `us_aqi` is the supervised target. **Future Open-Meteo `us_aqi` is deliberately excluded from model inputs**, preventing target leakage. Future weather and pollutant forecasts are treated as exogenous predictors.

## 4. Six-month data policy

The project does not use years of historical data. At every training run it calculates a dynamic six-calendar-month boundary. For example, a run on 6 September uses data beginning 6 March. The setting is also clamped so `HISTORY_MONTHS` cannot exceed six.

Scheduled upserts may leave older physical rows in Hopsworks after the system has operated for many months, but all training and EDA reads apply the six-month event-time filter. Therefore the model never trains on data older than the requested window.

## 5. Feature engineering

The current AQI state contains:

- current AQI
- AQI lagged by 1, 3, 6 and 24 hours
- 1-hour and 3-hour AQI changes
- 6-hour and 24-hour rolling AQI means

Next-hour inputs include:

- temperature
- relative humidity
- precipitation
- surface pressure
- wind speed
- PM2.5
- PM10
- carbon monoxide
- nitrogen dioxide
- sulphur dioxide
- ozone
- cyclic Lahore-local hour/day-of-week/month features

The supervised label is AQI one hour in the future.

## 6. Model experiments

| Model | Purpose |
|---|---|
| Ridge Regression | linear/statistical baseline |
| Random Forest | nonlinear bagged-tree model |
| Histogram Gradient Boosting | boosted-tree model |
| XGBoost | regularized gradient-boosted tree model |
| MLP Regressor (64, 32) | neural-network baseline |

The split is chronological: 70% training, 15% validation and 15% newest held-out testing. The selected model minimizes validation RMSE, is retrained on training+validation, and is evaluated once on the final held-out period.

Actual results are generated into `artifacts/model_comparison.csv` and `artifacts/metrics.json`. They must be copied into the final submitted report after a real Hopsworks training run; no fabricated values are included here.

## 7. Current-day and 72-hour forecasting

The dashboard now explicitly handles the current calendar day. It combines the most recent Open-Meteo AQI values available since Lahore midnight with model predictions for the remaining hours of the day. It then continues forecasting hourly for the next 72 hours.

Daily cards summarize Today, Tomorrow and subsequent dates covered by the 72-hour horizon. Each card displays peak AQI, mean AQI, forecast-hour count and AQI health category.

## 8. Explainability

The training pipeline uses SHAP on a bounded held-out sample and creates `artifacts/shap_summary.png`. The image is copied into the Hopsworks registered model artifact when successfully generated, so it can be shown in the Streamlit Model & SHAP tab.

## 9. Hopsworks objects

```text
Raw Feature Group:        lahore_aqi_raw_6m v1
Engineered Feature Group: lahore_aqi_engineered_6m v1
Feature View:             lahore_aqi_training_view_6m v1
Model Registry model:     lahore_aqi_forecaster_6m
```

## 10. Automation

GitHub Actions runs:

- the recent feature pipeline every hour
- the training pipeline every day at 06:35 Pakistan time

The feature pipeline re-fetches a seven-day overlap and upserts it, which helps repair short outages. Credentials are stored as GitHub repository secrets.

## 11. Dashboard

The Streamlit application includes:

- current AQI
- today's recent-data + predicted timeline
- separate +24h / +48h / +72h (Day 1 / Day 2 / Day 3) forecast cards
- Today/Tomorrow/following-day summaries
- 72-hour forecast chart and detailed table
- six-month historical AQI view from Hopsworks
- unhealthy/very-unhealthy/hazardous alerts
- Hopsworks model/version and held-out metrics
- validation comparison including XGBoost
- Actual vs Predicted held-out test chart
- SHAP summary when available

## 12. Limitations

- Open-Meteo/CAMS is gridded model data rather than a Lahore regulatory sensor feed.
- Future pollutant/weather inputs are forecasts and have their own uncertainty.
- Recursive one-hour prediction can accumulate model error at longer horizons.
- Sudden event-driven pollution spikes such as crop burning or dust storms may remain difficult to predict.
- The six-month policy favors recent seasonal behavior but intentionally discards older seasonal cycles.

## 13. Reproducibility

```bash
python scripts_check_open_meteo.py
python scripts_check_hopsworks.py
python -m src.backfill
python -m src.eda
python -m src.train
streamlit run app.py
```

After the first successful run, GitHub Actions maintains the data and retrains the model automatically.
