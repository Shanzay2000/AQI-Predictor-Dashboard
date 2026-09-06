# Lahore AQI Predictor — Final Open-Meteo + Hopsworks Project

**🔗 Live app:** [aqi-predictor-dashboard-shanzay.streamlit.app](https://aqi-predictor-dashboard-shanzay.streamlit.app/)
**Live demo:** [Live-demo-compress.mp4](./Live-demo-compress.mp4)

This is the final, Lahore-only version of the AQI project. It predicts **AQI for Lahore, Pakistan for the current day and the next 72 hours**, using a fully automated Open-Meteo + Hopsworks MLOps pipeline.

The model is deliberately trained on **no more than the latest six calendar months** of hourly Lahore data. The training code re-computes that six-month boundary every time it runs, so older rows are ignored even if the feature store has been operating for longer — this keeps the project comfortably inside Hopsworks' free-tier storage limit while still giving the models enough history to learn seasonal and daily patterns.

## Fixed location

```text
Lahore, Pakistan
Latitude:  31.5204
Longitude: 74.3587
Timezone:  Asia/Karachi (UTC+5)
```

## What the dashboard shows

- Latest / current Lahore AQI
- **Today's AQI timeline**: recent Open-Meteo observations so far + predictions for the remaining hours
- Calendar-day cards for **Today, Tomorrow, and the following forecast days**
- Next 72-hour hourly AQI forecast
- Daily mean / peak AQI and health category
- PM2.5, PM10, ozone, weather, and other exogenous forecast context
- Hazardous / unhealthy AQI alerts
- Latest six-month historical AQI chart, read from Hopsworks
- Registered model version, RMSE, MAE, and R²
- Validation comparison across candidate models, including XGBoost
- Held-out actual-vs-predicted chart
- SHAP feature-importance image, when training generated one

## Data and target

Open-Meteo supplies:

- **Weather:** temperature, humidity, precipitation, surface pressure, wind speed
- **Air quality:** PM2.5, PM10, carbon monoxide, nitrogen dioxide, sulphur dioxide, ozone, and historical AQI

Historical `us_aqi` is the supervised target. The model **does not use Open-Meteo's future `us_aqi`** — that would be target leakage. For future hours, it receives only Open-Meteo's weather/pollutant *forecasts* and generates its own AQI predictions from those.

## Hopsworks architecture

```text
Open-Meteo
   |
   +--> last 6 months raw hourly Lahore data
   |            |
   |            v
   |      Hopsworks Feature Store
   |      lahore_aqi_raw_6m
   |            |
   |      feature engineering
   |            v
   |      lahore_aqi_engineered_6m
   |            |
   |            v
   |      Hopsworks Feature View
   |      lahore_aqi_training_view_6m
   |            |
   |            v
   |      daily model comparison
   |      Ridge / Random Forest /
   |      HistGradientBoosting / XGBoost / MLP
   |            |
   |            v
   |      Hopsworks Model Registry
   |      lahore_aqi_forecaster_6m
   |            |
   +--> future weather/pollutants
                |
                v
          Streamlit + FastAPI
          Today + next 72 hours
```

## First-time run

Read **`SETUP_LAHORE.md`** for every click and command. The short version:

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
# copy .env.example to .env, then fill in HOPSWORKS_API_KEY and HOPSWORKS_PROJECT

python scripts_check_open_meteo.py
python scripts_check_hopsworks.py
python -m src.backfill
python -m src.eda
python -m src.train
streamlit run app.py
```

Then open:

```text
http://127.0.0.1:8000/forecast?hours=72
```

## Automation

| Workflow | Schedule | What it does |
|---|---|---|
| `.github/workflows/feature_pipeline.yml` | Hourly | Re-fetches a 7-day overlap and upserts it, so short outages repair themselves automatically. |
| `.github/workflows/training_pipeline.yml` | Daily, 06:35 Pakistan time | Retrains on **only the latest six months** and registers the best model. |


## Generated artifacts after training

```text
artifacts/eda/aqi_timeseries.png
artifacts/eda/aqi_daily_mean.png
artifacts/eda/correlation_matrix.png
artifacts/model_comparison.csv
artifacts/metrics.json
artifacts/test_predictions.csv
artifacts/test_predictions.png
artifacts/shap_summary.png
```

These training artifacts are also copied into the registered Hopsworks model artifact where available, so the Streamlit model tab can display SHAP after deployment.
