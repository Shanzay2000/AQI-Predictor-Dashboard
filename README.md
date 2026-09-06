# Lahore AQI Predictor — Final Open-Meteo + Hopsworks Project

This is the final Lahore-only version of the AQI project. It predicts **AQI for Lahore, Pakistan for the current day and the next 72 hours** using a fully automated Open-Meteo + Hopsworks MLOps pipeline.

The model is deliberately trained on **no more than the latest six calendar months** of hourly Lahore data. The training code re-computes that six-month boundary every time it runs, so older rows are ignored even if the Feature Store has been operating for longer.

## Fixed location

```text
Lahore, Pakistan
Latitude: 31.5204
Longitude: 74.3587
Timezone: Asia/Karachi (UTC+5)
```

## What the dashboard shows

- latest/current Lahore AQI
- **today's AQI timeline**: recent Open-Meteo hours so far + predictions for the remaining hours
- calendar-day cards for **Today, Tomorrow, and following forecast days**
- next 72-hour hourly AQI forecast
- daily mean/peak AQI and health category
- PM2.5, PM10, ozone, weather and other exogenous forecast context
- hazardous/unhealthy AQI alerts
- latest six-month historical AQI chart from Hopsworks
- registered model version, RMSE, MAE and R²
- validation comparison including XGBoost
- held-out Actual vs Predicted chart
- SHAP feature-importance image when training generated it

## Data and target

Open-Meteo supplies:

**Weather:** temperature, humidity, precipitation, surface pressure, wind speed.

**Air quality:** PM2.5, PM10, carbon monoxide, nitrogen dioxide, sulphur dioxide, ozone, and historical AQI.

Historical `us_aqi` is the supervised target. The model **does not use Open-Meteo's future `us_aqi`**. For future hours it receives Open-Meteo's weather/pollutant forecasts and generates its own AQI predictions.

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

Read **`SETUP_LAHORE.md`** for every click and command. The short version is:

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# copy .env.example to .env, then fill HOPSWORKS_API_KEY and HOPSWORKS_PROJECT
python scripts_check_open_meteo.py
python scripts_check_hopsworks.py
python -m src.backfill
python -m src.eda
python -m src.train
streamlit run app.py
```

Optional API:

```bash
uvicorn api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/forecast?hours=72
```

## Automation

- `.github/workflows/feature_pipeline.yml`: hourly, re-fetches a 7-day overlap and upserts it so short outages repair themselves.
- `.github/workflows/training_pipeline.yml`: daily at 06:35 Pakistan time and **always trains on only the latest six months**.

## Tests

```bash
python -m compileall src app.py api.py
pytest -q
```

The supplied final project passes its local unit tests without needing Hopsworks credentials.

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

These training artifacts are also copied into the registered Hopsworks model artifact where available, allowing the Streamlit model tab to display SHAP after deployment.
