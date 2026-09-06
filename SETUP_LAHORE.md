# Complete setup guide — Lahore AQI Predictor

This guide assumes a fresh setup. Follow the steps in order.

## 1. What you are building

The project uses:

- **Open-Meteo** for Lahore weather + air-quality data
- **Hopsworks Feature Store** for raw and engineered features
- **Hopsworks Feature View** for the training dataset
- **Hopsworks Model Registry** for the selected model
- **GitHub Actions** for hourly ingestion and daily retraining
- **Streamlit** for the dashboard
- **FastAPI** for an optional JSON API

The project trains on **only the latest six calendar months**. It predicts the **current day and the next 72 hours**.

## 2. Create the correct Hopsworks project

When Hopsworks shows project/demo choices, **do not choose the Time Series Prediction Dashboard demo**.

Choose **Create new project / blank project**.

Suggested project name:

```text
lahore_aqi_predictor
```

The Python code creates the Feature Groups, Feature View and Model Registry entries automatically.

## 3. Create a Hopsworks API key

In Hopsworks, open your user/account settings and create a new API key. Give it a name such as:

```text
lahore-aqi-project
```

For the external Python client, enable the normal project/feature-store scopes required by your Hopsworks account. If the UI offers the standard scopes used by Hopsworks Python integrations, include:

```text
featurestore
project
job
kafka
```

Copy the key immediately and keep it private.

## 4. Install Python

Use Python **3.11** if possible.

Verify:

```bash
python --version
pip --version
```

## 5. Extract the project and create a virtual environment

### Windows PowerShell

```powershell
cd path\to\lahore_aqi_predictor_openmeteo_hopsworks
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
cd path/to/lahore_aqi_predictor_openmeteo_hopsworks
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Create `.env`

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### macOS/Linux

```bash
cp .env.example .env
```

Open `.env` and change:

```text
HOPSWORKS_API_KEY=PASTE_YOUR_REAL_KEY_HERE
HOPSWORKS_PROJECT=lahore_aqi_predictor
```

Leave the Lahore values as supplied:

```text
CITY_NAME=Lahore
COUNTRY_NAME=Pakistan
LOCATION_NAME=Lahore, Pakistan
LATITUDE=31.5204
LONGITUDE=74.3587
LOCAL_TIMEZONE=Asia/Karachi
HISTORY_MONTHS=6
RECENT_OVERLAP_DAYS=7
FORECAST_HOURS=72
```

The code clamps `HISTORY_MONTHS` to a maximum of **6**, so the model cannot accidentally train on more history.

The final Hopsworks object names are:

```text
Raw Feature Group:        lahore_aqi_raw_6m
Engineered Feature Group: lahore_aqi_engineered_6m
Feature View:             lahore_aqi_training_view_6m
Model Registry model:     lahore_aqi_forecaster_6m
```

## 7. Check Open-Meteo

Run:

```bash
python scripts_check_open_meteo.py
```

You should see recent Lahore rows, a latest AQI value, and future exogenous rows. Open-Meteo does not require an API key for the normal endpoints used by this project.

## 8. Check Hopsworks

Run:

```bash
python scripts_check_hopsworks.py
```

Expected form:

```text
Hopsworks login PASSED
Project: lahore_aqi_predictor
Raw Feature Group will be: lahore_aqi_raw_6m
Engineered Feature Group will be: lahore_aqi_engineered_6m
Feature View will be: lahore_aqi_training_view_6m
Model Registry model will be: lahore_aqi_forecaster_6m
```

If login fails, first check the API key and exact project name.

## 9. Run the six-month backfill

Run:

```bash
python -m src.backfill
```

The start date is calculated dynamically. For example, if the run date is **6 September 2026**, the earliest date fetched is **6 March 2026**. A different run date automatically produces the corresponding six-month boundary.

The backfill combines:

- Open-Meteo archive data for the older part of the six-month window
- recent/live Open-Meteo past hours for the newest days and the current day

This avoids relying on the historical archive for the newest hours.

After it completes, open Hopsworks → **Feature Store**. You should see:

```text
lahore_aqi_raw_6m
lahore_aqi_engineered_6m
```

You should also see the Feature View:

```text
lahore_aqi_training_view_6m
```

The raw group keeps provenance/EDA data. The engineered group contains one-hour-ahead supervised rows with:

- current AQI
- 1h / 3h / 6h / 24h AQI lags
- 1h / 3h AQI change
- 6h / 24h rolling means
- next-hour weather/pollutant inputs
- Lahore-local cyclical time features
- target: next-hour `us_aqi`

## 10. Run EDA

```bash
python -m src.eda
```

Outputs:

```text
artifacts/eda/aqi_timeseries.png
artifacts/eda/aqi_daily_mean.png
artifacts/eda/correlation_matrix.png
```

EDA is also restricted to the latest six months.

## 11. Train and register the model

```bash
python -m src.train
```

The training pipeline reads through the **Hopsworks Feature View** and filters the event-time range to the latest six months.

It compares:

1. Ridge Regression
2. Random Forest
3. Histogram Gradient Boosting
4. XGBoost regressor
5. MLP neural network with two hidden layers

The split is chronological:

```text
oldest 70% -> training
next 15%   -> validation/model selection
newest 15% -> final held-out test
```

Model selection uses validation RMSE. The final selected model is evaluated with:

```text
RMSE
MAE
R²
```

Generated files:

```text
artifacts/model_comparison.csv
artifacts/metrics.json
artifacts/test_predictions.csv
artifacts/test_predictions.png
artifacts/shap_summary.png
```

The selected model is registered as:

```text
lahore_aqi_forecaster_6m
```

Open Hopsworks → **Model Registry** and verify a model version appears with RMSE/MAE/R².

## 12. Run the dashboard

```bash
streamlit run app.py
```

Usually:

```text
http://localhost:8501
```

The app loads automatically; there is no separate "generate" step in the final version.

### Overview tab

Shows:

- current Lahore AQI
- current AQI category
- today's peak AQI
- next-72-hour peak and mean
- daily cards beginning with **Today**
- a chart combining today's recent Open-Meteo AQI with predicted remaining hours

### 72-hour forecast tab

Shows hourly predictions plus PM2.5, PM10, ozone, temperature, humidity and wind context.

### Six-month history tab

Loads the latest six months from the raw Hopsworks Feature Group and plots daily mean AQI.

### Model performance tab

Shows:

- Hopsworks model/version
- held-out RMSE
- held-out MAE
- held-out R²
- training metadata
- validation model comparison including XGBoost
- interactive Actual vs Predicted held-out test chart
- SHAP summary image when available

## 13. Optional FastAPI

Run:

```bash
uvicorn api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/forecast?hours=72
http://127.0.0.1:8000/docs
```

The `/forecast` response contains:

- current AQI
- daily summaries
- hourly 72-hour predictions
- model version/metrics
- six-month training-window metadata

## 14. Put the project on GitHub

Create an empty repository and run:

```bash
git init
git add .
git commit -m "Final Lahore AQI predictor"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

`.env` is already ignored. Confirm it is not staged before pushing.

## 15. Add GitHub repository secrets

GitHub repository → **Settings → Secrets and variables → Actions**.

Create:

```text
HOPSWORKS_API_KEY
HOPSWORKS_PROJECT
```

No Open-Meteo secret is needed.

## 16. Test GitHub Actions manually

The repository already contains:

```text
.github/workflows/feature_pipeline.yml
.github/workflows/training_pipeline.yml
```

Go to **Actions** and manually run each once.

The feature pipeline runs hourly and re-fetches a seven-day overlap. This makes upserts idempotent and repairs short outages.

The training pipeline runs daily at **06:35 Pakistan time** and always ignores data older than six months.

## 17. Deploy Streamlit

A simple option is Streamlit Community Cloud:

1. Push the repository to GitHub.
2. Create a Streamlit app from the repository.
3. Main file: `app.py`.
4. Add secrets in Streamlit's app settings:

```toml
HOPSWORKS_API_KEY = "your-key"
HOPSWORKS_PROJECT = "lahore_aqi_predictor"
```

5. Deploy.

The live app pulls current Open-Meteo data and downloads the best Hopsworks-registered model at runtime.

## 18. Final submission evidence to collect

Take screenshots of:

1. Hopsworks project home page
2. `lahore_aqi_raw_6m` Feature Group
3. `lahore_aqi_engineered_6m` Feature Group
4. `lahore_aqi_training_view_6m` Feature View
5. `lahore_aqi_forecaster_6m` Model Registry entry with metrics
6. successful hourly GitHub Action
7. successful daily training GitHub Action
8. Streamlit Overview tab with current-day prediction
9. Streamlit 72-hour forecast tab
10. Streamlit six-month history tab
11. Streamlit Model & SHAP tab
12. EDA images and `model_comparison.csv`

Do not put invented metric values in the report. Run the training pipeline first and copy the real RMSE/MAE/R² from `artifacts/metrics.json` or Hopsworks.

## 19. Normal day-to-day operation

After the initial backfill and first model training, you normally do nothing manually:

```text
Every hour:
Open-Meteo -> 7-day rolling recent refresh -> Hopsworks upsert

Every day:
Hopsworks Feature View -> latest 6 months only -> retrain -> Model Registry

Whenever dashboard opens:
Current Open-Meteo data + best registered model -> Today + next 72 hours
```
