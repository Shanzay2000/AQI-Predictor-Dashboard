# Lahore AQI Predictor — UI + XGBoost Update

This update keeps the existing Hopsworks Feature Groups, Feature View, API credentials, and six-month data. **Do not delete `.env` and do not backfill again.**

## What changed

- User-facing text now says **AQI**, not "US AQI". The internal Open-Meteo column remains `us_aqi` because that is the source API field name.
- Streamlit UI redesigned with pastel cards and cleaner charts.
- Separate **Day 1 (+24h), Day 2 (+48h), Day 3 (+72h)** forecast cards. Each also shows the mean and peak for its 24-hour block.
- Model tab now shows an interactive **Actual vs Predicted** held-out test chart.
- **XGBoost** added to the model comparison.
- Correlation matrix now prints the numeric correlation coefficient in every cell.
- `test_predictions.csv` is saved and copied into the Hopsworks model artifact.
- Hopsworks 5.0/Windows fixes from setup are retained (certificate folder support, HUDI feature groups, Windows `C:\tmp`, labelled Feature View training read).
- Open-Meteo transient 429/5xx retry logic retained.

## Apply the update to an existing working project

1. Copy/overwrite the files from the patch ZIP into your existing project folder. The patch does **not** include `.env`.
2. Install XGBoost in the existing `lahore-aqi` environment:

```cmd
python -m pip install "xgboost>=2.1,<4"
```

3. Re-run EDA to regenerate the annotated correlation matrix:

```cmd
python -m src.eda
```

4. Re-run training so XGBoost is compared and the new Actual-vs-Predicted data is registered with the model:

```cmd
python -m src.train
```

5. Start the redesigned app:

```cmd
streamlit run app.py
```

No six-month backfill is required because the feature data already exists in Hopsworks.

## Hourly updating

`.github/workflows/feature_pipeline.yml` is scheduled every hour. Once the project is pushed to GitHub and the Hopsworks secrets are configured, it updates recent Feature Store rows hourly. The model training workflow runs daily. The Streamlit page fetches current Open-Meteo data and a fresh 72-hour forecast whenever the page is loaded or the **Refresh now** button is pressed.
