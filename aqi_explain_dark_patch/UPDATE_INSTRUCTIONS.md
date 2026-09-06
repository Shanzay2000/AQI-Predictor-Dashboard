# Explainability + Dark Mode Patch (model architecture unchanged)

This patch keeps the existing single best-model / recursive 72-hour forecasting design unchanged.
It only adds/retains explainability support and fixes light/dark theme readability.

Replace:
- app.py
- src/train.py
- src/predict.py
- src/hopsworks_io.py
- .streamlit/config.toml

Do not replace `.env`.

Then run:

```cmd
python -m src.train
streamlit run app.py
```

You do not need to rerun `python -m src.backfill`.

Why retrain once: the patched training script saves `artifacts/shap_background.csv`, which is used for local SHAP explanations.

Forecast-model behavior is unchanged:
- Candidate algorithms are compared on validation RMSE.
- The lowest validation-RMSE model is selected.
- The same selected model is used recursively for the 72-hour forecast.
- Day 1/2/3 values are the +24h/+48h/+72h points from that recursive trajectory.

SHAP behavior:
- Global SHAP shows which features matter across the held-out test set.
- Local SHAP explains the model decision at the selected forecast step.
- For +24h/+48h/+72h, the explanation is for the final one-hour decision at that step. Because the forecast is recursive, lag/rolling features may already contain earlier model predictions. SHAP does not eliminate or attribute the entire accumulated recursive error path.
