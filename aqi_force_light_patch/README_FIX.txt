READABILITY FIX ONLY

Replace in your current project:
  app.py
  .streamlit/config.toml

This does NOT change the trained model, prediction code, Hopsworks objects, SHAP logic, or .env.
It forces the app to a light background with dark text even when Windows/browser/Streamlit is in dark mode.

Restart Streamlit completely after replacing the files:
  streamlit run app.py

No backfill or retraining is required for this UI-only fix.
