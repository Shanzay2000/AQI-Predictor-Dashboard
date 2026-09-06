Replace these files in your existing project:
  app.py
  src/hopsworks_io.py

Do NOT replace .env.

Then run:
  python -m src.train
  streamlit run app.py

The app will use artifacts/model/model.pkl from the latest training run. Each training run compares all candidate models by validation RMSE and registers only the winner, so the displayed forecast uses the winner from the latest training run.
