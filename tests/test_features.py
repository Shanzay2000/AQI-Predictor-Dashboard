from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.config import subtract_calendar_months
from src.features import (
    MODEL_FEATURES,
    TARGET,
    add_time_features,
    build_feature_table,
    build_supervised,
    state_features_from_history,
)
from src.predict import aqi_category, build_daily_summary


def synthetic_raw(n=100):
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    x = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": 20 + np.sin(x / 8),
            "relative_humidity_2m": 60 + np.cos(x / 10),
            "precipitation": np.zeros(n),
            "surface_pressure": np.full(n, 1013.0),
            "wind_speed_10m": 8 + np.sin(x / 4),
            "pm2_5": 30 + 0.2 * x,
            "pm10": 50 + 0.3 * x,
            "carbon_monoxide": 200 + x,
            "nitrogen_dioxide": 20 + 0.1 * x,
            "sulphur_dioxide": 5 + 0.02 * x,
            "ozone": 70 + np.sin(x / 3),
            "us_aqi": 80 + 0.5 * x,
        }
    )


def test_feature_and_supervised_shapes():
    feature_df = build_feature_table(
        synthetic_raw(), "Lahore, Pakistan", timezone="Asia/Karachi"
    )
    supervised = build_supervised(feature_df)
    assert set(MODEL_FEATURES).issubset(supervised.columns)
    assert TARGET in supervised.columns
    assert len(supervised) > 40
    assert supervised[MODEL_FEATURES + [TARGET]].notna().all().all()


def test_lahore_local_hour_features():
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")]})
    out = add_time_features(df, timezone="Asia/Karachi")
    assert int(out.loc[0, "hour"]) == 5


def test_state_features_match_training_rolling_definition():
    raw = synthetic_raw(40)
    feature_df = build_feature_table(raw, "Lahore, Pakistan")
    history = raw["us_aqi"].astype(float).tolist()
    state = state_features_from_history(history)
    row = feature_df.iloc[-1]
    assert state["us_aqi"] == row["us_aqi"]
    assert state["aqi_lag_1"] == row["aqi_lag_1"]
    assert np.isclose(state["aqi_rolling_mean_6h"], row["aqi_rolling_mean_6h"])
    assert np.isclose(state["aqi_rolling_mean_24h"], row["aqi_rolling_mean_24h"])


def test_six_calendar_months_ago():
    assert subtract_calendar_months(date(2026, 9, 6), 6) == date(2026, 3, 6)
    assert subtract_calendar_months(date(2026, 8, 31), 6) == date(2026, 2, 28)


def test_aqi_categories():
    assert aqi_category(50) == "Good"
    assert aqi_category(151) == "Unhealthy"
    assert aqi_category(301) == "Hazardous"


def test_daily_summary_combines_observed_and_forecast(monkeypatch):
    # Make the function's current date deterministic by choosing timestamps around a
    # date and only checking aggregation/source logic, not the label text.
    now = pd.Timestamp.now(tz="Asia/Karachi")
    today = now.normalize()
    observed_times = pd.date_range(today, periods=3, freq="h").tz_convert("UTC")
    future_times = pd.date_range(now.ceil("h"), periods=4, freq="h").tz_convert("UTC")
    recent = pd.DataFrame({"timestamp": observed_times, "us_aqi": [90, 100, 110]})
    forecast = pd.DataFrame(
        {"timestamp": future_times, "predicted_us_aqi": [120, 130, 140, 150]}
    )
    summary = build_daily_summary(recent, forecast, "Asia/Karachi")
    assert not summary.empty
    assert summary.iloc[0]["date"] == now.date()
    assert summary.iloc[0]["recent_hours"] == 3
