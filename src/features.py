from __future__ import annotations

import math

import numpy as np
import pandas as pd

EXOGENOUS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

CURRENT_STATE_FEATURES = [
    "us_aqi",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_24",
    "aqi_change_1h",
    "aqi_change_3h",
    "aqi_rolling_mean_6h",
    "aqi_rolling_mean_24h",
]

NEXT_TIME_FEATURES = [
    "next_hour_sin",
    "next_hour_cos",
    "next_dow_sin",
    "next_dow_cos",
    "next_month_sin",
    "next_month_cos",
]

MODEL_FEATURES = CURRENT_STATE_FEATURES + [f"next_{c}" for c in EXOGENOUS] + NEXT_TIME_FEATURES
TARGET = "target_us_aqi_next_1h"


def add_time_features(df: pd.DataFrame, timezone: str = "Asia/Karachi") -> pd.DataFrame:
    out = df.copy()
    # Store event timestamps in UTC, but derive calendar features in Lahore local time.
    ts = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(timezone)
    out["hour"] = ts.dt.hour.astype("int16")
    out["day_of_week"] = ts.dt.dayofweek.astype("int16")
    out["month"] = ts.dt.month.astype("int16")
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)
    return out


def build_feature_table(
    raw_df: pd.DataFrame, city: str, timezone: str = "Asia/Karachi"
) -> pd.DataFrame:
    df = raw_df.copy().sort_values("timestamp").drop_duplicates("timestamp")
    df = add_time_features(df, timezone=timezone)
    df["city"] = city

    for lag in (1, 3, 6, 24):
        df[f"aqi_lag_{lag}"] = df["us_aqi"].shift(lag)
    df["aqi_change_1h"] = df["us_aqi"] - df["aqi_lag_1"]
    df["aqi_change_3h"] = df["us_aqi"] - df["aqi_lag_3"]
    df["aqi_rolling_mean_6h"] = df["us_aqi"].rolling(6, min_periods=1).mean()
    df["aqi_rolling_mean_24h"] = df["us_aqi"].rolling(24, min_periods=1).mean()

    # Hopsworks handles Pandas datetimes reliably when timezone information is removed.
    # All timestamps are still UTC by convention.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df.reset_index(drop=True)


def build_supervised(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Create a one-step-ahead forecasting dataset.

    At time t, the model receives the current AQI state + weather/pollutant values for
    t+1. In production those t+1 exogenous values come from Open-Meteo forecasts.
    This avoids leaking Open-Meteo's future US AQI while still allowing a 72-hour
    recursive forecast.
    """
    df = feature_df.copy().sort_values("timestamp").reset_index(drop=True)
    for col in EXOGENOUS:
        df[f"next_{col}"] = df[col].shift(-1)
    for src, dst in (
        ("hour_sin", "next_hour_sin"),
        ("hour_cos", "next_hour_cos"),
        ("dow_sin", "next_dow_sin"),
        ("dow_cos", "next_dow_cos"),
        ("month_sin", "next_month_sin"),
        ("month_cos", "next_month_cos"),
    ):
        df[dst] = df[src].shift(-1)
    df[TARGET] = df["us_aqi"].shift(-1)
    keep = ["timestamp", "city"] + MODEL_FEATURES + [TARGET]
    return df[keep].dropna().reset_index(drop=True)


def time_features_for_timestamp(timestamp: pd.Timestamp, timezone: str = "Asia/Karachi") -> dict[str, float]:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    ts = ts.tz_convert(timezone)
    hour = ts.hour
    dow = ts.dayofweek
    month = ts.month
    return {
        "next_hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "next_hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "next_dow_sin": math.sin(2 * math.pi * dow / 7.0),
        "next_dow_cos": math.cos(2 * math.pi * dow / 7.0),
        "next_month_sin": math.sin(2 * math.pi * (month - 1) / 12.0),
        "next_month_cos": math.cos(2 * math.pi * (month - 1) / 12.0),
    }


def state_features_from_history(history: list[float]) -> dict[str, float]:
    if not history:
        raise ValueError("AQI history is empty")

    def lag(hours: int) -> float:
        idx = len(history) - 1 - hours
        return float(history[idx]) if idx >= 0 else float(history[0])

    current = float(history[-1])
    last6 = history[-6:] if len(history) >= 6 else history
    last24 = history[-24:] if len(history) >= 24 else history
    return {
        "us_aqi": current,
        "aqi_lag_1": lag(1),
        "aqi_lag_3": lag(3),
        "aqi_lag_6": lag(6),
        "aqi_lag_24": lag(24),
        "aqi_change_1h": current - lag(1),
        "aqi_change_3h": current - lag(3),
        "aqi_rolling_mean_6h": float(np.mean(last6)),
        "aqi_rolling_mean_24h": float(np.mean(last24)),
    }
