from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .config import SETTINGS
from .features import (
    EXOGENOUS,
    MODEL_FEATURES,
    state_features_from_history,
    time_features_for_timestamp,
)
from .hopsworks_io import load_best_model
from .open_meteo import fetch_future_exogenous, fetch_recent_observations


def aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def forecast_72h(latitude: float, longitude: float, hours: int = 72):
    model, metadata, model_meta, model_dir = load_best_model()
    feature_names = metadata.get("feature_names", MODEL_FEATURES)

    # Three days of actual/recent AQI is more than enough to construct the 24h lag state.
    recent = fetch_recent_observations(latitude, longitude, past_days=3)
    recent = recent.dropna(subset=["us_aqi"]).sort_values("timestamp").reset_index(drop=True)
    if recent.empty:
        raise RuntimeError("No recent AQI observations are available from Open-Meteo")

    latest_timestamp = pd.to_datetime(recent.iloc[-1]["timestamp"], utc=True)
    history = recent["us_aqi"].astype(float).tolist()

    future = fetch_future_exogenous(latitude, longitude, hours=hours)
    future["timestamp"] = pd.to_datetime(future["timestamp"], utc=True)
    future = future[future["timestamp"] > latest_timestamp].head(hours).reset_index(drop=True)
    if future.empty:
        raise RuntimeError("Open-Meteo returned no future exogenous forecast rows")

    predictions: list[float] = []
    feature_rows: list[dict[str, float]] = []
    for _, row in future.iterrows():
        features = state_features_from_history(history)
        for col in EXOGENOUS:
            features[f"next_{col}"] = float(row[col]) if pd.notna(row[col]) else 0.0
        features.update(
            time_features_for_timestamp(row["timestamp"], SETTINGS.local_timezone)
        )

        X = pd.DataFrame(
            [[features[name] for name in feature_names]],
            columns=feature_names,
        )
        pred = float(model.predict(X)[0])
        pred = max(0.0, min(500.0, pred))
        predictions.append(pred)
        feature_rows.append({name: float(features[name]) for name in feature_names})
        history.append(pred)

    out = future[["timestamp"] + EXOGENOUS].copy()
    out["predicted_us_aqi"] = predictions
    out["category"] = out["predicted_us_aqi"].map(aqi_category)

    # Keep the exact model input used for each forecast step so the dashboard can
    # explain individual predictions with local SHAP values. Prefix these columns
    # so they never collide with display/output fields.
    feature_df = pd.DataFrame(feature_rows)
    for name in feature_names:
        out[f"_feature__{name}"] = feature_df[name].to_numpy()

    return out, recent, model_meta, metadata, model_dir


def build_daily_summary(
    recent: pd.DataFrame,
    forecast: pd.DataFrame,
    timezone: str = "Asia/Karachi",
) -> pd.DataFrame:
    """Build calendar-day summaries beginning with the current Lahore day.

    Today's row combines recent Open-Meteo AQI up to the current hour with model
    predictions for the remaining hours. Future dates contain forecast values only.
    """
    recent = recent.copy()
    forecast = forecast.copy()
    recent["local_time"] = pd.to_datetime(recent["timestamp"], utc=True).dt.tz_convert(timezone)
    forecast["local_time"] = pd.to_datetime(forecast["timestamp"], utc=True).dt.tz_convert(timezone)
    recent["local_date"] = recent["local_time"].dt.date
    forecast["local_date"] = forecast["local_time"].dt.date

    today = pd.Timestamp.now(tz=timezone).date()
    tomorrow = today + timedelta(days=1)
    all_dates = sorted(set([today]) | set(forecast["local_date"].tolist()))

    rows: list[dict] = []
    for day in all_dates:
        observed = recent.loc[recent["local_date"] == day, "us_aqi"].dropna().astype(float)
        predicted = forecast.loc[
            forecast["local_date"] == day, "predicted_us_aqi"
        ].dropna().astype(float)
        values = pd.concat([observed, predicted], ignore_index=True)
        if values.empty:
            continue

        if day == today:
            label = "Today"
        elif day == tomorrow:
            label = "Tomorrow"
        else:
            label = pd.Timestamp(day).strftime("%a %d %b")

        if len(observed) and len(predicted):
            source = "recent + forecast"
        elif len(predicted):
            source = "forecast"
        else:
            source = "recent"

        rows.append(
            {
                "date": day,
                "label": label,
                "mean_aqi": float(values.mean()),
                "peak_aqi": float(values.max()),
                "min_aqi": float(values.min()),
                "category": aqi_category(float(values.max())),
                "recent_hours": int(len(observed)),
                "forecast_hours": int(len(predicted)),
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def build_today_timeline(
    recent: pd.DataFrame,
    forecast: pd.DataFrame,
    timezone: str = "Asia/Karachi",
) -> pd.DataFrame:
    """Recent Open-Meteo hours today + predicted remaining hours for the current-day chart."""
    today = pd.Timestamp.now(tz=timezone).date()

    observed = recent[["timestamp", "us_aqi"]].copy()
    observed["local_time"] = pd.to_datetime(observed["timestamp"], utc=True).dt.tz_convert(timezone)
    observed = observed[observed["local_time"].dt.date == today]
    observed = observed.rename(columns={"us_aqi": "aqi"})
    observed["series"] = "Recent Open-Meteo"

    predicted = forecast[["timestamp", "predicted_us_aqi"]].copy()
    predicted["local_time"] = pd.to_datetime(predicted["timestamp"], utc=True).dt.tz_convert(timezone)
    predicted = predicted[predicted["local_time"].dt.date == today]
    predicted = predicted.rename(columns={"predicted_us_aqi": "aqi"})
    predicted["series"] = "Model forecast"

    return pd.concat(
        [
            observed[["local_time", "aqi", "series"]],
            predicted[["local_time", "aqi", "series"]],
        ],
        ignore_index=True,
    ).sort_values("local_time")
