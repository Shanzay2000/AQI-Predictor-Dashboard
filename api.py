from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.config import SETTINGS
from src.predict import build_daily_summary, forecast_72h

app = FastAPI(
    title="Lahore AQI Predictor API",
    version="2.0.0",
    description=(
        "Current-day and 72-hour Lahore AQI forecasts. The model is trained on a rolling "
        "maximum six-month history from Open-Meteo and registered in Hopsworks."
    ),
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "location": SETTINGS.location_name,
        "latitude": SETTINGS.latitude,
        "longitude": SETTINGS.longitude,
        "history_months": SETTINGS.history_months,
        "forecast_hours": SETTINGS.forecast_hours,
    }


@app.get("/forecast")
def forecast(hours: int = Query(72, ge=1, le=72)):
    try:
        df, recent, model_meta, metadata, _ = forecast_72h(
            SETTINGS.latitude,
            SETTINGS.longitude,
            hours=hours,
        )
        latest = recent.iloc[-1]
        daily = build_daily_summary(recent, df, SETTINGS.local_timezone)
        return {
            "location": {
                "name": SETTINGS.location_name,
                "latitude": SETTINGS.latitude,
                "longitude": SETTINGS.longitude,
                "timezone": SETTINGS.local_timezone,
            },
            "training_window_months": SETTINGS.history_months,
            "current": {
                "timestamp_utc": latest["timestamp"].isoformat(),
                "aqi": round(float(latest["us_aqi"]), 2),
            },
            "model": {
                "name": model_meta.name,
                "version": model_meta.version,
                "metrics": metadata.get("metrics", {}),
            },
            "daily_summary": daily.to_dict(orient="records"),
            "hourly_forecast": [
                {
                    "timestamp_utc": row.timestamp.isoformat(),
                    "predicted_aqi": round(float(row.predicted_us_aqi), 2),
                    "category": row.category,
                }
                for row in df[["timestamp", "predicted_us_aqi", "category"]].itertuples(
                    index=False
                )
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
