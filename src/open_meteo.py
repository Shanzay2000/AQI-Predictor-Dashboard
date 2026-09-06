from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable
import time

import pandas as pd
import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
]

POLLUTION_VARS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

AQ_VARS_WITH_TARGET = POLLUTION_VARS + ["us_aqi"]


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    country: str
    timezone: str

    @property
    def canonical_name(self) -> str:
        return f"{self.name}, {self.country}" if self.country else self.name


def _get_json(
    url: str,
    params: dict,
    timeout: int = 60,
    max_retries: int = 6,
) -> dict:
    """GET JSON with exponential-backoff retries for transient Open-Meteo failures."""
    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in retryable_statuses and attempt < max_retries - 1:
                wait_seconds = min(2 ** attempt, 30)
                print(
                    f"Open-Meteo returned HTTP {response.status_code}. "
                    f"Retrying in {wait_seconds}s ({attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(payload.get("reason", "Open-Meteo request failed"))
            return payload

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt == max_retries - 1:
                raise
            wait_seconds = min(2 ** attempt, 30)
            print(
                f"Open-Meteo connection error: {exc}. "
                f"Retrying in {wait_seconds}s ({attempt + 1}/{max_retries})..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Open-Meteo request failed after all retries")


def geocode_city(city: str) -> Location:
    payload = _get_json(
        GEOCODING_URL,
        {"name": city, "count": 1, "language": "en", "format": "json"},
    )
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"No Open-Meteo geocoding result for: {city}")
    r = results[0]
    return Location(
        name=r["name"],
        latitude=float(r["latitude"]),
        longitude=float(r["longitude"]),
        country=r.get("country", ""),
        timezone=r.get("timezone", "UTC"),
    )


def _hourly_to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("Open-Meteo response did not contain hourly data")
    df = pd.DataFrame(hourly)
    # Requests are made in UTC, so interpret the returned timestamps as UTC.
    df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True)
    return df


def _merge_weather_air(weather: pd.DataFrame, air: pd.DataFrame) -> pd.DataFrame:
    df = weather.merge(air, on="timestamp", how="inner", validate="one_to_one")
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_historical_range(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    weather = _hourly_to_frame(
        _get_json(
            WEATHER_ARCHIVE_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(WEATHER_VARS),
                "timezone": "UTC",
            },
        )
    )
    air = _hourly_to_frame(
        _get_json(
            AIR_QUALITY_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(AQ_VARS_WITH_TARGET),
                "timezone": "UTC",
                "domains": "auto",
            },
        )
    )
    return _merge_weather_air(weather, air)


def _date_chunks(
    start_date: str, end_date: str, chunk_days: int = 31
) -> Iterable[tuple[str, str]]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        yield cursor.isoformat(), chunk_end.isoformat()
        cursor = chunk_end + timedelta(days=1)


def fetch_historical_chunked(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    chunk_days: int = 31,
) -> pd.DataFrame:
    if date.fromisoformat(start_date) > date.fromisoformat(end_date):
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(start_date, end_date, chunk_days):
        print(f"Fetching Open-Meteo archive: {chunk_start} -> {chunk_end}")
        frames.append(
            fetch_historical_range(latitude, longitude, chunk_start, chunk_end)
        )
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_recent_observations(
    latitude: float,
    longitude: float,
    past_days: int = 7,
) -> pd.DataFrame:
    """Fetch recent completed hours, including the current calendar day.

    The forecast endpoints expose a rolling set of past hours. We use this for the
    newest days because the archive endpoint can lag behind the live feed.
    """
    weather = _hourly_to_frame(
        _get_json(
            WEATHER_FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ",".join(WEATHER_VARS),
                "past_days": past_days,
                "forecast_days": 1,
                "timezone": "UTC",
            },
        )
    )
    air = _hourly_to_frame(
        _get_json(
            AIR_QUALITY_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ",".join(AQ_VARS_WITH_TARGET),
                "past_days": past_days,
                "forecast_days": 1,
                "timezone": "UTC",
                "domains": "auto",
            },
        )
    )
    df = _merge_weather_air(weather, air)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    return df[df["timestamp"] <= now].reset_index(drop=True)


def fetch_rolling_history(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    recent_overlap_days: int = 7,
) -> pd.DataFrame:
    """Fetch one continuous rolling history up through the current day.

    Older rows come from the archive API; the newest overlap comes from the live
    forecast APIs. Duplicate timestamps are resolved in favor of the live/recent feed.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    archive_end = end - timedelta(days=recent_overlap_days)

    frames: list[pd.DataFrame] = []
    if start <= archive_end:
        frames.append(
            fetch_historical_chunked(
                latitude,
                longitude,
                start.isoformat(),
                archive_end.isoformat(),
                chunk_days=31,
            )
        )

    recent = fetch_recent_observations(
        latitude, longitude, past_days=max(1, recent_overlap_days)
    )
    frames.append(recent)

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    start_ts = pd.Timestamp(start_date, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    return combined[
        (combined["timestamp"] >= start_ts) & (combined["timestamp"] <= now)
    ].reset_index(drop=True)


def fetch_future_exogenous(
    latitude: float,
    longitude: float,
    hours: int = 72,
) -> pd.DataFrame:
    """Future weather + pollutant forecasts, excluding future AQI.

    The trained model predicts AQI. Feeding Open-Meteo's future `us_aqi` into it would
    leak the target, so only weather and pollutant forecasts are used as exogenous
    predictors.
    """
    # Ask for a small buffer because Open-Meteo can include the current hour.
    request_hours = hours + 6
    weather = _hourly_to_frame(
        _get_json(
            WEATHER_FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ",".join(WEATHER_VARS),
                "forecast_hours": request_hours,
                "timezone": "UTC",
            },
        )
    )
    air = _hourly_to_frame(
        _get_json(
            AIR_QUALITY_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ",".join(POLLUTION_VARS),
                "forecast_hours": request_hours,
                "timezone": "UTC",
                "domains": "auto",
            },
        )
    )
    return _merge_weather_air(weather, air)
