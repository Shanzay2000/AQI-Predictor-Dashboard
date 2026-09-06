from __future__ import annotations

import calendar
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()


def subtract_calendar_months(day: date, months: int) -> date:
    """Subtract whole calendar months while keeping a valid day-of-month."""
    if months < 0:
        raise ValueError("months must be non-negative")
    year = day.year
    month = day.month - months
    while month <= 0:
        month += 12
        year -= 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last_day))


@dataclass(frozen=True)
class Settings:
    # Fixed project location.
    city_name: str = os.getenv("CITY_NAME", "Lahore")
    country_name: str = os.getenv("COUNTRY_NAME", "Pakistan")
    location_name: str = os.getenv("LOCATION_NAME", "Lahore, Pakistan")
    latitude: float = float(os.getenv("LATITUDE", "31.5204"))
    longitude: float = float(os.getenv("LONGITUDE", "74.3587"))
    local_timezone: str = os.getenv("LOCAL_TIMEZONE", "Asia/Karachi")

    # The model is intentionally limited to a rolling six-month training window.
    # Even if HISTORY_MONTHS is set higher, the project clamps it to six.
    history_months: int = max(1, min(int(os.getenv("HISTORY_MONTHS", "6")), 6))
    recent_overlap_days: int = int(os.getenv("RECENT_OVERLAP_DAYS", "7"))
    forecast_hours: int = int(os.getenv("FORECAST_HOURS", "72"))

    # Hopsworks objects. Version 2/name suffixes avoid clashes with the older draft.
    raw_feature_group_name: str = os.getenv(
        "RAW_FEATURE_GROUP_NAME", "lahore_aqi_raw_6m"
    )
    raw_feature_group_version: int = int(os.getenv("RAW_FEATURE_GROUP_VERSION", "1"))
    feature_group_name: str = os.getenv(
        "FEATURE_GROUP_NAME", "lahore_aqi_engineered_6m"
    )
    feature_group_version: int = int(os.getenv("FEATURE_GROUP_VERSION", "1"))
    feature_view_name: str = os.getenv(
        "FEATURE_VIEW_NAME", "lahore_aqi_training_view_6m"
    )
    feature_view_version: int = int(os.getenv("FEATURE_VIEW_VERSION", "1"))
    model_name: str = os.getenv("MODEL_NAME", "lahore_aqi_forecaster_6m")

    def local_today(self) -> date:
        return datetime.now(ZoneInfo(self.local_timezone)).date()

    def history_start_date(self, reference: date | None = None) -> date:
        reference = reference or self.local_today()
        return subtract_calendar_months(reference, self.history_months)

    def history_end_date(self, reference: date | None = None) -> date:
        return reference or self.local_today()

    def training_window_utc(self) -> tuple[datetime, datetime]:
        """Exact Lahore-local six-month window converted to UTC."""
        today = self.local_today()
        start = self.history_start_date(today)
        start_local = datetime(
            start.year, start.month, start.day, tzinfo=ZoneInfo(self.local_timezone)
        )
        return start_local.astimezone(timezone.utc), datetime.now(timezone.utc)


SETTINGS = Settings()
