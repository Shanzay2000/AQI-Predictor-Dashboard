from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .config import SETTINGS
from .features import build_feature_table, build_supervised
from .hopsworks_io import ensure_feature_view, insert_engineered, insert_raw
from .open_meteo import fetch_rolling_history


def main() -> None:
    start_date = SETTINGS.history_start_date()
    end_date = SETTINGS.history_end_date()

    print(
        f"Backfilling ONLY the latest {SETTINGS.history_months} months for "
        f"{SETTINGS.location_name}: {start_date} -> {end_date}"
    )

    # Request one extra UTC date so Lahore midnight (UTC+5) is fully covered, then
    # trim to the exact local six-month boundary below.
    fetch_start = start_date - timedelta(days=1)
    raw = fetch_rolling_history(
        SETTINGS.latitude,
        SETTINGS.longitude,
        fetch_start.isoformat(),
        end_date.isoformat(),
        recent_overlap_days=SETTINGS.recent_overlap_days,
    )
    if raw.empty:
        raise RuntimeError("Open-Meteo rolling backfill returned no rows")

    window_start, window_end = SETTINGS.training_window_utc()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw[(raw["timestamp"] >= window_start) & (raw["timestamp"] <= window_end)].reset_index(drop=True)

    # Raw group is useful for EDA and provenance.
    insert_raw(raw, SETTINGS.location_name)

    feature_table = build_feature_table(
        raw, SETTINGS.location_name, timezone=SETTINGS.local_timezone
    )
    supervised = build_supervised(feature_table)
    if supervised.empty:
        raise RuntimeError("Feature engineering produced no supervised rows")
    insert_engineered(supervised)
    ensure_feature_view()

    print(f"Raw rows written: {len(raw)}")
    print(f"Engineered training rows written: {len(supervised)}")
    print(
        "Hopsworks objects ready: "
        f"{SETTINGS.raw_feature_group_name}, {SETTINGS.feature_group_name}, "
        f"{SETTINGS.feature_view_name}"
    )


if __name__ == "__main__":
    main()
