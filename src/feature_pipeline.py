from __future__ import annotations

from .config import SETTINGS
from .features import build_feature_table, build_supervised
from .hopsworks_io import ensure_feature_view, insert_engineered, insert_raw
from .open_meteo import fetch_recent_observations


def main() -> None:
    # Re-fetch a seven-day rolling overlap on every scheduled run. Upserts make this
    # idempotent and automatically repair short GitHub Actions/Hopsworks outages.
    raw = fetch_recent_observations(
        SETTINGS.latitude,
        SETTINGS.longitude,
        past_days=SETTINGS.recent_overlap_days,
    )
    if raw.empty:
        raise RuntimeError("Open-Meteo recent feed returned no rows")

    insert_raw(raw, SETTINGS.location_name)

    feature_table = build_feature_table(
        raw, SETTINGS.location_name, timezone=SETTINGS.local_timezone
    )
    supervised = build_supervised(feature_table)
    insert_engineered(supervised)
    ensure_feature_view()

    print(
        f"Upserted {len(raw)} recent raw rows and {len(supervised)} engineered rows "
        f"for {SETTINGS.location_name}. Training will still use only the latest "
        f"{SETTINGS.history_months} months."
    )


if __name__ == "__main__":
    main()
