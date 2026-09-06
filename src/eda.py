from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import SETTINGS
from .hopsworks_io import read_raw_features


def main() -> None:
    start_time, end_time = SETTINGS.training_window_utc()
    df = read_raw_features(
        SETTINGS.location_name,
        start_time=start_time,
        end_time=end_time,
    )
    if df.empty:
        raise RuntimeError("Raw Feature Group is empty. Run backfill.py first.")

    out = Path("artifacts/eda")
    out.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("timestamp")

    plt.figure(figsize=(12, 4.5))
    plt.plot(df["timestamp"], df["us_aqi"])
    plt.title(f"Lahore hourly AQI — latest {SETTINGS.history_months} months")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(out / "aqi_timeseries.png", dpi=160)
    plt.close()

    daily = df.set_index("timestamp")["us_aqi"].resample("D").mean()
    plt.figure(figsize=(12, 4.5))
    plt.plot(daily.index, daily.values)
    plt.title(f"Lahore daily mean AQI — latest {SETTINGS.history_months} months")
    plt.xlabel("Date")
    plt.ylabel("Mean AQI")
    plt.tight_layout()
    plt.savefig(out / "aqi_daily_mean.png", dpi=160)
    plt.close()

    cols = [
        "us_aqi",
        "pm2_5",
        "pm10",
        "nitrogen_dioxide",
        "ozone",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
    ]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr(numeric_only=True)
    display_names = {
        "us_aqi": "AQI",
        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "nitrogen_dioxide": "NO₂",
        "ozone": "Ozone",
        "temperature_2m": "Temperature",
        "relative_humidity_2m": "Humidity",
        "wind_speed_10m": "Wind speed",
        "precipitation": "Precipitation",
    }
    labels = [display_names.get(c, c) for c in available]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(available)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(available)), labels)

    # Print each correlation coefficient directly on the heatmap.
    for i in range(len(available)):
        for j in range(len(available)):
            value = float(corr.iloc[i, j])
            text_color = "white" if abs(value) >= 0.55 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(out / "correlation_matrix.png", dpi=160)
    plt.close(fig)

    print(f"EDA charts written to {out} using only the latest {SETTINGS.history_months} months")


if __name__ == "__main__":
    main()
