from src.config import SETTINGS
from src.open_meteo import fetch_future_exogenous, fetch_recent_observations

print(f"Checking Open-Meteo for {SETTINGS.location_name}...")
recent = fetch_recent_observations(SETTINGS.latitude, SETTINGS.longitude, past_days=1)
future = fetch_future_exogenous(SETTINGS.latitude, SETTINGS.longitude, hours=6)
latest = recent.dropna(subset=["us_aqi"]).iloc[-1]
print("Recent rows:", len(recent))
print("Latest completed timestamp:", latest["timestamp"])
print("Latest US AQI:", latest["us_aqi"])
print("Future exogenous rows returned:", len(future))
print("Rolling training history is capped at", SETTINGS.history_months, "months")
print("Open-Meteo check PASSED")
