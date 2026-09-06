from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import SETTINGS
from src.features import MODEL_FEATURES
from src.hopsworks_io import load_best_model, read_raw_features
from src.predict import (
    aqi_category,
    build_daily_summary,
    build_today_timeline,
    forecast_72h,
)

st.set_page_config(page_title="Lahore AQI Predictor", page_icon="🌿", layout="wide")

# Theme-safe UI: standard Streamlit text inherits the active light/dark theme.
# Only pastel cards use explicit dark text because their backgrounds remain light.
st.markdown(
    """
    <style>
      .block-container { max-width: 1320px; padding-top: 1.4rem; padding-bottom: 3rem; }
      [data-testid="stHeader"] { background: transparent; }

      .hero {
        background: linear-gradient(120deg, #dff5ec 0%, #e6efff 52%, #f5e8ff 100%);
        border: 1px solid rgba(80, 95, 120, 0.12);
        border-radius: 24px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 12px 30px rgba(82, 95, 120, 0.08);
        margin-bottom: 1rem;
      }
      .hero h1 { margin: 0; color: #24313d !important; font-size: 2.25rem; }
      .hero p { margin: .5rem 0 0 0; color: #536372 !important; font-size: 1.02rem; }

      .model-banner {
        background: #eef8f2;
        border: 1px solid #cfe8da;
        border-radius: 14px;
        padding: .8rem 1rem;
        margin: .55rem 0 1.2rem 0;
        color: #294a3b !important;
      }
      .model-banner * { color: #294a3b !important; }

      .method-note {
        border: 1px solid rgba(120, 130, 145, .25);
        border-radius: 14px;
        padding: .85rem 1rem;
        margin: .35rem 0 1rem 0;
      }

      .aqi-card {
        border-radius: 20px;
        padding: 1.05rem 1.15rem;
        min-height: 145px;
        border: 1px solid rgba(65, 75, 90, 0.09);
        box-shadow: 0 8px 24px rgba(80, 90, 110, 0.07);
      }
      .mint { background: #e3f6ee; }
      .blue { background: #e7f0ff; }
      .peach { background: #fff0e5; }
      .lavender { background: #f1e9ff; }
      .rose { background: #ffe9ef; }
      .card-label { color: #66717d !important; font-size: .83rem; font-weight: 700; letter-spacing: .02em; }
      .card-value { color: #24313d !important; font-size: 2rem; font-weight: 780; margin-top: .18rem; }
      .card-sub { color: #5f6c78 !important; font-size: .88rem; margin-top: .3rem; }

      .shap-note {
        background: #fff7e9;
        border: 1px solid #f1dfbd;
        border-radius: 14px;
        padding: .8rem 1rem;
        color: #594a32 !important;
      }
      .shap-note * { color: #594a32 !important; }

      div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 130, 145, .20);
        border-radius: 18px;
        padding: .8rem 1rem;
      }
      div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
      .stButton button { border-radius: 12px !important; font-weight: 650 !important; }

      @media (prefers-color-scheme: dark) {
        .method-note { background: rgba(255,255,255,.035); border-color: rgba(255,255,255,.14); }
        div[data-testid="stMetric"] { background: rgba(255,255,255,.025); border-color: rgba(255,255,255,.12); }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🌿 Lahore AQI Predictor</h1>
      <p>Current conditions, today’s outlook, a 72-hour forecast, and SHAP explanations using Open-Meteo + Hopsworks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

local_now = pd.Timestamp.now(tz=SETTINGS.local_timezone)
info_left, info_right = st.columns([5, 1])
with info_left:
    st.markdown(
        f"**{SETTINGS.location_name}** · {SETTINGS.latitude:.4f}, {SETTINGS.longitude:.4f} · "
        f"Lahore time: **{local_now.strftime('%a %d %b %Y, %I:%M %p')}**"
    )
    st.caption(
        "Training uses only the latest six months. The Feature Store ingestion workflow runs hourly; "
        "the selected model retrains daily. The live forecast refreshes whenever this page is loaded or refreshed."
    )
with info_right:
    if st.button("↻ Refresh now", use_container_width=True):
        st.rerun()

try:
    with st.spinner("Loading current Open-Meteo data and the selected best model..."):
        forecast, recent, model_meta, metadata, model_dir = forecast_72h(
            SETTINGS.latitude,
            SETTINGS.longitude,
            hours=SETTINGS.forecast_hours,
        )
except Exception as exc:
    st.error(str(exc))
    st.info(
        "First-time setup: configure HOPSWORKS_API_KEY/HOPSWORKS_PROJECT, run "
        "`python -m src.backfill`, then run `python -m src.train`."
    )
    st.stop()

latest = recent.iloc[-1]
current_aqi = float(latest["us_aqi"])
daily = build_daily_summary(recent, forecast, SETTINGS.local_timezone)
next72_peak = float(forecast["predicted_us_aqi"].max())
next72_mean = float(forecast["predicted_us_aqi"].mean())
metrics = metadata.get("metrics", {})
best_model_name = str(metrics.get("best_model", "unknown"))

st.markdown(
    f"""
    <div class="model-banner">
      <strong>Forecast model:</strong> {best_model_name.replace('_', ' ').title()} &nbsp;·&nbsp;
      <strong>Registry:</strong> {model_meta.name} v{model_meta.version} &nbsp;·&nbsp;
      <strong>Selection:</strong> lowest validation RMSE in the latest training run<br>
      <strong>Forecast method:</strong> the same selected model is applied recursively hour-by-hour for the 72-hour trajectory.
    </div>
    """,
    unsafe_allow_html=True,
)


def pastel_card(label: str, value: str, sub: str, css_class: str) -> None:
    st.markdown(
        f"""
        <div class="aqi-card {css_class}">
          <div class="card-label">{label}</div>
          <div class="card-value">{value}</div>
          <div class="card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def horizon_stats(start_idx: int, end_idx: int, point_idx: int) -> dict[str, float | str]:
    window = forecast.iloc[start_idx:end_idx]
    point = forecast.iloc[min(point_idx, len(forecast) - 1)]
    return {
        "point": float(point["predicted_us_aqi"]),
        "mean": float(window["predicted_us_aqi"].mean()),
        "peak": float(window["predicted_us_aqi"].max()),
        "category": aqi_category(float(point["predicted_us_aqi"])),
    }


def feature_label(name: str) -> str:
    direct = {
        "us_aqi": "Current AQI state",
        "aqi_lag_1": "AQI 1 hour ago",
        "aqi_lag_3": "AQI 3 hours ago",
        "aqi_lag_6": "AQI 6 hours ago",
        "aqi_lag_24": "AQI 24 hours ago",
        "aqi_change_1h": "AQI change over 1 hour",
        "aqi_change_3h": "AQI change over 3 hours",
        "aqi_rolling_mean_6h": "6-hour mean AQI",
        "aqi_rolling_mean_24h": "24-hour mean AQI",
        "next_temperature_2m": "Forecast temperature",
        "next_relative_humidity_2m": "Forecast humidity",
        "next_precipitation": "Forecast precipitation",
        "next_surface_pressure": "Forecast surface pressure",
        "next_wind_speed_10m": "Forecast wind speed",
        "next_pm2_5": "Forecast PM2.5",
        "next_pm10": "Forecast PM10",
        "next_carbon_monoxide": "Forecast carbon monoxide",
        "next_nitrogen_dioxide": "Forecast nitrogen dioxide",
        "next_sulphur_dioxide": "Forecast sulphur dioxide",
        "next_ozone": "Forecast ozone",
        "next_hour_sin": "Hour-of-day pattern (sin)",
        "next_hour_cos": "Hour-of-day pattern (cos)",
        "next_dow_sin": "Day-of-week pattern (sin)",
        "next_dow_cos": "Day-of-week pattern (cos)",
        "next_month_sin": "Month pattern (sin)",
        "next_month_cos": "Month pattern (cos)",
    }
    return direct.get(name, name.replace("_", " ").title())


def forecast_feature_row(row_idx: int, feature_names: list[str]) -> pd.DataFrame:
    row = forecast.iloc[min(row_idx, len(forecast) - 1)]
    values = {}
    for name in feature_names:
        col = f"_feature__{name}"
        if col not in forecast.columns:
            raise RuntimeError(
                "This forecast was created with an older predict.py. Replace src/predict.py "
                "with the SHAP patch and restart Streamlit."
            )
        values[name] = float(row[col])
    return pd.DataFrame([values], columns=feature_names)


def load_shap_background(feature_names: list[str]) -> pd.DataFrame:
    candidates = [
        Path(model_dir) / "shap_background.csv",
        Path("artifacts/shap_background.csv"),
        Path("artifacts/model/shap_background.csv"),
    ]
    for path in candidates:
        if path.exists():
            bg = pd.read_csv(path)
            if all(name in bg.columns for name in feature_names):
                return bg[feature_names].dropna().head(80)

    # Fallback: use the current 72-hour feature states as the reference distribution.
    cols = [f"_feature__{name}" for name in feature_names]
    if all(col in forecast.columns for col in cols):
        bg = forecast[cols].copy()
        bg.columns = feature_names
        return bg.dropna().head(72)
    raise RuntimeError("No SHAP background data is available. Retrain once with the updated train.py.")


def local_shap_explanation(row_idx: int) -> tuple[pd.DataFrame, float, float]:
    import shap

    model, model_metadata, _, _ = load_best_model()
    feature_names = list(model_metadata.get("feature_names", MODEL_FEATURES))
    X_one = forecast_feature_row(row_idx, feature_names)
    background = load_shap_background(feature_names)

    explainer = shap.Explainer(model.predict, background)
    explanation = explainer(
        X_one,
        max_evals=max(2 * len(feature_names) + 1, 51),
    )
    shap_values = np.asarray(explanation.values)[0].astype(float)
    base_values = np.asarray(explanation.base_values).reshape(-1)
    base_value = float(base_values[0]) if len(base_values) else float("nan")
    model_output = float(model.predict(X_one)[0])

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "Feature": [feature_label(name) for name in feature_names],
            "Input value": [float(X_one.iloc[0][name]) for name in feature_names],
            "SHAP effect": shap_values,
        }
    )
    result["Absolute effect"] = result["SHAP effect"].abs()
    return result.sort_values("Absolute effect", ascending=False), base_value, model_output


overview_tab, details_tab, explain_tab, history_tab, model_tab = st.tabs(
    [
        "Overview",
        "72-hour forecast",
        "Why this prediction (SHAP)",
        "Six-month history",
        "Model & evaluation",
    ]
)

with overview_tab:
    st.subheader("Current conditions")
    today_row = daily.iloc[0] if not daily.empty else None
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pastel_card("CURRENT AQI", f"{current_aqi:.0f}", aqi_category(current_aqi), "mint")
    with c2:
        if today_row is not None:
            pastel_card(
                "TODAY'S PEAK",
                f"{float(today_row['peak_aqi']):.0f}",
                str(today_row["category"]),
                "blue",
            )
        else:
            pastel_card("TODAY'S PEAK", "—", "No data", "blue")
    with c3:
        pastel_card("NEXT 72H PEAK", f"{next72_peak:.0f}", aqi_category(next72_peak), "peach")
    with c4:
        pastel_card("NEXT 72H MEAN", f"{next72_mean:.0f}", "Average predicted AQI", "lavender")

    if next72_peak >= 301:
        st.error("Hazardous AQI is forecast in Lahore within the next 72 hours (AQI ≥ 301).")
    elif next72_peak >= 201:
        st.warning("Very unhealthy AQI is forecast in Lahore within the next 72 hours.")
    elif next72_peak >= 151:
        st.warning("Unhealthy AQI is forecast in Lahore within the next 72 hours.")

    st.subheader("Day 1 · Day 2 · Day 3 predictions")
    st.caption(
        "The large number is the selected best model's point forecast at +24h, +48h and +72h. "
        "Each card also shows the mean and peak within that 24-hour forecast block."
    )
    st.markdown(
        """
        <div class="method-note">
          <strong>Model behavior:</strong> this version intentionally keeps the original model unchanged.
          Day 1, Day 2 and Day 3 are points on the same recursive hourly forecast, not separate horizon models.
          Earlier predicted AQI values can therefore influence later lag/rolling features.
        </div>
        """,
        unsafe_allow_html=True,
    )
    horizons = [
        ("DAY 1 · +24H", 0, min(24, len(forecast)), 23, "blue"),
        ("DAY 2 · +48H", min(24, len(forecast)), min(48, len(forecast)), 47, "peach"),
        ("DAY 3 · +72H", min(48, len(forecast)), min(72, len(forecast)), 71, "lavender"),
    ]
    hcols = st.columns(3)
    for col, (label, start_idx, end_idx, point_idx, css_class) in zip(hcols, horizons):
        with col:
            if start_idx < len(forecast) and end_idx > start_idx:
                hs = horizon_stats(start_idx, end_idx, point_idx)
                pastel_card(
                    label,
                    f"{hs['point']:.0f} AQI",
                    f"{hs['category']} · mean {hs['mean']:.0f} · peak {hs['peak']:.0f}",
                    css_class,
                )
            else:
                pastel_card(label, "—", "Forecast not available", css_class)

    st.subheader("Today: observed + predicted remaining hours")
    today_timeline = build_today_timeline(recent, forecast, SETTINGS.local_timezone)
    if today_timeline.empty:
        st.info("No current-day timeline is available yet.")
    else:
        fig_today = px.line(
            today_timeline,
            x="local_time",
            y="aqi",
            color="series",
            markers=True,
            labels={"local_time": "Lahore time", "aqi": "AQI", "series": ""},
            color_discrete_map={
                "Recent Open-Meteo": "#7aa8d8",
                "Model forecast": "#b78bd4",
            },
        )
        fig_today.update_layout(
            legend_orientation="h",
            margin=dict(l=20, r=20, t=20, b=20),
            template="streamlit",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_today, use_container_width=True)

    st.subheader("Calendar-day outlook")
    if not daily.empty:
        display_daily = daily[["label", "mean_aqi", "peak_aqi", "min_aqi", "category", "source"]].copy()
        display_daily.columns = ["Day", "Mean AQI", "Peak AQI", "Minimum AQI", "Category", "Source"]
        st.dataframe(display_daily.round(1), use_container_width=True, hide_index=True)

with details_tab:
    plot_df = forecast[["timestamp", "predicted_us_aqi"]].copy()
    plot_df["Lahore time"] = pd.to_datetime(plot_df["timestamp"], utc=True).dt.tz_convert(
        SETTINGS.local_timezone
    )
    fig = px.line(
        plot_df,
        x="Lahore time",
        y="predicted_us_aqi",
        labels={"predicted_us_aqi": "Predicted AQI"},
        title="Lahore AQI forecast — next 72 hours",
        markers=True,
    )
    fig.update_traces(line=dict(color="#8d8fd8", width=3), marker=dict(color="#b7b8ef"))
    fig.update_layout(
        template="streamlit",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    table = forecast.copy()
    table["Lahore time"] = pd.to_datetime(table["timestamp"], utc=True).dt.tz_convert(
        SETTINGS.local_timezone
    )
    st.dataframe(
        table[
            [
                "Lahore time",
                "predicted_us_aqi",
                "category",
                "pm2_5",
                "pm10",
                "ozone",
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
            ]
        ]
        .rename(
            columns={
                "predicted_us_aqi": "AQI",
                "category": "Category",
                "temperature_2m": "Temperature °C",
                "relative_humidity_2m": "Humidity %",
                "wind_speed_10m": "Wind speed",
            }
        )
        .round(1),
        use_container_width=True,
        hide_index=True,
    )

with explain_tab:
    st.subheader("Why did the model make this prediction?")
    st.markdown(
        """
        <div class="shap-note">
          <strong>SHAP explanation:</strong> each SHAP value measures how much a feature pushed this
          particular forecast above or below the model's baseline prediction. Positive values push AQI
          higher; negative values push it lower. This explains the model's behaviour — it does not prove
          that a feature caused the pollution level.<br><br>
          <strong>Recursive-forecast note:</strong> for Day 1/2/3, SHAP explains the model's final one-hour
          decision at that forecast step. Some lag and rolling inputs at that point may already contain
          earlier model predictions, so SHAP is not a decomposition of the entire accumulated 24/48/72-hour path.
        </div>
        """,
        unsafe_allow_html=True,
    )

    options = {
        "Next hour": 0,
        "Day 1 (+24 hours)": min(23, len(forecast) - 1),
        "Day 2 (+48 hours)": min(47, len(forecast) - 1),
        "Day 3 (+72 hours)": min(71, len(forecast) - 1),
    }
    selected_label = st.selectbox("Prediction to explain", list(options.keys()), index=1)
    selected_idx = options[selected_label]
    selected_row = forecast.iloc[selected_idx]
    selected_time = pd.to_datetime(selected_row["timestamp"], utc=True).tz_convert(SETTINGS.local_timezone)
    selected_prediction = float(selected_row["predicted_us_aqi"])

    p1, p2, p3 = st.columns(3)
    p1.metric("Prediction", f"{selected_prediction:.1f} AQI")
    p2.metric("Category", aqi_category(selected_prediction))
    p3.metric("Forecast time", selected_time.strftime("%d %b, %I:%M %p"))

    if st.button("Explain with SHAP", key="run_local_shap"):
        try:
            with st.spinner("Computing a local SHAP explanation for this forecast..."):
                shap_df, base_value, raw_output = local_shap_explanation(selected_idx)

            top = shap_df.head(12).sort_values("SHAP effect")
            colors = ["#9ec5df" if v < 0 else "#e9a6a6" for v in top["SHAP effect"]]
            shap_fig = go.Figure(
                go.Bar(
                    x=top["SHAP effect"],
                    y=top["Feature"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.2f}" for v in top["SHAP effect"]],
                    textposition="outside",
                )
            )
            shap_fig.update_layout(
                title=f"Local SHAP explanation — {selected_label}",
                xaxis_title="Effect on predicted AQI (SHAP value)",
                yaxis_title="",
                template="streamlit",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=30, t=60, b=20),
            )
            shap_fig.add_vline(x=0, line_width=1, line_color="#6d7480")
            st.plotly_chart(shap_fig, use_container_width=True)

            upward = shap_df[shap_df["SHAP effect"] > 0].head(3)
            downward = shap_df[shap_df["SHAP effect"] < 0].head(3)
            up_text = ", ".join(
                f"**{r['Feature']}** ({r['SHAP effect']:+.2f})" for _, r in upward.iterrows()
            ) or "none among the strongest features"
            down_text = ", ".join(
                f"**{r['Feature']}** ({r['SHAP effect']:+.2f})" for _, r in downward.iterrows()
            ) or "none among the strongest features"

            st.markdown(
                f"The model's SHAP baseline for this explanation is approximately **{base_value:.1f} AQI**. "
                f"The strongest upward pushes were {up_text}. The strongest downward pushes were {down_text}. "
                f"Together, all feature contributions produce a raw model output of approximately **{raw_output:.1f} AQI**, "
                f"shown on the dashboard as **{selected_prediction:.1f} AQI** after the forecast pipeline's safety bounds."
            )

            shap_table = shap_df.head(12)[["Feature", "Input value", "SHAP effect"]].copy()
            shap_table["Direction"] = np.where(shap_table["SHAP effect"] >= 0, "Pushes AQI up", "Pushes AQI down")
            st.dataframe(shap_table.round(3), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not compute the local SHAP explanation: {exc}")
            st.info("Run `python -m src.train` once with the updated train.py, then restart Streamlit.")

    shap_path = Path(model_dir) / "shap_summary.png"
    if not shap_path.exists() and Path("artifacts/shap_summary.png").exists():
        shap_path = Path("artifacts/shap_summary.png")
    if shap_path.exists():
        st.subheader("Global SHAP summary")
        st.caption(
            "This beeswarm summarizes which features mattered most across the held-out test period; "
            "the local explanation above explains one specific forecast."
        )
        st.image(str(shap_path), use_container_width=True)

with history_tab:
    st.write(
        "The training and EDA window is always restricted to the latest **six months**. "
        "Older Hopsworks rows, if they exist after long-term operation, are ignored by training."
    )
    if st.button("Load six-month history", key="load_history"):
        try:
            start_time, end_time = SETTINGS.training_window_utc()
            hist = read_raw_features(
                SETTINGS.location_name,
                start_time=start_time,
                end_time=end_time,
            )
            hist["Lahore time"] = pd.to_datetime(hist["timestamp"], utc=True).dt.tz_convert(
                SETTINGS.local_timezone
            )
            daily_hist = hist.set_index("Lahore time")["us_aqi"].resample("D").mean().reset_index()
            hist_fig = px.line(
                daily_hist,
                x="Lahore time",
                y="us_aqi",
                title="Daily mean Lahore AQI — latest six months",
                labels={"us_aqi": "Mean AQI"},
            )
            hist_fig.update_traces(line=dict(color="#7fb7a6", width=2.5))
            hist_fig.update_layout(
                template="streamlit",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(hist_fig, use_container_width=True)
            st.caption(f"Loaded {len(hist):,} hourly raw rows from Hopsworks.")
        except Exception as exc:
            st.error(f"Could not load Hopsworks history: {exc}")

with model_tab:
    st.subheader("Selected best model")
    st.write(f"**{best_model_name.replace('_', ' ').title()}**")
    st.caption(
        f"Registered as **{model_meta.name} v{model_meta.version}**. In each training run, Ridge, Random Forest, "
        "HistGradientBoosting, XGBoost and MLP are compared on the validation period. The model with the "
        "lowest validation RMSE is refit on train+validation and used for every dashboard forecast."
    )

    if metrics:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Held-out RMSE", f"{metrics.get('rmse', float('nan')):.3f}")
        metric_cols[1].metric("Held-out MAE", f"{metrics.get('mae', float('nan')):.3f}")
        metric_cols[2].metric("Held-out R²", f"{metrics.get('r2', float('nan')):.3f}")

    comparison_path = Path(model_dir) / "model_comparison.csv"
    if not comparison_path.exists() and Path("artifacts/model_comparison.csv").exists():
        comparison_path = Path("artifacts/model_comparison.csv")
    if comparison_path.exists():
        st.subheader("Validation model comparison")
        comparison = pd.read_csv(comparison_path).sort_values("val_rmse")
        pretty = comparison.rename(
            columns={
                "model": "Model",
                "val_rmse": "Validation RMSE",
                "val_mae": "Validation MAE",
                "val_r2": "Validation R²",
            }
        )
        st.dataframe(pretty.round(4), use_container_width=True, hide_index=True)
        cmp_fig = px.bar(
            pretty,
            x="Model",
            y="Validation RMSE",
            title="Model selection — lower validation RMSE is better",
            text_auto=".2f",
        )
        cmp_fig.update_traces(marker_color="#b4a7df")
        cmp_fig.update_layout(
            template="streamlit",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(cmp_fig, use_container_width=True)

    st.subheader("Actual vs predicted — held-out test period")
    predictions_path = Path(model_dir) / "test_predictions.csv"
    if not predictions_path.exists() and Path("artifacts/test_predictions.csv").exists():
        predictions_path = Path("artifacts/test_predictions.csv")

    if predictions_path.exists():
        test_df = pd.read_csv(predictions_path)
        test_df["Lahore time"] = pd.to_datetime(test_df["timestamp"], utc=True).dt.tz_convert(
            SETTINGS.local_timezone
        )
        pred_fig = go.Figure()
        pred_fig.add_trace(
            go.Scatter(
                x=test_df["Lahore time"],
                y=test_df["actual_aqi"],
                name="Actual",
                mode="lines",
                line=dict(color="#6fae9b", width=2.5),
            )
        )
        pred_fig.add_trace(
            go.Scatter(
                x=test_df["Lahore time"],
                y=test_df["predicted_aqi"],
                name="Predicted",
                mode="lines",
                line=dict(color="#9d8cc7", width=2.5),
            )
        )
        pred_fig.update_layout(
            xaxis_title="Lahore time",
            yaxis_title="AQI",
            legend_orientation="h",
            template="streamlit",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(pred_fig, use_container_width=True)
    else:
        test_plot = Path(model_dir) / "test_predictions.png"
        if test_plot.exists():
            st.image(str(test_plot), use_container_width=True)
            st.caption("Retrain once with the updated code to get the interactive version of this chart.")
        else:
            st.info("Retrain once to generate the held-out Actual vs Predicted chart.")

    shap_path = Path(model_dir) / "shap_summary.png"
    if not shap_path.exists() and Path("artifacts/shap_summary.png").exists():
        shap_path = Path("artifacts/shap_summary.png")
    if shap_path.exists():
        st.subheader("Global SHAP feature importance")
        st.image(str(shap_path), use_container_width=True)

    with st.expander("Training metadata"):
        st.json(metrics)

st.divider()
st.caption(
    f"Selected model: {best_model_name.replace('_', ' ').title()} · Registry: {model_meta.name} v{model_meta.version} · "
    "Data: Open-Meteo weather + CAMS air quality · Forecasts are informational and are not official local health advisories."
)
