from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Always save report/SHAP figures with a high-contrast light palette.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "axes.edgecolor": "#4b5563",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "legend.labelcolor": "#111111",
})
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import SETTINGS
from .features import MODEL_FEATURES, TARGET
from .hopsworks_io import read_training_features, register_sklearn_model


def make_models() -> dict[str, object]:
    return {
        "ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        l2_regularization=0.2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=500,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "mlp_neural_network": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        early_stopping=True,
                        max_iter=300,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_shap_summary(model, X: pd.DataFrame, out_path: Path) -> None:
    try:
        import shap

        sample = X.tail(min(40, len(X))).copy()
        background = X.iloc[:: max(1, len(X) // 50)].head(50).copy()
        explainer = shap.Explainer(model.predict, background)
        values = explainer(sample, max_evals=2 * X.shape[1] + 1)
        shap.plots.beeswarm(values, max_display=15, show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor("white")
        for ax in fig.axes:
            ax.set_facecolor("white")
            ax.tick_params(colors="#111111")
            ax.xaxis.label.set_color("#111111")
            ax.yaxis.label.set_color("#111111")
            ax.title.set_color("#111111")
            for text in ax.texts:
                text.set_color("#111111")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        out_path.with_suffix(".error.txt").write_text(str(exc))


def main() -> None:
    start_time, end_time = SETTINGS.training_window_utc()
    print(
        f"Reading rolling {SETTINGS.history_months}-month training window: "
        f"{start_time.isoformat()} -> {end_time.isoformat()}"
    )

    df = read_training_features(start_time, end_time)
    if df.empty:
        raise RuntimeError("No training rows found. Run: python -m src.backfill")

    required = MODEL_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Training data is missing columns: {missing}")

    df = (
        df.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .dropna(subset=required)
        .reset_index(drop=True)
    )
    if len(df) < 500:
        raise RuntimeError(
            f"Only {len(df)} usable hourly rows are available. "
            "A six-month backfill should normally provide several thousand rows."
        )

    X = df[MODEL_FEATURES]
    y = df[TARGET]

    # Chronological split avoids leaking future information backwards in time.
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    rows: list[dict] = []
    fitted: dict[str, object] = {}
    for name, model in make_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        pred = model.predict(X_val)
        metrics = regression_metrics(y_val, pred)
        rows.append({"model": name, **{f"val_{k}": v for k, v in metrics.items()}})

    comparison = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted[best_name]

    # Refit only after model selection, then evaluate once on the newest held-out period.
    best_model.fit(X.iloc[:val_end], y.iloc[:val_end])
    test_pred = best_model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_pred)

    metrics = {
        **test_metrics,
        "best_model": best_name,
        "history_months": SETTINGS.history_months,
        "window_start_utc": start_time.isoformat(),
        "window_end_utc": end_time.isoformat(),
        "usable_rows": int(n),
        "train_plus_validation_rows": int(val_end),
        "test_rows": int(len(X_test)),
    }

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    comparison.to_csv(artifacts / "model_comparison.csv", index=False)
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_n = min(500, len(y_test))
    test_times = df["timestamp"].iloc[val_end:]
    plt.figure(figsize=(11, 4.5))
    plt.plot(test_times.iloc[-plot_n:], y_test.iloc[-plot_n:].to_numpy(), label="Actual")
    plt.plot(test_times.iloc[-plot_n:], test_pred[-plot_n:], label="Predicted")
    plt.title(f"Held-out Lahore AQI forecast — {best_name}")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.legend()
    plt.tight_layout()
    test_plot = artifacts / "test_predictions.png"
    plt.savefig(test_plot, dpi=160)
    plt.close()

    # Store held-out predictions as data as well as an image so the Streamlit app
    # can render an interactive Actual vs Predicted chart.
    test_predictions_csv = artifacts / "test_predictions.csv"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(test_times, utc=True),
            "actual_aqi": y_test.to_numpy(),
            "predicted_aqi": test_pred,
        }
    ).to_csv(test_predictions_csv, index=False)

    shap_plot = artifacts / "shap_summary.png"
    save_shap_summary(best_model, X_test, shap_plot)

    # Persist a small, deterministic background sample for local SHAP explanations
    # in Streamlit. This lets the app explain a specific +1h/+24h/+48h/+72h
    # prediction without loading the full six-month training set again.
    background_count = min(80, len(X.iloc[:val_end]))
    step = max(1, len(X.iloc[:val_end]) // max(1, background_count))
    shap_background = X.iloc[:val_end:step].head(background_count).copy()
    shap_background_path = artifacts / "shap_background.csv"
    shap_background.to_csv(shap_background_path, index=False)

    registered = register_sklearn_model(
        best_model,
        metrics=metrics,
        feature_names=MODEL_FEATURES,
        extra_artifacts=[comparison_path for comparison_path in [
            artifacts / "model_comparison.csv",
            artifacts / "metrics.json",
            test_predictions_csv,
            test_plot,
            shap_plot,
            shap_background_path,
        ] if comparison_path.exists()],
    )

    print("\nModel comparison (validation):")
    print(comparison.to_string(index=False))
    print("\nHeld-out test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nRegistered Hopsworks model: {registered.name} v{registered.version}")


if __name__ == "__main__":
    main()
