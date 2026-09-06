from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from .config import SETTINGS
from .features import TARGET


def get_project():
    import hopsworks

    # Hopsworks' Windows Kafka/PEM helper can use the root-relative \tmp path.
    # Ensure it exists so Feature Store writes do not fail on a fresh Windows setup.
    if os.name == "nt":
        try:
            Path(r"C:\tmp").mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    kwargs = {"engine": "python"}
    if os.getenv("HOPSWORKS_PROJECT"):
        kwargs["project"] = os.environ["HOPSWORKS_PROJECT"]
    if os.getenv("HOPSWORKS_API_KEY"):
        kwargs["api_key_value"] = os.environ["HOPSWORKS_API_KEY"]
    if os.getenv("HOPSWORKS_HOST"):
        kwargs["host"] = os.environ["HOPSWORKS_HOST"]
    if os.getenv("HOPSWORKS_CERT_FOLDER"):
        kwargs["cert_folder"] = os.environ["HOPSWORKS_CERT_FOLDER"]
    return hopsworks.login(**kwargs)


def get_raw_feature_group(project=None):
    project = project or get_project()
    fs = project.get_feature_store()
    return fs.get_or_create_feature_group(
        name=SETTINGS.raw_feature_group_name,
        version=SETTINGS.raw_feature_group_version,
        description=(
            "Lahore hourly Open-Meteo raw weather, pollutant concentrations, and AQI. "
            "The project trains only on the latest six months."
        ),
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
        stream=True,
        statistics_config=False,
    )


def get_engineered_feature_group(project=None):
    project = project or get_project()
    fs = project.get_feature_store()
    return fs.get_or_create_feature_group(
        name=SETTINGS.feature_group_name,
        version=SETTINGS.feature_group_version,
        description=(
            "Lahore one-hour-ahead AQI training rows with lag, rolling, calendar, "
            "and next-hour Open-Meteo exogenous features."
        ),
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
        stream=True,
        statistics_config=False,
    )


def ensure_feature_view(project=None):
    """Create the Hopsworks Feature View used by the training pipeline."""
    project = project or get_project()
    fs = project.get_feature_store()
    fg = get_engineered_feature_group(project)
    return fs.get_or_create_feature_view(
        name=SETTINGS.feature_view_name,
        version=SETTINGS.feature_view_version,
        query=fg.select_all(),
        labels=[TARGET],
        description=(
            "Rolling six-month Lahore AQI training view. The label is next-hour AQI."
        ),
    )


def _prepare_timestamp_for_hopsworks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_localize(None)
    return out


def insert_raw(raw_df: pd.DataFrame, city: str | None = None, project=None) -> None:
    if raw_df.empty:
        return
    df = raw_df.copy()
    df["city"] = city or SETTINGS.location_name
    df = _prepare_timestamp_for_hopsworks(df)
    fg = get_raw_feature_group(project)
    fg.insert(df, operation="upsert", wait=True)


def insert_engineered(df: pd.DataFrame, project=None) -> None:
    if df.empty:
        return
    df = _prepare_timestamp_for_hopsworks(df)
    fg = get_engineered_feature_group(project)
    fg.insert(df, operation="upsert", wait=True)


def _normalize_read(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _filter_window(
    df: pd.DataFrame,
    start_time: datetime | None,
    end_time: datetime | None,
) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    if start_time is not None:
        start = pd.Timestamp(start_time)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        df = df[df["timestamp"] >= start]
    if end_time is not None:
        end = pd.Timestamp(end_time)
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        df = df[df["timestamp"] <= end]
    return df.reset_index(drop=True)


def read_raw_features(
    city: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    project=None,
) -> pd.DataFrame:
    fg = get_raw_feature_group(project)
    try:
        df = fg.read()
    except Exception:
        # Hopsworks free tier occasionally has Arrow Flight issues; Hive is slower but robust.
        df = fg.read(read_options={"use_hive": True})
    df = _normalize_read(df)
    if city is not None and "city" in df.columns:
        df = df[df["city"].astype(str).str.lower() == city.lower()]
    return _filter_window(df, start_time, end_time)


def read_training_features(
    start_time: datetime,
    end_time: datetime,
    project=None,
) -> pd.DataFrame:
    """Read labelled training rows through the Hopsworks Feature View."""
    project = project or get_project()
    fv = ensure_feature_view(project)

    try:
        # training_data() returns labels separately; get_batch_data() is intended
        # for inference and can omit the Feature View label.
        X, y = fv.training_data(
            start_time=start_time,
            end_time=end_time,
            description=(
                f"Rolling {SETTINGS.history_months}-month Lahore AQI training window"
            ),
            event_time=True,
        )
        X = X.reset_index(drop=True)
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.rename(TARGET).to_frame()
            else:
                y = y.reset_index(drop=True)
                if TARGET not in y.columns and len(y.columns) == 1:
                    y.columns = [TARGET]
            df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        else:
            df = X
    except Exception as exc:
        print(
            f"Feature View training read failed: {exc}\n"
            "Falling back to engineered Feature Group..."
        )
        fg = get_engineered_feature_group(project)
        try:
            df = fg.read()
        except Exception:
            df = fg.read(read_options={"use_hive": True})

    df = _normalize_read(df)
    if "city" in df.columns:
        df = df[df["city"].astype(str).str.lower() == SETTINGS.location_name.lower()]
    return _filter_window(df, start_time, end_time)


def register_sklearn_model(
    model,
    metrics: dict,
    feature_names: list[str],
    model_name: str | None = None,
    extra_artifacts: list[Path] | None = None,
    project=None,
) -> object:
    project = project or get_project()
    mr = project.get_model_registry()
    model_name = model_name or SETTINGS.model_name

    artifact_dir = Path("artifacts/model")
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.pkl")
    metadata = {
        "feature_names": feature_names,
        "metrics": metrics,
        "location": SETTINGS.location_name,
        "history_months": SETTINGS.history_months,
        "forecast_hours": SETTINGS.forecast_hours,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    for path in extra_artifacts or []:
        if path.exists():
            shutil.copy2(path, artifact_dir / path.name)

    registry_metrics = {
        k: float(v)
        for k, v in metrics.items()
        if k in {"rmse", "mae", "r2"} and isinstance(v, (int, float))
    }
    registered = mr.sklearn.create_model(
        model_name,
        metrics=registry_metrics,
        description=(
            "Lahore AQI forecaster trained only on the latest six months. "
            "It predicts one hour ahead and is used recursively with Open-Meteo "
            "weather/pollutant forecasts for the current day and next 72 hours."
        ),
    )
    return registered.save(str(artifact_dir))


def load_best_model(model_name: str | None = None, project=None):
    """Load the winner from the most recent training run.

    Each call to ``src.train`` compares all candidate algorithms using validation
    RMSE and registers only that run's winning model. Therefore the newest
    registered version is the best model from the newest six-month experiment.

    When running locally immediately after training, prefer the local artifact.
    This avoids a Hopsworks 5.0.x external-client download issue seen on Windows
    (``Client object has no attribute '_project_id'``). If local artifacts are not
    available, fall back to the normal Hopsworks Model Registry download API.
    """
    project = project or get_project()
    mr = project.get_model_registry()
    model_name = model_name or SETTINGS.model_name

    models = mr.get_models(model_name)
    if not models:
        raise RuntimeError(f"No Hopsworks model found with name '{model_name}'")

    # src.train already selected the best candidate before registration.
    # Use the newest registered winner so the dashboard matches the latest run.
    model_meta = max(models, key=lambda m: int(m.version))

    local_dir = Path("artifacts/model")
    local_model = local_dir / "model.pkl"
    local_metadata = local_dir / "metadata.json"

    if local_model.exists() and local_metadata.exists():
        model = joblib.load(local_model)
        metadata = json.loads(local_metadata.read_text())
        metadata.setdefault("model_source", "local artifact from latest training run")
        return model, metadata, model_meta, local_dir

    # Useful for deployed environments where the local training artifact is absent.
    try:
        local_dir = Path(model_meta.download())
    except AttributeError as exc:
        if "_project_id" in str(exc):
            raise RuntimeError(
                "Hopsworks could not download the registered model with the current "
                "Windows external client. Run `python -m src.train` once in this "
                "project folder so `artifacts/model/model.pkl` exists, then restart "
                "Streamlit."
            ) from exc
        raise

    model = joblib.load(local_dir / "model.pkl")
    metadata = json.loads((local_dir / "metadata.json").read_text())
    metadata.setdefault("model_source", "Hopsworks Model Registry")
    return model, metadata, model_meta, local_dir
