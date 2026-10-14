from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.config.settings import get_settings
from app.data.storage import load_parquet
from app.inference.load_model import get_model_file_path, get_model_metadata_path
from app.monitoring.forecast_metrics import load_latest_prediction_file
from app.utils.database import (
    get_recent_api_forecast_logs,
    get_recent_batch_runs,
    get_recent_data_quality_runs,
    get_recent_monitoring_runs,
    initialize_database,
)

st.set_page_config(
    page_title="M5 Forecasting Monitoring Dashboard",
    page_icon="📈",
    layout="wide",
)

settings = get_settings()
initialize_database()


@st.cache_data
def load_model_metadata() -> dict:
    metadata_path = get_model_metadata_path()
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_latest_batch_predictions() -> pd.DataFrame:
    try:
        prediction_file = load_latest_prediction_file()
        return load_parquet(prediction_file)
    except Exception:
        return pd.DataFrame()


def render_model_overview() -> None:
    st.subheader("Model Overview")

    metadata = load_model_metadata()
    model_path = get_model_file_path()
    metadata_path = get_model_metadata_path()

    col1, col2, col3, col4 = st.columns(4)

    metrics = metadata.get("metrics", {})
    feature_columns = metadata.get("feature_columns", [])

    col1.metric("Model Name", metadata.get("model_name", "N/A"))
    col2.metric("Validation RMSE", f"{metrics.get('rmse', 'N/A')}")
    col3.metric("Validation MAE", f"{metrics.get('mae', 'N/A')}")
    col4.metric("Feature Count", len(feature_columns))

    st.markdown("**Artifacts**")
    st.write(f"Model file exists: `{model_path.exists()}`")
    st.write(f"Metadata file exists: `{metadata_path.exists()}`")

    with st.expander("Model Metadata"):
        st.json(metadata if metadata else {"message": "No metadata found"})


def render_batch_runs() -> None:
    st.subheader("Recent Batch Forecast Runs")
    batch_runs_df = get_recent_batch_runs(limit=10)

    if batch_runs_df.empty:
        st.info("No batch forecast runs found.")
        return

    st.dataframe(batch_runs_df, width="stretch")


def render_monitoring_summary() -> None:
    st.subheader("Monitoring Summary")

    monitoring_df = get_recent_monitoring_runs(limit=10)
    dq_df = get_recent_data_quality_runs(limit=10)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Recent Monitoring Runs**")
        if monitoring_df.empty:
            st.info("No monitoring runs found.")
        else:
            latest = monitoring_df.iloc[0]
            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "Latest RMSE",
                f"{latest['rmse']:.4f}" if pd.notna(latest["rmse"]) else "N/A",
            )
            metric_cols[1].metric(
                "Latest MAE",
                f"{latest['mae']:.4f}" if pd.notna(latest["mae"]) else "N/A",
            )
            metric_cols[2].metric(
                "Latest sMAPE",
                f"{latest['smape']:.2f}" if pd.notna(latest["smape"]) else "N/A",
            )
            metric_cols[3].metric(
                "Latest Bias",
                f"{latest['bias']:.4f}" if pd.notna(latest["bias"]) else "N/A",
            )

            st.dataframe(monitoring_df, width="stretch")

    with col2:
        st.markdown("**Recent Data Quality Runs**")
        if dq_df.empty:
            st.info("No data quality runs found.")
        else:
            latest_dq = dq_df.iloc[0]
            dq_cols = st.columns(3)
            dq_cols[0].metric("Row Count", int(latest_dq["row_count"]))
            dq_cols[1].metric("Missing Values", int(latest_dq["missing_value_count"]))
            dq_cols[2].metric("Duplicates", int(latest_dq["duplicate_count"]))

            st.dataframe(dq_df, width="stretch")


def render_api_logs() -> None:
    st.subheader("Recent API Forecast Logs")
    api_logs_df = get_recent_api_forecast_logs(limit=20)

    if api_logs_df.empty:
        st.info("No API forecast logs found.")
        return

    st.dataframe(api_logs_df, width="stretch")


def render_latest_predictions() -> None:
    st.subheader("Latest Batch Prediction Preview")
    latest_predictions_df = load_latest_batch_predictions()

    if latest_predictions_df.empty:
        st.info("No prediction parquet file found.")
        return

    st.dataframe(latest_predictions_df.head(50), width="stretch")


def render_drift_report_info() -> None:
    st.subheader("Drift Report")
    drift_report_path = settings.artifacts_dir / "drift_report.html"

    st.write(f"Expected drift report path: `{drift_report_path}`")
    st.write(f"Drift report exists: `{drift_report_path.exists()}`")

    if drift_report_path.exists():
        st.success("Drift report has been generated successfully.")
    else:
        st.warning(
            "Drift report file not found yet. Run the monitoring pipeline first."
        )


def main() -> None:
    st.title("M5 Demand Forecasting Monitoring Dashboard")
    st.caption(
        "Production-style monitoring dashboard for the Walmart M5 forecasting platform"
    )

    render_model_overview()
    st.divider()

    render_batch_runs()
    st.divider()

    render_monitoring_summary()
    st.divider()

    render_api_logs()
    st.divider()

    render_latest_predictions()
    st.divider()

    render_drift_report_info()


if __name__ == "__main__":
    main()
