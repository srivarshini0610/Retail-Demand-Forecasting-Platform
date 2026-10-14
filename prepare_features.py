from __future__ import annotations

import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.data.storage import load_parquet
from app.utils.paths import get_processed_file_path

logger = get_logger()


def load_feature_table_for_inference() -> pd.DataFrame:
    settings = get_settings()
    feature_path = get_processed_file_path(settings.feature_table_name)

    logger.info("Loading feature table for inference from {}", feature_path)
    df = load_parquet(feature_path)

    if not pd.api.types.is_datetime64_any_dtype(df[settings.feature_date_column]):
        df[settings.feature_date_column] = pd.to_datetime(
            df[settings.feature_date_column]
        )

    return df


def get_latest_feature_row(
    feature_df: pd.DataFrame,
    item_id: str,
    store_id: str,
    forecast_date: str | None = None,
) -> pd.DataFrame:
    settings = get_settings()

    filtered = feature_df[
        (feature_df["item_id"].astype(str) == item_id)
        & (feature_df["store_id"].astype(str) == store_id)
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No feature rows found for item_id={item_id} and store_id={store_id}"
        )

    if forecast_date:
        forecast_date_ts = pd.to_datetime(forecast_date)
        filtered = filtered[
            filtered[settings.feature_date_column] <= forecast_date_ts
        ].copy()

        if filtered.empty:
            raise ValueError(
                f"No feature rows found on or before forecast_date={forecast_date} "
                f"for item_id={item_id} and store_id={store_id}"
            )

    latest_row = (
        filtered.sort_values(settings.feature_date_column)
        .tail(1)
        .reset_index(drop=True)
    )

    logger.info(
        "Prepared latest feature row for inference | item_id={} | store_id={} | date={}",
        item_id,
        store_id,
        latest_row.loc[0, settings.feature_date_column],
    )

    return latest_row


def extract_model_input(
    feature_row: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    missing_columns = [col for col in feature_columns if col not in feature_row.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required feature columns for inference: {missing_columns}"
        )

    model_input = feature_row[feature_columns].copy()
    return model_input
