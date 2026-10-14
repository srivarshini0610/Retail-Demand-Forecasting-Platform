from __future__ import annotations

import pandas as pd

from app.config.logging_config import get_logger

logger = get_logger()


SALES_REQUIRED_COLUMNS = {
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
}

CALENDAR_REQUIRED_COLUMNS = {
    "date",
    "wm_yr_wk",
    "d",
    "weekday",
    "month",
    "year",
}

PRICES_REQUIRED_COLUMNS = {
    "store_id",
    "item_id",
    "wm_yr_wk",
    "sell_price",
}


def _check_required_columns(
    df: pd.DataFrame, required_columns: set[str], df_name: str
) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def validate_sales_data(sales_df: pd.DataFrame) -> None:
    _check_required_columns(sales_df, SALES_REQUIRED_COLUMNS, "sales_df")

    day_columns = [col for col in sales_df.columns if col.startswith("d_")]
    if not day_columns:
        raise ValueError("sales_df does not contain day columns like d_1, d_2, ...")

    if sales_df.duplicated(subset=["id"]).any():
        logger.warning("Duplicate ids found in sales_df")

    logger.info("sales_df validation passed")


def validate_calendar_data(calendar_df: pd.DataFrame) -> None:
    _check_required_columns(calendar_df, CALENDAR_REQUIRED_COLUMNS, "calendar_df")

    if calendar_df["d"].duplicated().any():
        raise ValueError("calendar_df contains duplicate 'd' keys")

    logger.info("calendar_df validation passed")


def validate_prices_data(prices_df: pd.DataFrame) -> None:
    _check_required_columns(prices_df, PRICES_REQUIRED_COLUMNS, "prices_df")

    if (prices_df["sell_price"] < 0).any():
        raise ValueError("prices_df contains negative sell_price values")

    logger.info("prices_df validation passed")


def validate_raw_m5_data(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> None:
    validate_sales_data(sales_df)
    validate_calendar_data(calendar_df)
    validate_prices_data(prices_df)

    logger.info("All raw dataset validations passed")
