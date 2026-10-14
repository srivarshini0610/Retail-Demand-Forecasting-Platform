from __future__ import annotations

import pandas as pd

from app.config.logging_config import get_logger

logger = get_logger()

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def melt_sales_to_long(sales_df: pd.DataFrame) -> pd.DataFrame:
    day_columns = [col for col in sales_df.columns if col.startswith("d_")]

    sales_long = sales_df.melt(
        id_vars=ID_COLUMNS,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    logger.info(f"Sales melted to long format | shape={sales_long.shape}")
    return sales_long


def preprocess_calendar(calendar_df: pd.DataFrame) -> pd.DataFrame:
    calendar_df = calendar_df.copy()
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])

    logger.info(f"Calendar preprocessed | shape={calendar_df.shape}")
    return calendar_df


def preprocess_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    prices_df = prices_df.copy()
    logger.info(f"Prices preprocessed | shape={prices_df.shape}")
    return prices_df


def build_base_table(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    sales_long = melt_sales_to_long(sales_df)
    calendar_df = preprocess_calendar(calendar_df)
    prices_df = preprocess_prices(prices_df)

    merged_df = sales_long.merge(
        calendar_df,
        on="d",
        how="left",
        validate="many_to_one",
    )

    merged_df = merged_df.merge(
        prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )

    merged_df = merged_df.sort_values(["store_id", "item_id", "date"]).reset_index(
        drop=True
    )

    logger.info(f"Base table built successfully | shape={merged_df.shape}")
    return merged_df
