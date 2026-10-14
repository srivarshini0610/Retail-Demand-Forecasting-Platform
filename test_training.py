import pandas as pd

from app.training.datasets import build_train_valid_split


def test_build_train_valid_split() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store_id": ["CA_1"] * 60,
            "item_id": ["ITEM_1"] * 60,
            "sales_lag_28": [1.0] * 60,
            "sales_roll_mean_7": [1.0] * 60,
            "sales_roll_mean_28": [1.0] * 60,
            "target": [2.0] * 60,
        }
    )

    split = build_train_valid_split(df)

    assert len(split.train_df) > 0
    assert len(split.valid_df) > 0
    assert "sales_lag_28" in split.feature_columns
