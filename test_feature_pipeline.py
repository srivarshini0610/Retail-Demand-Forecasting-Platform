import pandas as pd

from app.features.calendar_features import add_calendar_features
from app.features.lag_features import add_lag_features
from app.features.price_features import add_price_features


def test_calendar_features_add_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2016-01-01", "2016-01-02"]),
            "event_name_1": [None, "Holiday"],
            "event_type_1": [None, "Cultural"],
            "snap_CA": [0, 1],
            "snap_TX": [0, 0],
            "snap_WI": [1, 1],
        }
    )

    out = add_calendar_features(df)

    assert "day_of_week" in out.columns
    assert "is_weekend" in out.columns
    assert "has_event" in out.columns


def test_lag_features_create_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "store_id": ["CA_1"] * 40,
            "item_id": ["ITEM_1"] * 40,
            "date": pd.date_range("2016-01-01", periods=40, freq="D"),
            "sales": list(range(40)),
        }
    )

    out = add_lag_features(df)

    assert "sales_lag_1" in out.columns
    assert "sales_lag_7" in out.columns
    assert "sales_roll_mean_7" in out.columns


def test_price_features_create_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "store_id": ["CA_1"] * 40,
            "item_id": ["ITEM_1"] * 40,
            "date": pd.date_range("2016-01-01", periods=40, freq="D"),
            "sell_price": [10.0] * 40,
        }
    )

    out = add_price_features(df)

    assert "price_lag_1" in out.columns
    assert "price_change_1" in out.columns
    assert "price_roll_mean_7" in out.columns
