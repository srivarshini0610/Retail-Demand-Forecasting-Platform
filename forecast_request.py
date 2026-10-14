from pydantic import BaseModel, Field


class ItemStoreForecastRequest(BaseModel):
    item_id: str = Field(..., description="Item identifier from the M5 dataset")
    store_id: str = Field(..., description="Store identifier from the M5 dataset")
    forecast_date: str | None = Field(
        default=None,
        description="Optional cutoff date (YYYY-MM-DD). Uses latest available row on or before this date.",
    )


class BatchForecastRequest(BaseModel):
    limit: int | None = Field(
        default=100,
        ge=1,
        description="Optional number of item-store pairs to forecast in batch mode",
    )
