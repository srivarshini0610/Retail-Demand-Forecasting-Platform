from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_cached_feature_table, get_cached_model_bundle
from app.api.schemas.forecast_request import (
    BatchForecastRequest,
    ItemStoreForecastRequest,
)
from app.api.schemas.forecast_response import (
    BatchForecastResponse,
    ItemStoreForecastResponse,
)
from app.inference.batch_forecast import run_batch_forecast
from app.inference.predict import generate_item_store_forecast
from app.utils.common import utc_timestamp
from app.utils.database import insert_api_forecast_log

router = APIRouter()


@router.post("/forecast/item-store", response_model=ItemStoreForecastResponse)
def forecast_item_store(
    request: ItemStoreForecastRequest,
    model_bundle=Depends(get_cached_model_bundle),
    feature_df=Depends(get_cached_feature_table),
) -> ItemStoreForecastResponse:
    try:
        result = generate_item_store_forecast(
            item_id=request.item_id,
            store_id=request.store_id,
            forecast_date=request.forecast_date,
            model_bundle=model_bundle,
            feature_df=feature_df,
        )

        insert_api_forecast_log(
            {
                "timestamp": utc_timestamp(),
                "item_id": result["item_id"],
                "store_id": result["store_id"],
                "feature_row_date": result["feature_row_date"],
                "target_date": result["target_date"],
                "forecast_horizon_days": result["forecast_horizon_days"],
                "predicted_sales": result["predicted_sales"],
                "model_name": result["model_name"],
            }
        )

        return ItemStoreForecastResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected inference error: {str(e)}"
        )


@router.post("/forecast/batch", response_model=BatchForecastResponse)
def forecast_batch(
    request: BatchForecastRequest,
    model_bundle=Depends(get_cached_model_bundle),
    feature_df=Depends(get_cached_feature_table),
) -> BatchForecastResponse:
    try:
        output_path = run_batch_forecast(
            model_bundle=model_bundle,
            feature_df=feature_df,
            limit=request.limit,
        )
        return BatchForecastResponse(
            status="success",
            output_path=str(output_path),
            requested_limit=request.limit,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected batch inference error: {str(e)}"
        )
