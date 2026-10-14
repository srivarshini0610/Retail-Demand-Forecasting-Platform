from fastapi import APIRouter, HTTPException

from app.inference.load_model import load_model_metadata

router = APIRouter()


@router.get("/model-info")
def get_model_info() -> dict:
    try:
        metadata = load_model_metadata()
        return metadata
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
