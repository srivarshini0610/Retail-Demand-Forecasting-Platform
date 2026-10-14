from __future__ import annotations

import json
from pathlib import Path

import joblib

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.utils.paths import get_artifact_path

logger = get_logger()


def get_model_artifact_dir() -> Path:
    settings = get_settings()
    model_dir = get_artifact_path(settings.model_artifact_dir_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_model_bundle(
    model,
    feature_columns: list[str],
    metrics: dict[str, float],
    model_name: str = "lightgbm_model",
) -> dict[str, Path]:
    model_dir = get_model_artifact_dir()

    model_path = model_dir / f"{model_name}.joblib"
    metadata_path = model_dir / f"{model_name}_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "metrics": metrics,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Saved model bundle | model_path={} | metadata_path={}",
        model_path,
        metadata_path,
    )

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
    }
