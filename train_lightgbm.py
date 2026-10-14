from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.training.backtest import evaluate_multiple_prediction_sets
from app.training.baseline_models import run_baseline_models
from app.training.datasets import DatasetSplit
from app.training.evaluate import evaluate_predictions
from app.training.register_model import save_model_bundle

logger = get_logger()


@dataclass
class TrainingResult:
    model: lgb.LGBMRegressor
    metrics: dict[str, float]
    baseline_results: pd.DataFrame
    artifact_paths: dict
    feature_columns: list[str]


def cast_categorical_columns(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    categorical_candidates = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "weekday",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]
    categorical_cols = [col for col in categorical_candidates if col in feature_columns]

    for col in categorical_cols:
        train_df[col] = train_df[col].astype("category")
        valid_df[col] = valid_df[col].astype("category")

    return train_df, valid_df


def train_lightgbm_model(split: DatasetSplit) -> TrainingResult:
    settings = get_settings()

    logger.info("Using MLflow tracking URI: {}", settings.mlflow_tracking_uri)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    train_df, valid_df = cast_categorical_columns(
        split.train_df,
        split.valid_df,
        split.feature_columns,
    )

    X_train = train_df[split.feature_columns]
    y_train = train_df[split.target_column].astype("float32")

    X_valid = valid_df[split.feature_columns]
    y_valid = valid_df[split.target_column].astype("float32")

    baseline_predictions = run_baseline_models(valid_df)
    baseline_results = evaluate_multiple_prediction_sets(
        y_true=y_valid, prediction_dict=baseline_predictions
    )

    logger.info("Best baseline by RMSE: {}", baseline_results.iloc[0].to_dict())

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": settings.random_state,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="lightgbm_training"):
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("valid_rows", len(valid_df))
        mlflow.log_param("num_features", len(split.feature_columns))

        for _, row in baseline_results.iterrows():
            prefix = row["model_name"]
            mlflow.log_metric(f"{prefix}_rmse", float(row["rmse"]))
            mlflow.log_metric(f"{prefix}_mae", float(row["mae"]))
            mlflow.log_metric(f"{prefix}_smape", float(row["smape"]))
            mlflow.log_metric(f"{prefix}_bias", float(row["bias"]))

        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50),
            ],
        )

        preds = pd.Series(model.predict(X_valid), index=valid_df.index, dtype="float32")
        metrics = evaluate_predictions(y_true=y_valid, y_pred=preds)

        mlflow.log_metrics(
            {
                "lightgbm_rmse": metrics["rmse"],
                "lightgbm_mae": metrics["mae"],
                "lightgbm_smape": metrics["smape"],
                "lightgbm_bias": metrics["bias"],
            }
        )

        mlflow.lightgbm.log_model(model, artifact_path="model")

        artifact_paths = save_model_bundle(
            model=model,
            feature_columns=split.feature_columns,
            metrics=metrics,
            model_name="lightgbm_model",
        )

    logger.info("LightGBM training complete | metrics={}", metrics)

    return TrainingResult(
        model=model,
        metrics=metrics,
        baseline_results=baseline_results,
        artifact_paths=artifact_paths,
        feature_columns=split.feature_columns,
    )
