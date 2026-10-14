from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "M5 Demand Forecasting Platform"
    environment: str = "dev"
    debug: bool = True

    root_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path | None = None
    raw_data_dir: Path | None = None
    interim_data_dir: Path | None = None
    processed_data_dir: Path | None = None
    predictions_dir: Path | None = None
    artifacts_dir: Path | None = None
    logs_dir: Path | None = None
    mlruns_dir: Path | None = None

    database_url: str = "sqlite:///artifacts/metadata.db"
    mlflow_tracking_uri: str | None = None
    log_level: str = "INFO"

    sales_file_name: str = "sales_train_validation.csv"
    calendar_file_name: str = "calendar.csv"
    prices_file_name: str = "sell_prices.csv"

    processed_base_table_name: str = "base_table.parquet"
    feature_table_name: str = "feature_table.parquet"
    feature_build_start_date: str | None = None
    history_days_for_training: int = 365
    forecast_horizon_days: int = 28

    feature_date_column: str = "date"
    target_column: str = "target"
    model_artifact_dir_name: str = "models"
    train_validation_days: int = 28
    mlflow_experiment_name: str = "m5_demand_forecasting"
    random_state: int = 42

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    model_file_name: str = "lightgbm_model.joblib"
    model_metadata_file_name: str = "lightgbm_model_metadata.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context) -> None:
        if self.data_dir is None:
            self.data_dir = self.root_dir / "data"
        if self.raw_data_dir is None:
            self.raw_data_dir = self.data_dir / "raw"
        if self.interim_data_dir is None:
            self.interim_data_dir = self.data_dir / "interim"
        if self.processed_data_dir is None:
            self.processed_data_dir = self.data_dir / "processed"
        if self.predictions_dir is None:
            self.predictions_dir = self.data_dir / "predictions"
        if self.artifacts_dir is None:
            self.artifacts_dir = self.root_dir / "artifacts"
        if self.logs_dir is None:
            self.logs_dir = self.root_dir / "logs"
        if self.mlruns_dir is None:
            self.mlruns_dir = self.root_dir / "mlruns"

        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = self.mlruns_dir.resolve().as_uri()


@lru_cache
def get_settings() -> Settings:
    return Settings()
