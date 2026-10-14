from app.config.logging_config import configure_logging
from app.inference.batch_forecast import run_batch_forecast
from app.inference.load_model import load_model_bundle
from app.inference.prepare_features import load_feature_table_for_inference
from app.utils.database import initialize_database
from app.utils.paths import ensure_directories


def main() -> None:
    configure_logging()

    from app.config.logging_config import get_logger

    logger = get_logger()

    ensure_directories()
    initialize_database()

    logger.info("Loading cached objects for standalone batch forecast")
    model_bundle = load_model_bundle()
    feature_df = load_feature_table_for_inference()

    logger.info("Starting standalone batch forecast run")
    output_path = run_batch_forecast(
        model_bundle=model_bundle,
        feature_df=feature_df,
        limit=100,
    )
    logger.info("Standalone batch forecast completed: {}", output_path)


if __name__ == "__main__":
    main()
