.PHONY: format lint test ingest features train api batch monitor dashboard flow-features flow-training flow-batch flow-monitor flow-full

format:
	uv run black app scripts tests

lint:
	uv run ruff check app scripts tests

lint-fix:
	uv run ruff check --fix app scripts tests

test:
	uv run pytest -q

ingest:
	uv run python -m scripts.run_ingestion

features:
	uv run python -m scripts.run_feature_build

train:
	uv run python -m scripts.run_training

api:
	uv run python -m scripts.run_api

batch:
	uv run python -m scripts.run_batch_forecast

monitor:
	uv run python -m scripts.run_monitoring

dashboard:
	uv run python -m scripts.run_dashboard

flow-features:
	uv run python -m scripts.run_flow --flow features

flow-training:
	uv run python -m scripts.run_flow --flow training

flow-batch:
	uv run python -m scripts.run_flow --flow batch --batch-limit 100

flow-monitor:
	uv run python -m scripts.run_flow --flow monitoring

flow-full:
	uv run python -m scripts.run_flow --flow full --batch-limit 100