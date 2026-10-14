# 🛒 Production-Ready Demand Forecasting Platform

> An end-to-end ML engineering platform built on Walmart's M5 dataset — forecasting retail demand at the item-store level across the full ML lifecycle: ingestion, feature engineering, training, serving, monitoring, orchestration, and containerized deployment.

---

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [API Reference](#api-reference)
- [Monitoring & Observability](#monitoring--observability)
- [Orchestration](#orchestration)
- [Local Setup](#local-setup)
- [Docker Deployment](#docker-deployment)
- [Testing & Linting](#testing--linting)
- [Highlights](#highlights)
- [Future Improvements](#future-improvements)

---

## Overview

This project simulates a production-style retail demand forecasting platform. It predicts future item-store demand using historical sales, calendar context, price signals, and hierarchy-level aggregates — designed to demonstrate the complete ML engineering lifecycle from raw data to deployed, monitored inference.

**Platform capabilities at a glance:**

| Stage | Component |
|---|---|
| Ingestion | Raw CSV loading, schema validation |
| Features | Lag, rolling, calendar, price, hierarchy features |
| Training | Baseline benchmarking + LightGBM, tracked via MLflow |
| Serving | FastAPI (online) + batch inference pipeline |
| Monitoring | Drift detection (Evidently), RMSE/MAE/bias alerts, SQLite logging |
| Dashboard | Streamlit observability UI |
| Orchestration | Prefect flows for each pipeline stage |
| Deployment | Docker + docker-compose |

---

## Business Problem

Retail and e-commerce operations teams need accurate demand forecasts to support:

- **Inventory planning** — reduce overstock and carrying costs
- **Replenishment** — trigger restocking before stockouts occur
- **Stockout risk reduction** — protect revenue and customer satisfaction
- **Revenue planning** — support financial forecasting at the category and store level
- **Category / store operations** — inform assortment and space decisions

This platform forecasts product demand at the item-store level and exposes predictions through both API-based online inference and batch workflows.

---

## Dataset

This project uses the **Walmart M5 Forecasting dataset**, which includes:

| File | Description |
|---|---|
| `sales_train_validation.csv` | Daily item-level sales history |
| `calendar.csv` | Calendar metadata including events and SNAP flags |
| `sell_prices.csv` | Weekly item-store price history |

Raw M5 files are ingested into a canonical base table and then enriched with engineered forecasting features.

---

## Architecture

### Core Pipeline

```
Raw Data Ingestion
      │
      ▼
Feature Engineering  ───────────────────────────────────────────────┐
      │                                                             │
      ▼                                                             │
Model Training (LightGBM + baselines)                               │
      │                                                             │
      ├──► Online Serving (FastAPI)                                 │
      │                                                             │
      └──► Batch Inference ──► Monitoring & Drift Detection ────────┘
                                        │
                                        ▼
                              Streamlit Dashboard
```

### Stage Descriptions

**1. Raw Data Ingestion**
- Load `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv`
- Validate schema integrity and file completeness

**2. Feature Engineering**
- Calendar features (day-of-week, week-of-year, month, events, SNAP)
- Lag features (1 / 7 / 14 / 28 days)
- Rolling statistics (means, standard deviations, trend ratios)
- Price dynamics (lagged price, price change, relative price)
- Hierarchy-level aggregate features (store, category, department, state)

**3. Model Training**
- Time-based train/validation split
- Naive and rolling-mean baselines
- LightGBM regression model
- MLflow experiment tracking

**4. Serving**
- FastAPI endpoint for single item-store forecasting
- Batch forecast generation for multiple item-store pairs

**5. Monitoring**
- Data quality checks
- Forecast vs. actual evaluation
- Drift detection using Evidently
- SQLite-based prediction and metric logging
- Streamlit observability dashboard

**6. Orchestration**
- Prefect flows for feature build, training, batch inference, monitoring, and full pipeline execution

**7. Deployment**
- Dockerized API and dashboard services
- `docker-compose` for local production deployment

---

## Repository Structure

```
m5-demand-forecasting-platform/
├── app/
│   ├── api/                  # FastAPI app and route handlers
│   ├── config/               # Configuration and settings
│   ├── data/                 # Data ingestion and validation
│   ├── features/             # Feature engineering pipeline
│   ├── training/             # Model training and evaluation
│   ├── inference/            # Online and batch inference
│   ├── monitoring/           # Drift detection and quality checks
│   ├── orchestration/        # Prefect flows
│   └── utils/                # Shared utilities
├── dashboard/                # Streamlit observability dashboard
├── scripts/                  # CLI entry points
├── tests/                    # Unit and integration tests
├── configs/                  # YAML configs
├── docker/                   # Dockerfiles
├── data/                     # Raw and processed data
├── artifacts/                # Model artifacts and batch outputs
├── mlruns/                   # MLflow experiment logs
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Tech Stack

**Machine Learning**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![pandas](https://img.shields.io/badge/pandas-2.x-lightblue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange) ![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)

**MLOps & Monitoring**

![MLflow](https://img.shields.io/badge/MLflow-2.x-red) ![Evidently](https://img.shields.io/badge/Evidently-0.4.x-purple) ![Prefect](https://img.shields.io/badge/Prefect-3.x-teal) ![SQLite](https://img.shields.io/badge/SQLite-logging-lightgrey)

**Serving & Application Layer**

![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688) ![Pydantic](https://img.shields.io/badge/Pydantic-2.x-e92063) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b)

**Deployment & Developer Tooling**

![Docker](https://img.shields.io/badge/Docker-Compose-2496ED) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI-2088FF) ![uv](https://img.shields.io/badge/uv-package_manager-5C4EE5) ![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC) ![Ruff](https://img.shields.io/badge/Ruff-linting-FCC21B) ![Black](https://img.shields.io/badge/Black-formatting-black)

---

## Feature Engineering

The training pipeline builds a forecasting feature table with **9.45M rows** across the following feature groups:

### Calendar Features
| Feature | Description |
|---|---|
| `day_of_week` | Day of week (0–6) |
| `week_of_year` | ISO week number |
| `month` / `quarter` | Month and quarter indicators |
| `is_weekend` | Weekend flag |
| `event_indicators` | National/cultural event flags |
| `snap_*` | SNAP benefit day flags by state |

### Sales History Features
| Feature | Description |
|---|---|
| `lag_1`, `lag_7`, `lag_14`, `lag_28` | Lagged unit sales |
| `rolling_mean_*` | Rolling average unit sales (various windows) |
| `rolling_std_*` | Rolling standard deviation |
| `trend_ratio` | Ratio of recent to longer-term rolling mean |

### Price Features
| Feature | Description |
|---|---|
| `sell_price` | Current week price |
| `price_lag_1` | Previous period price |
| `price_change` | Week-over-week price delta |
| `rolling_price_mean` | Rolling average price |
| `price_vs_hist_mean` | Price relative to item-store historical mean |

### Hierarchy Features
| Feature | Description |
|---|---|
| `store_rolling_demand` | Rolling demand aggregated at store level |
| `category_rolling_demand` | Rolling demand at category level |
| `dept_rolling_demand` | Rolling demand at department level |
| `state_rolling_demand` | Rolling demand at state level |

---

## Model Training

### Models Trained

| Model | Type | Description |
|---|---|---|
| Naive lag baseline | Heuristic | Last observed value |
| Rolling mean baseline | Heuristic | Recent rolling average |
| **LightGBM** | Gradient boosting | Full feature set regression model |

### Validation Strategy

- **Split type:** Temporal train/validation split
- **Forecast horizon:** 28 days
- **Tracking:** MLflow (parameters, metrics, artifacts)

### Validation Performance

| Model | RMSE |
|---|---|
| Best baseline | 2.3055 |
| **LightGBM** | **2.1540** |
| Improvement | **~6.6% over strongest baseline** |

---

## API Reference

The FastAPI server exposes the following endpoints:

### `GET /health`
Returns service health status.

### `GET /model-info`
Returns metadata about the currently loaded model.

### `POST /forecast/item-store`
Generate a 28-day demand forecast for a single item-store pair.

**Request:**
```json
{
  "item_id": "FOODS_1_001",
  "store_id": "CA_1"
}
```

**Response:**
```json
{
  "item_id": "FOODS_1_001",
  "store_id": "CA_1",
  "feature_row_date": "2016-03-27",
  "target_date": "2016-04-24",
  "forecast_horizon_days": 28,
  "predicted_sales": 1.0530,
  "model_name": "lightgbm_model"
}
```

### `POST /forecast/batch`
Generate forecasts for multiple item-store pairs.

**Request:**
```json
{
  "limit": 25
}
```

Interactive API docs available at `http://localhost:8001/docs` when running locally.

---

## Monitoring & Observability

### Monitoring Pipeline

The monitoring pipeline tracks:

- **Data quality** — column completeness, value distributions, schema checks
- **Forecast performance** — RMSE, MAE, and bias against actuals
- **Drift detection** — Evidently-powered data drift reports
- **Alerting** — configurable thresholds for RMSE, MAE, and forecast bias

### Generated Artifacts

| Artifact | Format | Description |
|---|---|---|
| Monitoring logs | SQLite | Per-run metrics and API forecast history |
| Batch predictions | Parquet | Batch inference outputs |
| Drift report | HTML | Evidently data drift report |

### Streamlit Dashboard

The Streamlit dashboard (`http://localhost:8501`) provides:

- Model overview and metadata
- Recent batch forecast run summaries
- Recent monitoring results and metric trends
- Recent API forecast logs
- Latest batch prediction preview
- Drift report status

---

## Orchestration

The project uses **Prefect** to orchestrate all pipeline stages as flows:

| Flow | Description |
|---|---|
| `feature_build` | Rebuild the full feature table |
| `training` | Retrain the LightGBM model |
| `batch_inference` | Run batch forecasts |
| `monitoring` | Run monitoring and drift checks |
| `full_pipeline` | Execute all stages end-to-end |

**Run any flow via:**
```bash
uv run python -m scripts.run_flow --flow monitoring
```

---

## Local Setup

### 1. Create environment

```bash
uv init
uv sync
```

### 2. Run ingestion

```bash
uv run python -m scripts.run_ingestion
```

### 3. Build features

```bash
uv run python -m scripts.run_feature_build
```

### 4. Train model

```bash
uv run python -m scripts.run_training
```

### 5. Start the API

```bash
uv run python -m scripts.run_api
```

### 6. Run monitoring

```bash
uv run python -m scripts.run_monitoring
```

### 7. Start the dashboard

```bash
uv run python -m scripts.run_dashboard
```

---

## Docker Deployment

Build and start both the API and dashboard services:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API docs | http://localhost:8001/docs |
| Dashboard | http://localhost:8501 |

---

## Testing & Linting

```bash
# Run tests
uv run pytest -q

# Lint
uv run ruff check app scripts tests dashboard

# Format check
uv run black --check app scripts tests dashboard
```

---

## Highlights

- 📦 Engineered a forecasting feature table with **9.45M rows** across lag, rolling, price, and hierarchy dimensions
- 🎯 LightGBM model achieves **~6.6% RMSE improvement** over the strongest naive baseline
- ⚡ Dual inference paths: **online FastAPI** serving and **batch Parquet pipeline**
- 🔍 **SQLite-based logging**, Evidently drift detection, and configurable alert thresholds
- 📊 **Streamlit observability dashboard** with forecast previews and drift report status
- 🔁 **Prefect orchestration** for all pipeline stages with a single full-pipeline flow
- 🐳 Fully **Dockerized** with `docker-compose` for local production deployment

---

## Future Improvements

- [ ] Recursive multi-step forecasting for longer horizons
- [ ] Richer alerting and smarter threshold tuning
- [ ] Scoped data quality checks targeting only critical columns
- [ ] External object storage (S3/GCS) for model and data artifacts
- [ ] Cloud deployment of API and dashboard (e.g., Cloud Run, ECS)

---

<p align="center">
  Built with 🧡 using the <a href="https://www.kaggle.com/competitions/m5-forecasting-accuracy">Walmart M5 Forecasting Dataset</a>
</p>