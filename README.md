# Bike Demand Forecasting - Quickstart

This file is a cleaner, launch-oriented version of the project documentation.
The existing `README.md` is left unchanged.

## Overview

This project lets you:
- forecast bike demand at the station level
- expose forecasts through a FastAPI service
- visualize results in a Streamlit interface
- orchestrate dataset rebuilding and retraining workflows with Airflow

## Prerequisites

- Docker and Docker Compose
- or Python `>= 3.10` if you want to run the project locally without containers
- Capital Bikeshare 2024 and 2025 data files

## Useful project structure

- `streamlit_app/app.py`: Streamlit interface
- `api_service/main.py`: FastAPI service
- `airflow/dags/`: Airflow DAGs
- `scripts/`: dataset build, training, and prediction scripts
- `data/raw/2024` and `data/raw/2025`: source zip files
- `data/processed/`: intermediate outputs and predictions
- `models/`: saved model artifacts

## 1. Configuration

If you do not already have a `.env` file, copy the example:

```bash
cp .env.example .env
```

On Windows PowerShell:

```bash
Copy-Item .env.example .env
```

Important variables include:
- `API_HOST`, `API_PORT`
- `STREAMLIT_HOST`, `STREAMLIT_PORT`
- `AIRFLOW_WEBSERVER_HOST`, `AIRFLOW_WEBSERVER_PORT`
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

## 2. Add raw data

Place the zip archives, without extracting them, into:

```bash
data/raw/2024
data/raw/2025
```

## 3. Quick start with Docker

Before running Docker commands on Windows with WSL:
- make sure Docker Desktop is open
- make sure WSL integration is enabled for your Ubuntu distribution in Docker Desktop
- in practice, this is usually configured under `Settings > Resources > WSL Integration`

The easiest way to start the API and Streamlit is:

```bash
docker -f docker-compose.yml up -d --build
```

Then open:
- Streamlit: `http://localhost:8501`
- API: `http://localhost:8000`
- API healthcheck: `http://localhost:8000/health`

To follow logs:

```bash
docker compose logs -f
```

To stop services:

```bash
docker compose down
```

Important:
- starting the API and Streamlit containers does **not** create the dataset or train the model by itself
- on a fresh setup, you must first build the offline dataset and train the model before generating predictions
- you can do that either through Airflow and trigger dag or by running the scripts directly

## 4. Restart only Streamlit after a UI change

In the current setup, the Streamlit code is not mounted as a volume inside the container.
After changing files under `streamlit_app/`, you therefore need to rebuild the service:

```bash
docker compose up -d --build streamlit
```

## 5. Start the Airflow stack

The Airflow stack uses `docker-compose.airflow.yml`.

Start it with:

```bash
docker compose -f docker-compose.airflow.yml up -d --build
```

Open:
- Airflow API server / UI depending on your configuration: `http://localhost:8080`

Stop it with:

```bash
docker compose -f docker-compose.airflow.yml down
```

To follow Airflow logs:

```bash
docker compose -f docker-compose.airflow.yml logs -f
```

For a first-time setup, Airflow is useful because it can initialize the pipeline before you try to predict:
- build the offline dataset
- regenerate features
- retrain the final model

If no processed dataset or trained artifacts exist yet, run the retraining workflow once before using prediction endpoints or the prediction form in Streamlit.

## 6. Local Python installation

If you prefer to run the project without Docker:

### Create a virtual environment

Linux / macOS:

```bash
python3 -m venv env_bike
source env_bike/bin/activate
```

Windows PowerShell:

```bash
python -m venv env_bike
.\env_bike\Scripts\Activate.ps1
```

### Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### Install Airflow dependencies if needed

Docker remains the easiest option for Airflow. If you really want to install it locally, use the official constraints:

```bash
python -m pip install -r requirements-airflow.txt \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.8/constraints-3.12.txt"
```

## 7. Useful local commands

On a fresh project, the usual order is:

```bash
python scripts/build_offline_dataset.py
python scripts/train_offline_final_model.py --features-filename features_3segments.csv --artifact-prefix catboost_station_3segments_fullfit_v1 --n-jobs 4 --overwrite
python scripts/predict_recursive_days.py --features-filename features_3segments.csv --artifact-prefix catboost_station_3segments_fullfit_v1 --days 7 --start-date 2026-03-10T06:00:00
```

### Build the offline dataset

```bash
python scripts/build_offline_dataset.py
```

Useful options:

```bash
python scripts/build_offline_dataset.py --skip-extract --skip-merge
```

### Retrain the final model

```bash
python scripts/train_offline_final_model.py \
  --features-filename features_3segments.csv \
  --artifact-prefix catboost_station_3segments_fullfit_v1 \
  --n-jobs 4 \
  --overwrite
```

### Generate a recursive forecast

This step assumes that:
- `data/processed/features_3segments.csv` already exists
- trained model artifacts already exist in `models/`

```bash
python scripts/predict_recursive_days.py \
  --features-filename features_3segments.csv \
  --artifact-prefix catboost_station_3segments_fullfit_v1 \
  --days 7 \
  --start-date 2026-03-10T06:00:00
```

### Export a station-level view

```bash
python scripts/export_station_forecast.py \
  --input-filename predictions_recursive_days.csv \
  --station-id 30200
```

## 8. Run services without Docker

### FastAPI

```bash
uvicorn main:app --app-dir api_service --host 0.0.0.0 --port 8000
```

### Streamlit

```bash
python -m streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port 8501
```

## 9. What the application shows

In Streamlit, you can:
- generate a network forecast
- inspect station-level details
- run a rebalancing plan
- visualize the recommended route order

On a first run, predictions will only work if the dataset and model have already been initialized.

## 10. Useful notes

- the project uses a dual-model setup to differentiate high-volume and low-volume stations
- forecasting is performed at a 3-segment-per-day granularity
- rebalancing relies on operational assumptions and a greedy distance-based heuristic

## 11. Main documentation files

- `README.md`: historical project documentation
- `NOTE_METHODE.md`: additional methodology notes
