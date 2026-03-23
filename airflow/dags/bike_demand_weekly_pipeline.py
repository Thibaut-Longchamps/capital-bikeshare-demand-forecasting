from __future__ import annotations

import os
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = os.environ.get("BIKE_PYTHON_BIN", "python3")
PYTHONPATH_SRC = PROJECT_ROOT / "src"


default_args = {
    "owner": "bike-demand-forecasting",
    "depends_on_past": False,
    "retries": 2,
}


def env_prefix() -> str:
    return (
        f'cd "{PROJECT_ROOT}" '
        f'&& export PYTHONPATH="{PYTHONPATH_SRC}:${{PYTHONPATH:-}}"'
    )
    # cd "/mnt/c/.../bike-demand-forecasting" && export PYTHONPATH="/mnt/c/.../bike-demand-forecasting/src:${PYTHONPATH:-}"



# Weekly retraining DAG: rebuild the offline dataset, then retrain the production model.
with DAG(
    dag_id="bike_demand_weekly_retrain",
    description="Weekly model retraining on refreshed offline dataset.",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    schedule="0 2 * * 1",  # Every Monday at 02:00 Europe/Paris
    catchup=False,
    max_active_runs=1,
    tags=["bike", "train", "weekly"],
    params={
        "skip_extract": Param(False, type="boolean"),
        "skip_merge": Param(False, type="boolean"),
        "n_jobs": Param(4, type="integer", minimum=1, maximum=16),
        "artifact_prefix": Param("catboost_station_3segments_fullfit_v1", type="string"),
    },
) as dag_weekly_retrain:
    build_offline_dataset_weekly = BashOperator(
        task_id="build_offline_dataset",
        bash_command=(
            f"{env_prefix()} && {PYTHON_BIN} scripts/build_offline_dataset.py"
            "{% if params.skip_extract %} --skip-extract{% endif %}"
            "{% if params.skip_merge %} --skip-merge{% endif %}"
        ),
    )

    train_final_model = BashOperator(
        task_id="train_final_model",
        bash_command=(
            f"{env_prefix()} && {PYTHON_BIN} scripts/train_offline_final_model.py "
            "--features-filename features_3segments.csv "
            "--n-jobs {{ params.n_jobs }} "
            '--artifact-prefix "{{ params.artifact_prefix }}" '
            "--overwrite"
        ),
    )

    build_offline_dataset_weekly >> train_final_model
    
## INSTALL
# source env_bike/bin/activate
# python -m pip install "apache-airflow==2.10.5"


# python -m pip install apache-airflow-providers-fab \
#   --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.8/constraints-3.12.txt"
