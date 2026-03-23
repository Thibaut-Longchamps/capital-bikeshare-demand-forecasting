from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from bike_demand_forecasting.utils import (
    format_timestamp_for_filename,
    get_paths,
)
from bike_demand_forecasting.rebalancing import compute_rebalancing_plan
from bike_demand_forecasting.recursive_forecast import main as predict_recursive_main
from bike_demand_forecasting.station_export import main as export_station_main

PATHS = get_paths(PROJECT_ROOT)
PREDICTIONS_SUBDIR = "predictions"


app = FastAPI(
    title="Bike Demand Forecasting API",
    version="0.1.0",
    description="API for recursive network forecast and station-level export.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecursivePredictRequest(BaseModel):
    features_filename: str = "features_3segments.csv"
    artifact_prefix: str = "catboost_station_3segments_fullfit_v1"
    days: int = Field(default=7, ge=1, le=31)
    start_date: datetime


class StationExportRequest(BaseModel):
    input_filename: str
    station_id: int = Field(ge=0)


class RebalancingPlanRequest(BaseModel):
    forecast_filename: str
    capacity_filename: str = "station_capacity.csv"
    realtime_filename: str = "realtime/station_status_realtime.csv"
    horizon_segments: int = Field(default=3, ge=1, le=12)
    net_out_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    max_transfers: int | None = Field(default=20, ge=1)
    save_outputs: bool = True


def _timestamp() -> str:
    return format_timestamp_for_filename()


def _prediction_filename() -> str:
    return f"{PREDICTIONS_SUBDIR}/predictions_recursive_api_{_timestamp()}.csv"


def _processed_path(filename: str) -> Path:
    return PATHS["DATA_DIR"] / "processed" / filename


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "processed_dir": str(PATHS["DATA_DIR"] / "processed"),
        "models_dir": str(PATHS["WORK_DIR"] / "models"),
    }


@app.post("/predict/recursive")
def predict_recursive(payload: RecursivePredictRequest) -> dict:
    output_filename = _prediction_filename()

    try:
        predict_recursive_main(
            features_filename=payload.features_filename,
            artifact_prefix=payload.artifact_prefix,
            days=payload.days,
            output_filename=output_filename,
            start_date=payload.start_date.isoformat(),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    response = {
        "status": "success",
        "output_filename": output_filename,
        "output_path": str(_processed_path(output_filename)),
    }
    return response


@app.post("/station/export")
def export_station(payload: StationExportRequest) -> dict:
    input_path = Path(payload.input_filename)
    output_filename = str(input_path.parent / f"{input_path.stem}_station_{payload.station_id}.csv")

    try:
        export_station_main(
            input_filename=payload.input_filename,
            station_id=payload.station_id,
            output_filename=output_filename,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    return {
        "status": "success",
        "input_filename": payload.input_filename,
        "station_id": payload.station_id,
        "output_filename": output_filename,
        "output_path": str(_processed_path(output_filename)),
    }


@app.post("/ops/rebalancing/plan")
def rebalancing_plan(payload: RebalancingPlanRequest) -> dict:
    try:
        result = compute_rebalancing_plan(
            paths=PATHS,
            forecast_filename=payload.forecast_filename,
            capacity_filename=payload.capacity_filename,
            realtime_filename=payload.realtime_filename,
            horizon_segments=payload.horizon_segments,
            net_out_ratio=payload.net_out_ratio,
            max_transfers=payload.max_transfers,
            save_outputs=payload.save_outputs,
            output_prefix=None,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    return {
        "status": "success",
        "summary": result["summary"],
        "generated_inputs": result["generated_inputs"],
        "files": result["files"],
        "alerts_rows": result["alerts_df"].to_dict(orient="records"),
        "transfer_rows": result["transfer_df"].to_dict(orient="records"),
        "route_rows": result["route_df"].to_dict(orient="records"),
    }
