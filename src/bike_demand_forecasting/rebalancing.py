from __future__ import annotations

"""
Rebalancing engine for short-horizon bike operations planning.

The module computes shortage/surplus alerts from forecast + realtime stock,
then proposes inter-station transfers and an ordered route plan.
"""

from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd

from bike_demand_forecasting.utils import (
    to_naive_timestamp,
    format_timestamp_for_filename,
)

# Canonical transfer table schema used across generation/aggregation/export.
TRANSFER_COLUMNS = [
    "from_station_id",
    "to_station_id",
    "qty_bikes",
    "distance_km",
    "priority",
    "from_lat",
    "from_lng",
    "to_lat",
    "to_lng",
]
SEVERITY_RANK = {"critical": 0, "warning": 1, "ok": 2}
REBALANCING_OUTPUT_SUBDIR = "rebalancing"
ALERTS_OUTPUT_SUBDIR = "alerts"
TRANSFER_OUTPUT_SUBDIR = "transfer_plan"
ROUTE_OUTPUT_SUBDIR = "route_plan"
REALTIME_OUTPUT_SUBDIR = "realtime"
DEFAULT_REALTIME_FILENAME = f"{REALTIME_OUTPUT_SUBDIR}/station_status_realtime.csv"


# Path/schema validation helpers.
def _processed_file(paths: dict[str, Path], filename: str) -> Path:
    """Return an absolute path inside `data/processed`."""
    return paths["DATA_DIR"] / "processed" / filename


def _validate_required_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    """Raise if required columns are missing from a loaded input table."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} file missing required columns: {missing}")


def _missing_station_ids(df: pd.DataFrame, expected_ids: set[int]) -> list[int]:
    """Return station IDs expected by forecast but absent in `df`."""
    return sorted(expected_ids - set(df["start_station_id"].tolist()))


def _empty_transfer_df() -> pd.DataFrame:
    """Create an empty transfer plan with the canonical column schema."""
    return pd.DataFrame(columns=TRANSFER_COLUMNS)


def _safe_sum(df: pd.DataFrame, col: str) -> float:
    """Safe numeric sum helper (returns 0.0 when column/data is absent)."""
    return float(df[col].sum()) if col in df.columns and not df.empty else 0.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two (lat, lon) points."""
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# Input loaders and normalization.
def _load_forecast(paths: dict[str, Path], forecast_filename: str) -> pd.DataFrame:
    """
    Load and normalize a forecast file used by rebalancing.

    Required columns: start_station_id, date, segment_id, y_pred.
    """
    # Resolve and validate forecast file path.
    forecast_path = _processed_file(paths, forecast_filename)
    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast file not found: {forecast_path}")

    # Read and validate schema.
    df = pd.read_csv(forecast_path)
    _validate_required_columns(df, {"start_station_id", "date", "segment_id", "y_pred"}, "Forecast")

    # Normalize types and drop invalid rows.
    numeric_cols = ["start_station_id", "segment_id", "y_pred"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["start_station_id", "date", "segment_id", "y_pred"]).copy()

    df["start_station_id"] = df["start_station_id"].astype(int)
    df["segment_id"] = df["segment_id"].astype(int)
    df["date"] = df["date"].map(to_naive_timestamp)
    return df.sort_values(["date", "start_station_id"]).reset_index(drop=True)


def _load_capacity(
    paths: dict[str, Path],
    df_forecast: pd.DataFrame,
    capacity_filename: str,
) -> pd.DataFrame:
    """
    Load and validate station capacity settings.
    """
    capacity_path = _processed_file(paths, capacity_filename)
    if not capacity_path.exists():
        raise FileNotFoundError(
            f"Capacity file not found: {capacity_path}. "
            "Provide a valid station_capacity.csv (auto-generation is disabled)."
        )
    # Read and validate required columns (strict schema for a stable static file).
    df = pd.read_csv(capacity_path)
    required_cols = {
        "start_station_id",
        "capacity_bikes",
        "min_buffer",
        "target_fill_ratio",
        "max_fill_ratio",
    }
    _validate_required_columns(df, required_cols, "Capacity")

    # Normalize and enforce operational bounds.
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(required_cols)).copy()

    df["start_station_id"] = df["start_station_id"].astype(int)
    df["capacity_bikes"] = df["capacity_bikes"].round().astype(int).clip(lower=1, upper=200)
    df["min_buffer"] = df["min_buffer"].round().astype(int).clip(lower=0)
    df["target_fill_ratio"] = df["target_fill_ratio"].clip(0.1, 0.95)
    df["max_fill_ratio"] = df["max_fill_ratio"].clip(0.2, 0.99)
    df["min_buffer"] = np.minimum(df["min_buffer"], df["capacity_bikes"]).astype(int)

    # Keep one row per station (last occurrence wins).
    df = df.drop_duplicates(subset=["start_station_id"], keep="last").reset_index(drop=True)

    # Ensure all forecast stations are covered.
    forecast_ids = set(df_forecast["start_station_id"].astype(int).unique().tolist())
    missing_ids = _missing_station_ids(df, forecast_ids)
    if missing_ids:
        raise ValueError(f"Capacity file missing stations: {missing_ids[:10]} ...")

    return df


def _load_realtime(
    paths: dict[str, Path],
    capacity_df: pd.DataFrame,
    realtime_filename: str,
) -> pd.DataFrame:
    """
    Load realtime station stock from a static processed realtime file and merge with station capacity
    """
    realtime_path = _processed_file(paths, realtime_filename)
    if not realtime_path.exists():
        raise FileNotFoundError(
            f"Realtime file not found: {realtime_path}. "
            f"Expected a classic realtime CSV in data/processed/{DEFAULT_REALTIME_FILENAME}."
        )

    df = pd.read_csv(realtime_path)
    required_cols = {"start_station_id", "bikes_available", "docks_available"}
    _validate_required_columns(df, required_cols, "Realtime")
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(required_cols)).copy()

    df["start_station_id"] = df["start_station_id"].astype(int)
    df["bikes_available"] = df["bikes_available"].round().astype(int).clip(lower=0)
    df["docks_available"] = df["docks_available"].round().astype(int).clip(lower=0)
    all_ids = set(capacity_df["start_station_id"].astype(int).tolist())
    missing_ids = _missing_station_ids(df, all_ids)
    if missing_ids:
        raise ValueError(f"Realtime file missing stations: {missing_ids[:10]} ...")

    df = df.drop_duplicates(subset=["start_station_id"], keep="last").reset_index(drop=True)
    merged = df.merge(
        capacity_df[["start_station_id", "capacity_bikes"]],
        on="start_station_id",
        how="left",
    )
    # Enforce physical limits and recompute docks as residual capacity.
    merged["bikes_available"] = np.minimum(
        merged["bikes_available"], merged["capacity_bikes"]
    ).astype(int)
    merged["docks_available"] = (merged["capacity_bikes"] - merged["bikes_available"]).astype(int)
    out = merged[["start_station_id", "bikes_available", "docks_available"]].copy()

    return out


def _load_station_coordinates(paths: dict[str, Path]) -> pd.DataFrame:
    """
    Load station coordinates from the processed static coordinates file.

    Returned columns: start_station_id, start_lat, start_lng.
    """
    coords_path = _processed_file(paths, "station_coordinates.csv")
    if not coords_path.exists():
        raise FileNotFoundError(f"Station coordinates file not found: {coords_path}")

    df = pd.read_csv(coords_path)
    _validate_required_columns(df, {"start_station_id", "start_lat", "start_lng"}, "Station coordinates")
    df = df.dropna(subset=["start_station_id", "start_lat", "start_lng"]).copy()
    df["start_station_id"] = pd.to_numeric(df["start_station_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["start_station_id"]).copy()
    df["start_station_id"] = df["start_station_id"].astype(int)
    return df[["start_station_id", "start_lat", "start_lng"]]


def _severity_from_row(row: pd.Series) -> str:
    """Classify alert severity from projected stock and deficit magnitude."""
    if row["deficit_qty"] <= 0:
        return "ok"
    if row["stock_proj"] < 0 or row["deficit_qty"] >= max(5, round(0.30 * row["capacity_bikes"])):
        return "critical"
    return "warning"


# Transfer and route construction helpers.
def _build_transfers(df_state: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """
    Build an inter-station bike transfer plan from deficits/surpluses.

    Stations with transferable stock become donors; deficit stations become receivers.
    Transfers are prioritized by severity, then by distance where coordinates are available.
    """
    # Split state into donor and receiver candidate tables.
    donors = (
        df_state.loc[df_state["transfer_available_qty"] > 0, ["start_station_id", "transfer_available_qty", "severity"]]
        .rename(columns={"start_station_id": "from_station_id", "transfer_available_qty": "available_qty"})
        .copy()
    )
    receivers = (
        df_state.loc[df_state["deficit_qty"] > 0, ["start_station_id", "deficit_qty", "severity"]]
        .rename(columns={"start_station_id": "to_station_id", "deficit_qty": "needed_qty"})
        .copy()
    )
    if donors.empty or receivers.empty:
        return _empty_transfer_df()

    # Build fast coordinate lookup for distance-aware matching.
    coord_map = {}
    if not coords.empty:
        coords_idx = coords.drop_duplicates("start_station_id").set_index("start_station_id")
        coord_map = coords_idx.to_dict(orient="index")

    donor_left = {int(r.from_station_id): int(r.available_qty) for r in donors.itertuples(index=False)}
    transfer_rows: list[dict[str, Any]] = []

    # Critical receivers first.
    receivers = receivers.sort_values(
        by=["severity", "needed_qty"],
        ascending=[True, False],
        key=lambda col: col.map(SEVERITY_RANK) if col.name == "severity" else col,
    )

    for recv in receivers.itertuples(index=False):
        to_sid = int(recv.to_station_id)
        need_left = int(recv.needed_qty)
        while need_left > 0:
            candidates = [sid for sid, qty in donor_left.items() if qty > 0 and sid != to_sid]
            if not candidates:
                break

            # Choose nearest donor when coordinates exist, otherwise largest donor stock.
            if to_sid in coord_map:
                to_lat = float(coord_map[to_sid]["start_lat"])
                to_lng = float(coord_map[to_sid]["start_lng"])

                def donor_score(sid: int) -> tuple[float, int]:
                    if sid in coord_map:
                        d = _haversine_km(
                            float(coord_map[sid]["start_lat"]),
                            float(coord_map[sid]["start_lng"]),
                            to_lat,
                            to_lng,
                        )
                    else:
                        d = 1e9
                    return (d, -donor_left[sid])

                chosen = min(candidates, key=donor_score)
            else:
                # Without coordinates, prioritize the donor that can send most bikes.
                chosen = max(candidates, key=lambda sid: donor_left[sid])

            qty = min(need_left, donor_left[chosen])
            donor_left[chosen] -= qty
            need_left -= qty

            # Attach optional geometry and distance for route optimization.
            from_coords = coord_map.get(chosen, {})
            to_coords = coord_map.get(to_sid, {})
            if from_coords and to_coords:
                dist_km = _haversine_km(
                    float(from_coords["start_lat"]),
                    float(from_coords["start_lng"]),
                    float(to_coords["start_lat"]),
                    float(to_coords["start_lng"]),
                )
            else:
                dist_km = np.nan

            transfer_rows.append(
                {
                    "from_station_id": chosen,
                    "to_station_id": to_sid,
                    "qty_bikes": int(qty),
                    "distance_km": float(dist_km) if pd.notna(dist_km) else np.nan,
                    "priority": str(recv.severity),
                    "from_lat": from_coords.get("start_lat", np.nan),
                    "from_lng": from_coords.get("start_lng", np.nan),
                    "to_lat": to_coords.get("start_lat", np.nan),
                    "to_lng": to_coords.get("start_lng", np.nan),
                }
            )

    if not transfer_rows:
        return _empty_transfer_df()

    # Merge duplicate source-target pairs and keep the best (minimum) distance.
    out = pd.DataFrame(transfer_rows)
    out = (
        out.groupby(
            [
                "from_station_id",
                "to_station_id",
                "priority",
                "from_lat",
                "from_lng",
                "to_lat",
                "to_lng",
            ],
            as_index=False,
        )
        .agg(
            qty_bikes=("qty_bikes", "sum"),
            distance_km=("distance_km", "min"),
        )
    )
    out = out.sort_values(
        by=["priority", "distance_km", "qty_bikes"],
        ascending=[True, True, False],
        key=lambda col: col.map(SEVERITY_RANK) if col.name == "priority" else col,
    ).reset_index(drop=True)
    return out


def _build_route_plan(transfers: pd.DataFrame) -> pd.DataFrame:
    """Assign ordered route steps from the selected transfer list."""
    route = transfers.copy().reset_index(drop=True)
    route["step"] = np.arange(1, len(route) + 1, dtype=int)
    cols = ["step", "from_station_id", "to_station_id", "qty_bikes", "distance_km", "priority"]
    return route[cols]


# Public API used by scripts, FastAPI and Streamlit.
def compute_rebalancing_plan(
    paths: dict[str, Path],
    forecast_filename: str,
    capacity_filename: str = "station_capacity.csv",
    realtime_filename: str = DEFAULT_REALTIME_FILENAME,
    horizon_segments: int = 3,
    net_out_ratio: float = 0.35,
    max_transfers: int | None = 20,
    save_outputs: bool = True,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    """
    Compute station alerts and bike transfer recommendations for a short forecast horizon.

    Workflow:
    1) load forecast + capacity + realtime
    2) project stock after expected net outflow
    3) derive deficits/surpluses and severities
    4) build transfer and route plans
    5) optionally persist CSV outputs
    """
    # Sanity check arguments (horizon_segment, net_out_ratio, max_transfer)
    if horizon_segments <= 0:
        raise ValueError("horizon_segments must be >= 1")
    if not (0 <= net_out_ratio <= 1):
        raise ValueError("net_out_ratio must be between 0 and 1.")
    if max_transfers is not None and max_transfers <= 0:
        raise ValueError("max_transfers must be >= 1 when provided.")

    # Load prediction csv
    df_forecast = _load_forecast(paths, forecast_filename)
    unique_slots = sorted(df_forecast["date"].dropna().unique().tolist())
    
    if len(unique_slots) < horizon_segments:
        raise ValueError(
            f"Forecast has only {len(unique_slots)} time slots, need at least {horizon_segments}."
        )
    # Business rule: operate on the first chronological horizon slots.
    horizon_slots = unique_slots[:horizon_segments]
    df_horizon = df_forecast[df_forecast["date"].isin(horizon_slots)].copy()
    # Total forecasted demand over the considered horizon, per station.
    pred_out_horizon = (
        df_horizon.groupby("start_station_id", as_index=True)["y_pred"].sum()
        .rename("pred_out_horizon")
    )

    capacity_df = _load_capacity(
        paths=paths,
        df_forecast=df_forecast,
        capacity_filename=capacity_filename,
    )
    realtime_df = _load_realtime(
        paths=paths,
        capacity_df=capacity_df,
        realtime_filename=realtime_filename,
    )

    state = capacity_df.merge(
        realtime_df[["start_station_id", "bikes_available", "docks_available"]],
        on="start_station_id",
        how="left",
    )
    # Core projection model for operational stock risk.
    state["pred_out_horizon"] = state["start_station_id"].map(pred_out_horizon).fillna(0.0)
    state["pred_net_out_horizon"] = state["pred_out_horizon"] * float(net_out_ratio)
    state["target_stock"] = (state["capacity_bikes"] * state["target_fill_ratio"]).round()
    # Use net outflow estimate (outflow minus expected returns) to avoid over-triggering shortages.
    state["stock_proj"] = state["bikes_available"] - state["pred_net_out_horizon"]
    state["deficit_qty"] = np.maximum(np.ceil(state["min_buffer"] - state["stock_proj"]), 0).astype(int)
    # Operational alert surplus (above target stock).
    state["surplus_qty"] = np.maximum(np.floor(state["stock_proj"] - state["target_stock"]), 0).astype(int)
    # Transfer availability (above safety buffer), used to source bikes for shortage stations.
    state["transfer_available_qty"] = np.maximum(
        np.floor(state["stock_proj"] - state["min_buffer"]), 0
    ).astype(int)
    state["severity"] = state.apply(_severity_from_row, axis=1)

    coords_df = _load_station_coordinates(paths)
    state = state.merge(coords_df, on="start_station_id", how="left")

    alerts_df = state.loc[
        (state["deficit_qty"] > 0) | (state["surplus_qty"] > 0),
        [
            "start_station_id",
            "capacity_bikes",
            "min_buffer",
            "bikes_available",
            "pred_out_horizon",
            "pred_net_out_horizon",
            "stock_proj",
            "deficit_qty",
            "surplus_qty",
            "transfer_available_qty",
            "severity",
            "start_lat",
            "start_lng",
        ],
    ].sort_values(["severity", "deficit_qty", "surplus_qty"], ascending=[True, False, False])

    transfer_df_full = _build_transfers(state, coords_df)
    # Optional operational cap: keep only top-N recommended transfers.
    if max_transfers is None:
        transfer_df = transfer_df_full.copy()
    else:
        transfer_df = transfer_df_full.head(int(max_transfers)).copy()
    route_df = _build_route_plan(transfer_df)

    files_out: dict[str, str | None] = {
        "forecast_file": str(_processed_file(paths, forecast_filename)),
        "capacity_file": str(_processed_file(paths, capacity_filename)),
        "realtime_file": str(_processed_file(paths, realtime_filename)),
        "alerts_file": None,
        "transfer_plan_file": None,
        "route_plan_file": None,
    }

    if save_outputs:
        # Use one local timestamp per run and store each artifact in its dedicated folder.
        stamp = format_timestamp_for_filename()
        base_prefix = output_prefix or f"rebalancing_{Path(forecast_filename).stem}"
        run_name = f"{base_prefix}_{stamp}"
        rebalancing_root = paths["DATA_DIR"] / "processed" / REBALANCING_OUTPUT_SUBDIR
        run_dir = rebalancing_root / run_name

        alerts_dir = run_dir / ALERTS_OUTPUT_SUBDIR
        transfer_dir = run_dir / TRANSFER_OUTPUT_SUBDIR
        route_dir = run_dir / ROUTE_OUTPUT_SUBDIR
        alerts_dir.mkdir(parents=True, exist_ok=True)
        transfer_dir.mkdir(parents=True, exist_ok=True)
        route_dir.mkdir(parents=True, exist_ok=True)

        alerts_path = alerts_dir / f"alerts_{stamp}.csv"
        transfer_path = transfer_dir / f"transfer_plan_{stamp}.csv"
        route_path = route_dir / f"route_plan_{stamp}.csv"
        alerts_df.to_csv(alerts_path, index=False)
        transfer_df.to_csv(transfer_path, index=False)
        route_df.to_csv(route_path, index=False)
        files_out.update(
            {
                "alerts_file": str(alerts_path),
                "transfer_plan_file": str(transfer_path),
                "route_plan_file": str(route_path),
            }
        )

    # Summary metrics for API/UI dashboards.
    total_move = int(_safe_sum(transfer_df, "qty_bikes"))
    total_distance = _safe_sum(transfer_df, "distance_km")
    total_move_raw = int(_safe_sum(transfer_df_full, "qty_bikes"))
    total_distance_raw = _safe_sum(transfer_df_full, "distance_km")
    critical_cnt = int((alerts_df["severity"] == "critical").sum()) if not alerts_df.empty else 0
    warning_cnt = int((alerts_df["severity"] == "warning").sum()) if not alerts_df.empty else 0

    return {
        "summary": {
            "stations_total": int(state["start_station_id"].nunique()),
            "alerts_total": int(len(alerts_df)),
            "alerts_critical": critical_cnt,
            "alerts_warning": warning_cnt,
            "transfers_total_raw": int(len(transfer_df_full)),
            "transfers_total": int(len(transfer_df)),
            "max_transfers": int(max_transfers) if max_transfers is not None else None,
            "bikes_to_move_total_raw": total_move_raw,
            "bikes_to_move_total": total_move,
            "distance_total_km_raw": round(total_distance_raw, 3),
            "distance_total_km": round(total_distance, 3),
            "horizon_segments": int(horizon_segments),
            "net_out_ratio": float(net_out_ratio),
            "horizon_start": str(min(horizon_slots)),
            "horizon_end": str(max(horizon_slots)),
        },
        "generated_inputs": {
            "realtime_source": "file",
            "realtime_file": str(_processed_file(paths, realtime_filename)),
        },
        "files": files_out,
        "alerts_df": alerts_df.reset_index(drop=True),
        "transfer_df": transfer_df.reset_index(drop=True),
        "route_df": route_df.reset_index(drop=True),
    }
