from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from bike_demand_forecasting.inference import load_models_and_meta, predict_dual
from bike_demand_forecasting.metrics import compute_metrics
from bike_demand_forecasting.training import load_feature_table
from bike_demand_forecasting.utils import (
    align_to_segment_start,
    get_paths,
    next_segment_start,
    to_naive_timestamp,
)


def parse_lags_and_windows(feature_cols: list[str]) -> tuple[list[int], list[int]]:
    # Recover recursive requirements directly from the trained model interface.
    lags = sorted(
        {int(c.split("_")[1]) for c in feature_cols if c.startswith("lag_")}
    )
    windows = sorted(
        {
            int(c.split("_")[2])
            for c in feature_cols
            if c.startswith("roll_mean_") or c.startswith("roll_std_")
        }
    )
    return lags, windows


def main(
    features_filename: str,
    artifact_prefix: str,
    days: int,
    output_filename: str,
    start_date: str | None,
    station_id: int | None = None,
    station_output_filename: str | None = None,
    save_station_output: bool = True,
) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    paths = get_paths(project_root)

    if days <= 0:
        raise ValueError("--days must be >= 1")

    # 1) Load the trained model bundle and infer how much past history is needed.
    # `feature_cols` defines the exact prediction interface expected at inference time.
    bundle = load_models_and_meta(artifact_prefix=artifact_prefix)
    feature_cols = bundle["feature_cols_cb"]
    lags, windows = parse_lags_and_windows(feature_cols)
    max_needed = max(lags + windows) if (lags or windows) else 1

    # 2) Load historical features and keep a stable chronological order per station.
    # This table is both the seed for recursion and the optional source of evaluation targets.
    df_feat = load_feature_table(paths, features_filename)
    if "y_station" not in df_feat.columns:
        raise ValueError("Input features must include y_station for recursive forecasting.")
    df_feat = df_feat.sort_values(["start_station_id", "date"]).reset_index(drop=True)
    station_ids = sorted(df_feat["start_station_id"].astype(int).unique().tolist())

    # 3) Resolve the forecast start, then expand it into the sequence of future slots to predict.
    # When the user gives a custom start date, it is aligned to the next valid segment boundary.
    if start_date:
        start_ts = align_to_segment_start(pd.to_datetime(start_date))
    else:
        start_ts = next_segment_start(to_naive_timestamp(df_feat["date"].max()))

    # Guardrail: do not allow large jumps far beyond the last observed timestamp.
    last_hist_ts = to_naive_timestamp(df_feat["date"].max())
    max_allowed_start_ts = last_hist_ts + pd.Timedelta(days=7)
    if start_ts > max_allowed_start_ts:
        raise ValueError(
            "start_date is too far from available history. "
            f"Last feature timestamp={last_hist_ts}, "
            f"max allowed start={max_allowed_start_ts} (7 days), "
            f"received aligned start={start_ts}."
        )

    steps = days * 3  # 3 segments/day
    slots: list[pd.Timestamp] = []
    ts = start_ts
    for _ in range(steps):
        slots.append(ts)
        ts = next_segment_start(ts)

    # 4) Precompute holiday dates once for the full forecast window.
    # Each slot later converts this into a simple binary `is_holiday` feature.
    cal = USFederalHolidayCalendar()
    holiday_days = set(
        pd.DatetimeIndex(
            cal.holidays(start=min(slots).normalize(), end=max(slots).normalize())
        ).normalize()
    )

    # 5) Build the recursive seed: one rolling history buffer per station, using only past data.
    # These buffers are the source of lag/rolling features and will later receive predictions too.
    df_hist = df_feat.loc[df_feat["date"] < start_ts, ["start_station_id", "date", "y_station"]]
    histories: dict[int, deque] = {}
    for sid in station_ids:
        arr = df_hist.loc[
            df_hist["start_station_id"].astype(int) == sid, "y_station"
        ].to_numpy(dtype=float)
        if len(arr) < max_needed:
            raise ValueError(
                f"Station {sid}: only {len(arr)} rows before {start_ts}, need {max_needed}."
            )
        histories[sid] = deque(arr.tolist(), maxlen=max_needed + 400)

    # 6) Recursive forecast loop.
    # For each future slot: rebuild model features from the current histories, predict, then
    # append the predictions back into the histories so the next slot can depend on them.
    pred_parts: list[pd.DataFrame] = []

    for ts in slots:
        # Calendar features shared by all stations for the current forecast slot.
        dayofweek = int(ts.dayofweek)
        dayofyear = int(ts.dayofyear)
        month = int(ts.month)
        hour = int(ts.hour)
        segment_id = 0 if hour == 0 else 1 if hour == 6 else 2

        common = {
            "date": ts,
            "segment_id": segment_id,
            "year": int(ts.year),
            "month": month,
            "month_num": month,
            "dayofweek": dayofweek,
            "dayofyear": dayofyear,
            "hour": hour,
            "is_weekend": int(dayofweek >= 5),
            "is_holiday": int(ts.normalize() in holiday_days),
            "dayofw_sin": np.sin(2 * np.pi * dayofweek / 7),
            "dayofw_cos": np.cos(2 * np.pi * dayofweek / 7),
            "dayofy_sin": np.sin(2 * np.pi * dayofyear / 365),
            "dayofy_cos": np.cos(2 * np.pi * dayofyear / 365),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
        }

        # Build one model-input row per station for this exact slot.
        # `common` is identical across stations; lags and rolling stats come from station history.
        step_rows = []
        for sid in station_ids:
            station_history = histories[sid]
            row = {"start_station_id": sid, **common}

            for lag in lags:
                row[f"lag_{lag}"] = station_history[-lag]

            for win in windows:
                recent_values = np.array(list(station_history)[-win:], dtype=float)
                row[f"roll_mean_{win}"] = float(recent_values.mean())
                row[f"roll_std_{win}"] = (
                    float(recent_values.std(ddof=1)) if len(recent_values) > 1 else np.nan
                )

            step_rows.append(row)

        df_step = pd.DataFrame(step_rows)
        missing = sorted(set(feature_cols) - set(df_step.columns))
        if missing:
            raise ValueError(f"Missing features at {ts}: {missing}")

        # Predict all stations for this slot, then feed predictions back into history.
        y_pred = predict_dual(df_step, bundle)

        # Keep only the output columns needed downstream for saving and evaluation.
        out_step = df_step[["start_station_id", "date", "segment_id"]].copy()
        out_step["y_pred"] = y_pred
        pred_parts.append(out_step)

        # This update is what makes the procedure recursive: future slots can use prior predictions.
        for sid, y_hat in zip(df_step["start_station_id"].to_numpy(), y_pred):
            histories[int(sid)].append(float(y_hat))

    pred_df = pd.concat(pred_parts, ignore_index=True)

    # 7) If the historical table already contains targets on the requested horizon, evaluate them.
    # Otherwise the forecast is treated as pure future inference and no metric is computed.
    actual = df_feat.loc[
        (df_feat["date"] >= pred_df["date"].min()) & (df_feat["date"] <= pred_df["date"].max()),
        ["start_station_id", "date", "segment_id", "y_station"],
    ]
    eval_df = pred_df.merge(actual, on=["start_station_id", "date", "segment_id"], how="left")

    if eval_df["y_station"].notna().any():
        m = eval_df["y_station"].notna()
        compute_metrics(eval_df.loc[m, "y_station"], eval_df.loc[m, "y_pred"], "RecursiveOffline")
    else:
        print("No ground truth on requested horizon (pure future forecast).")

    # 8) Persist the full forecast table with predictions and optional ground truth side by side.
    output_path = paths["DATA_DIR"] / "processed" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_path, index=False)
    print(f"Saved recursive forecast: {output_path}")
    print(f"Forecast window: {pred_df['date'].min()} -> {pred_df['date'].max()} | rows={len(pred_df)}")

    result: dict = {
        "output_path": output_path,
        "eval_df": eval_df,
        "station_output_path": None,
        "station_df": None,
    }

    # 9) Optional station-level extraction: return and optionally save one station's horizon only.
    if station_id is not None:
        station_mask = eval_df["start_station_id"].astype(int) == int(station_id)
        station_df = eval_df.loc[station_mask].copy()

        if station_df.empty:
            raise ValueError(
                f"No predictions found for station_id={station_id}. "
                "Check that this station exists in the training features."
            )

        if station_df["y_station"].notna().any():
            m_station = station_df["y_station"].notna()
            compute_metrics(
                station_df.loc[m_station, "y_station"],
                station_df.loc[m_station, "y_pred"],
                f"RecursiveOffline_Station_{station_id}",
            )

        if station_output_filename:
            station_filename = station_output_filename
        else:
            station_filename = f"predictions_recursive_days_station_{station_id}.csv"

        station_output_path = None
        if save_station_output:
            station_output_path = paths["DATA_DIR"] / "processed" / station_filename
            station_output_path.parent.mkdir(parents=True, exist_ok=True)
            station_df.to_csv(station_output_path, index=False)
            print(f"Saved station recursive forecast: {station_output_path}")

        result["station_output_path"] = station_output_path
        result["station_df"] = station_df

    return result
