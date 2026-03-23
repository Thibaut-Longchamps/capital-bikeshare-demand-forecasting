from pathlib import Path
import numpy as np
import pandas as pd

from bike_demand_forecasting.features import (
    add_holiday_feature_us,
    add_lag_features_by_station,
    add_rolling_features_by_station,
    add_sin_cos_features,
)

from bike_demand_forecasting.utils import get_paths


def build_station_segment_demand_from_csv(
    data_merged_path: str | Path,
    chunksize: int = 500_000,
    started_at_col: str = "started_at",
    station_col: str = "start_station_id",
) -> pd.DataFrame:
    """
    Build station demand at 3-segment granularity (00:00, 06:00, 16:00) from raw trips CSV.
    This function takes a raw CSV of individual trips and converts it into an aggregated table:

    one row = one station for one time-of-day segment
    y_station = number of trips in that segment
    
    So it transforms the data from “1 row = 1 trip” into “1 row = 1 aggregated demand count.
    """
    # Collect per-chunk station demand before global aggregation.
    list_station_demand: list[pd.DataFrame] = []

    # Stream the input file to keep memory usage bounded.
    reader = pd.read_csv(
        data_merged_path,
        usecols=[started_at_col, station_col],
        dtype={started_at_col: "string", station_col: "string"},
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in reader:
        # Clean timestamp strings and parse to datetime.
        s = chunk[started_at_col].str.strip().str.replace(r"\.\d+$", "", regex=True)
        dt = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        # Parse station id as numeric, invalid ids become NA.
        sid = pd.to_numeric(chunk[station_col].str.strip(), errors="coerce").astype("Int32")

        # Keep only rows with both valid timestamp and station id.
        valid = dt.notna() & sid.notna()
        if not valid.any():
            continue

        dtv = dt.loc[valid]
        sidv = sid.loc[valid]
        hour = dtv.dt.hour

        # Map each trip start hour to one of the 3 daily segments.
        segment_id = np.select([hour < 6, hour < 16], [0, 1], default=2).astype("int8")
        # Segment representative timestamps are fixed at 00:00 / 06:00 / 16:00.
        segment_start_hour = np.select([hour < 6, hour < 16], [0, 6], default=16).astype("int8")
        date_segment = dtv.dt.normalize() + pd.to_timedelta(segment_start_hour, unit="h")

        # Build a minimal per-trip table before counting demand.
        station_demand = pd.DataFrame(
            {
                "start_station_id": sidv.to_numpy(),
                "date": date_segment.to_numpy(),
                "segment_id": segment_id,
            }
        )
        # Count number of trips per station x segment timestamp.
        station_demand = (
            station_demand.groupby(["start_station_id", "date", "segment_id"], as_index=False)
            .size()
            .rename(columns={"size": "y_station"})
        )
        list_station_demand.append(station_demand)

    if not list_station_demand:
        raise ValueError("No valid station demand rows were built from input CSV.")

    # Merge chunk-level counts into one final demand table.
    df_station = (
        pd.concat(list_station_demand, ignore_index=True)
        .groupby(["start_station_id", "date", "segment_id"], as_index=False)["y_station"]
        .sum()
        .sort_values(["start_station_id", "date", "segment_id"])
        .reset_index(drop=True)
    )
    df_station["y_station"] = df_station["y_station"].astype("int32")
    return df_station



def build_complete_station_segment_panel(df_station: pd.DataFrame) -> pd.DataFrame:
    """
    Build full station x day x segment panel and fill missing demand with zero.
    """
    # Unique station list used to create the full panel index.
    stations = pd.DataFrame(
        {"start_station_id": df_station["start_station_id"].drop_duplicates().sort_values()}
    )

    # Continuous daily range over the full dataset period.
    day_range = pd.date_range(
        df_station["date"].min().normalize(),
        df_station["date"].max().normalize(),
        freq="D",
    )
    days = pd.DataFrame({"day": day_range})

    # Fixed 3-segment definition for each day.
    segment_slots = pd.DataFrame(
        {
            "segment_id": pd.Series([0, 1, 2], dtype="int8"),
            "segment_start_hour": pd.Series([0, 6, 16], dtype="int8"),
        }
    )

    # Expand each day into 3 timestamped segment slots.
    day_segment_slots = days.merge(segment_slots, how="cross")
    day_segment_slots["date"] = day_segment_slots["day"] + pd.to_timedelta(
        day_segment_slots["segment_start_hour"], unit="h"
    )
    day_segment_slots = day_segment_slots[["date", "segment_id"]]

    # Cartesian product station x (day, segment) = complete panel index.
    full_index = stations.merge(day_segment_slots, how="cross")
    # Left-join observed demand; missing combos represent zero-demand slots.
    panel = full_index.merge(
        df_station,
        on=["start_station_id", "date", "segment_id"],
        how="left",
    )

    # Fill missing demand with zero and track which rows were synthetic fills.
    panel["y_station"] = panel["y_station"].fillna(0).astype("int32")
    panel["is_filled_zero"] = (panel["y_station"] == 0).astype("int8")

    # Sort for deterministic temporal order per station.
    df_station_halfday = panel.sort_values(
        ["start_station_id", "date", "segment_id"]
    ).reset_index(drop=True)
    return df_station_halfday


def add_time_features_3segments(df_station_halfday: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features at 3-segment granularity.
    """
    df = df_station_halfday.copy()
    # Calendar decompositions extracted from the segment timestamp.
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["month_num"] = df["month"]
    df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")
    df["dayofyear"] = df["date"].dt.dayofyear.astype("int16")
    df["hour"] = df["date"].dt.hour.astype("int8")
    # Keep segment id compact and consistent.
    df["segment_id"] = df["segment_id"].astype("int8")
    # Weekend flag used as a categorical signal.
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    return df


def add_segment_name(df_station_halfday: pd.DataFrame) -> pd.DataFrame:
    """
    Add readable segment labels.
    """
    df = df_station_halfday.copy()
    # Human-readable segment labels used for analysis/UI only.
    segment_name_map = {0: "night_00_05", 1: "day_06_15", 2: "evening_16_23"}
    df["segment_name"] = df["segment_id"].map(segment_name_map)
    return df


def build_feature_table_3segments(
    df_station_halfday: pd.DataFrame,
    target_col: str = "y_station",
) -> pd.DataFrame:
    """
    Build model feature table (holiday, lags, rolling, cyclical, dtype optimization).
    """
    # Holiday flag on the segment timestamp date.
    df_feat = add_holiday_feature_us(df_station_halfday, dt_col="date")
    # Station-specific autoregressive lags in segment units.
    df_feat = add_lag_features_by_station(
        df_feat,
        target_col=target_col,
        lags=(1, 2, 3, 21, 42),
        station_col="start_station_id",
        time_col="date",
    )
    # Station-specific rolling stats computed from past observations only.
    df_feat = add_rolling_features_by_station(
        df_feat,
        target_col=target_col,
        windows=(3, 21, 42),
        shift_steps=1,
        station_col="start_station_id",
        time_col="date",
    )
    # Cyclical encoding for temporal variables (sin/cos transforms).
    df_feat = add_sin_cos_features(df_feat)

    num_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_21",
        "lag_42",
        "roll_mean_3",
        "roll_std_3",
        "roll_mean_21",
        "roll_std_21",
        "roll_mean_42",
        "roll_std_42",
        "dayofw_sin",
        "dayofw_cos",
        "dayofy_sin",
        "dayofy_cos",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "year",
    ]

    # Downcast numeric features to reduce memory footprint.
    for col in num_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype("float32")

    # Keep binary/categorical calendar flags compact.
    for col in ["is_holiday", "is_weekend", "segment_id"]:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype("int8")

    # Drop warm-up rows where lag/rolling features are undefined.
    lag_roll_cols = [col for col in df_feat.columns if col.startswith(("lag_", "roll_"))]
    df_feat = df_feat.dropna(subset=lag_roll_cols).reset_index(drop=True)
    return df_feat
