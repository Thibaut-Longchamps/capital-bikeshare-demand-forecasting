from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from bike_demand_forecasting.features import make_cv_splits_by_date
from bike_demand_forecasting.metrics import bias, smape


NUM_COLS_FLOAT32 = [
    "lag_1", "lag_2", "lag_3", "lag_21", "lag_42",
    "roll_mean_3", "roll_std_3", "roll_mean_21", "roll_std_21", "roll_mean_42", "roll_std_42",
    "dayofw_sin", "dayofw_cos", "dayofy_sin", "dayofy_cos",
    "month_sin", "month_cos", "hour_sin", "hour_cos", "year",
]


def load_feature_table(
    paths: dict[str, Path],
    filename: str,
    num_cols_float32: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the precomputed feature table and enforce stable dtypes used in training.

    The function parses `date`, downcasts numeric/categorical columns, coerces station IDs,
    and drops rows where lag/rolling features are not available.
    """
    # Resolve `data/processed/<filename>`.
    feat_path = paths["DATA_DIR"] / "processed" / filename
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")

    # Load and sanitize the timestamp column.
    df_feat = pd.read_csv(feat_path)
    df_feat["date"] = pd.to_datetime(df_feat["date"], errors="coerce")
    df_feat = df_feat.dropna(subset=["date"]).reset_index(drop=True)

    # Keep a compact float dtype for model numerical features.
    cols_to_cast = num_cols_float32 or NUM_COLS_FLOAT32
    for col in cols_to_cast:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype("float32")

    # Keep binary/categorical columns compact and explicit.
    for col in ["is_holiday", "is_weekend", "segment_id"]:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype("int8")

    # Normalize station IDs to nullable integer type.
    if "start_station_id" in df_feat.columns:
        df_feat["start_station_id"] = pd.to_numeric(
            df_feat["start_station_id"], errors="coerce"
        ).astype("Int32")

    # Remove warm-up rows that cannot provide lag/rolling values.
    lag_roll_cols = [c for c in df_feat.columns if c.startswith(("lag_", "roll_"))]
    if lag_roll_cols:
        df_feat = df_feat.dropna(subset=lag_roll_cols).reset_index(drop=True)

    return df_feat




def split_station_groups(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    volume_share: float = 0.80,
) -> tuple[set[int], set[int]]:
    """
    Split stations into high-volume and low-volume groups based on cumulative demand share.

    Stations are sorted by total target volume. The first stations covering `volume_share`
    of demand become the high-volume group; all others become the low-volume group.
    """
    # Compute per-station total demand and ranking.
    station_volume = (
        pd.DataFrame({"start_station_id": X_train["start_station_id"], "y_station": y_train})
        .groupby("start_station_id", as_index=False)["y_station"]
        .sum()
        .sort_values("y_station", ascending=False)
        .reset_index(drop=True)
    )
    station_volume["cum_share"] = station_volume["y_station"].cumsum() / station_volume["y_station"].sum()

    # Select top-demand stations until the cumulative share threshold.
    high_stations = set(
        station_volume.loc[station_volume["cum_share"] <= volume_share, "start_station_id"]
        .astype(int)
        .tolist()
    )
    # Ensure the high-volume set is never empty.
    if not high_stations:
        high_stations = {int(station_volume["start_station_id"].iloc[0])}

    # Remaining stations define the low-volume set.
    all_stations = set(station_volume["start_station_id"].astype(int).tolist())
    low_stations = all_stations - high_stations
    # Ensure the low-volume set is never empty.
    if not low_stations:
        sid = int(station_volume["start_station_id"].iloc[-1])
        high_stations.discard(sid)
        low_stations.add(sid)

    return high_stations, low_stations





def fit_one_model(
    X_subset_full: pd.DataFrame,
    y_subset: pd.Series,
    feature_cols: list[str],
    cat_cols: list[str],
    param_grid: dict,
    n_jobs: int,
    test_size: int,
    gap: int,
) -> GridSearchCV:
    """
    Train a CatBoost model with time-aware CV and hyperparameter grid search.

    CV folds are generated on dates with a temporal gap to reduce leakage.
    The best model is selected with MAE (`refit="mae"`).
    """
    from catboost import CatBoostRegressor

    # Base model configuration shared across all hyperparameter candidates.
    base_cb = CatBoostRegressor(
        loss_function="Poisson",
        eval_metric="Poisson",
        random_seed=42,
        thread_count=1,
        verbose=100,
        allow_writing_files=False,
    )

    # Build chronological CV folds from date-based splits.
    cv_splits = make_cv_splits_by_date(
        X_subset_full,
        n_splits=2,
        test_size=test_size,
        gap=gap,
    )

    # Multi-metric evaluation; MAE is used to pick the final estimator.
    scoring = {
        "mae": "neg_mean_absolute_error",
        "smape": make_scorer(smape, greater_is_better=False),
        "bias": make_scorer(bias, greater_is_better=False),
    }

    # Exhaustive grid search over the provided parameter grid.
    grid = GridSearchCV(
        estimator=base_cb,
        param_grid=param_grid,
        cv=cv_splits,
        scoring=scoring,
        refit="mae",
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        verbose=100,
    )
    # Train on selected model columns with CatBoost categorical feature indices/names.
    grid.fit(X_subset_full[feature_cols], y_subset, cat_features=cat_cols)
    return grid




def save_artifacts(
    out_dir: Path,
    best_model_high,
    best_model_low,
    feature_cols: list[str],
    cat_cols: list[str],
    high_stations: set[int],
    low_stations: set[int],
    cutoff,
    artifact_prefix: str = "catboost_station_3segments",
    overwrite: bool = True,
    meta_extra: dict | None = None,
) -> dict[str, Path]:
    """
    Persist trained models and metadata to disk.

    Saves two model files (high/low volume) and one metadata file describing
    feature columns, categorical columns, station groups, and training context.
    """
    # Ensure output directory exists.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Standardized artifact paths.
    high_path = out_dir / f"{artifact_prefix}_high.joblib"
    low_path = out_dir / f"{artifact_prefix}_low.joblib"
    meta_path = out_dir / f"{artifact_prefix}_dual_meta.joblib"

    # Optional safety check to avoid accidental overwrite.
    if not overwrite:
        for path in (high_path, low_path, meta_path):
            if path.exists():
                raise FileExistsError(f"Artifact already exists: {path}")

    # Metadata needed at inference time to rebuild the exact feature interface.
    meta = {
        "feature_cols_cb": feature_cols,
        "cat_cols": cat_cols,
        "high_stations": sorted(list(high_stations)),
        "low_stations": sorted(list(low_stations)),
        "volume_split_cum_share": 0.80,
        "cutoff": str(cutoff),
        "artifact_prefix": artifact_prefix,
    }
    if meta_extra:
        meta.update(meta_extra)

    # Serialize models and metadata with joblib.
    joblib.dump(best_model_high, high_path)
    joblib.dump(best_model_low, low_path)
    joblib.dump(meta, meta_path)

    print("Saved models and metadata:")
    print(high_path)
    print(low_path)
    print(meta_path)

    return {"high": high_path, "low": low_path, "meta": meta_path}
