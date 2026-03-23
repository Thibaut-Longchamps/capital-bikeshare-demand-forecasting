from pathlib import Path

from bike_demand_forecasting.training import (
    fit_one_model,
    load_feature_table,
    save_artifacts,
    split_station_groups,
)
from bike_demand_forecasting.utils import get_paths


CAT_COLS = ["start_station_id", "is_holiday", "is_weekend", "segment_id"]


def main(
    features_filename: str,
    n_jobs: int,
    artifact_prefix: str,
    overwrite: bool,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = get_paths(project_root)

    print(f"Loading feature table from data/processed/{features_filename}...", flush=True)
    df_feat = load_feature_table(paths, features_filename)
    print(f"Feature table loaded: {len(df_feat):,} rows, {len(df_feat.columns)} columns.", flush=True)

    print("Preparing training matrices...", flush=True)
    X_train = df_feat.drop(columns="y_station").reset_index(drop=True)
    y_train = df_feat["y_station"].reset_index(drop=True)

    feature_cols = [
        c for c in X_train.columns if c not in ["date", "segment_name", "is_filled_zero"]
    ]
    print("Splitting stations into high-volume and low-volume groups...", flush=True)
    high_stations, low_stations = split_station_groups(X_train, y_train, volume_share=0.80)

    mask_train_high = X_train["start_station_id"].astype(int).isin(high_stations)
    mask_train_low = X_train["start_station_id"].astype(int).isin(low_stations)

    X_train_high = X_train.loc[mask_train_high].reset_index(drop=True)
    y_train_high = y_train.loc[mask_train_high].reset_index(drop=True)
    X_train_low = X_train.loc[mask_train_low].reset_index(drop=True)
    y_train_low = y_train.loc[mask_train_low].reset_index(drop=True)

    print(
        "Prepared subsets: "
        f"high_volume={len(high_stations)} stations / {len(X_train_high):,} rows, "
        f"low_volume={len(low_stations)} stations / {len(X_train_low):,} rows.",
        flush=True,
    )

    # Reuse validated params from offline model selection.
    param_grid_high = {
        "depth": [8],
        "learning_rate": [0.03],
        "iterations": [1000],
        "l2_leaf_reg": [3],
    }
    param_grid_low = {
        "depth": [6],
        "learning_rate": [0.03],
        "iterations": [1000],
        "l2_leaf_reg": [8],
    }

    print("Training final high-volume model...", flush=True)
    grid_high = fit_one_model(
        X_subset_full=X_train_high,
        y_subset=y_train_high,
        feature_cols=feature_cols,
        cat_cols=CAT_COLS,
        param_grid=param_grid_high,
        n_jobs=n_jobs,
        test_size=3 * 30,
        gap=42,
    )

    print("Refit/cross-validation for high-volume model completed.", flush=True)
    print("Training final low-volume model...", flush=True)
    grid_low = fit_one_model(
        X_subset_full=X_train_low,
        y_subset=y_train_low,
        feature_cols=feature_cols,
        cat_cols=CAT_COLS,
        param_grid=param_grid_low,
        n_jobs=n_jobs,
        test_size=3 * 30,
        gap=42,
    )

    best_model_high = grid_high.best_estimator_
    best_model_low = grid_low.best_estimator_
    print("Refit/cross-validation for low-volume model completed.", flush=True)
    print("High best params:", grid_high.best_params_, flush=True)
    print("Low best params :", grid_low.best_params_, flush=True)

    save_artifacts(
        out_dir=paths["WORK_DIR"] / "models",
        best_model_high=best_model_high,
        best_model_low=best_model_low,
        feature_cols=feature_cols,
        cat_cols=CAT_COLS,
        high_stations=high_stations,
        low_stations=low_stations,
        cutoff="full_data_refit",
        artifact_prefix=artifact_prefix,
        overwrite=overwrite,
        meta_extra={"train_mode": "full_fit"},
    )
