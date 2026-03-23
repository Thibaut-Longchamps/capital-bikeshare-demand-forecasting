from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from bike_demand_forecasting.utils import get_paths


def load_models_and_meta(artifact_prefix: str = "catboost_station_3segments_fullfit_v1", parents_n=2):
    paths = get_paths(Path(__file__).resolve().parents[parents_n])
    model_dir = paths["WORK_DIR"] / "models"

    model_high = joblib.load(model_dir / f"{artifact_prefix}_high.joblib")
    model_low = joblib.load(model_dir / f"{artifact_prefix}_low.joblib")
    meta = joblib.load(model_dir / f"{artifact_prefix}_dual_meta.joblib")

    return {
        "model_high": model_high,
        "model_low": model_low,
        "meta": meta,
        "feature_cols_cb": meta["feature_cols_cb"],
        "high_stations": set(meta["high_stations"]),
        "low_stations": set(meta["low_stations"]),
    }
    
    
def predict_dual(df_in: pd.DataFrame, bundle: dict) -> np.ndarray:
    model_high = bundle["model_high"]
    model_low = bundle["model_low"]
    feature_cols = bundle["feature_cols_cb"]
    high_stations = bundle["high_stations"]
    low_stations = bundle["low_stations"]

    X = df_in[feature_cols].copy()

    sid = X["start_station_id"].astype(int)
    mask_high = sid.isin(high_stations).to_numpy()
    mask_low = sid.isin(low_stations).to_numpy()
    mask_unknown = ~(mask_high | mask_low)

    y_pred = np.zeros(len(X), dtype=float)

    if mask_high.any():
        y_pred[mask_high] = model_high.predict(X.loc[mask_high])
    if mask_low.any():
        y_pred[mask_low] = model_low.predict(X.loc[mask_low])
    if mask_unknown.any():
        # Fallback if the station was never seen during training.
        y_pred[mask_unknown] = model_low.predict(X.loc[mask_unknown])

    return y_pred
