from pathlib import Path

from bike_demand_forecasting.metrics import compute_metrics
from bike_demand_forecasting.training import load_feature_table
from bike_demand_forecasting.utils import get_paths


def main(input_filename: str, station_id: int, output_filename: str | None) -> None:
    if not input_filename:
        raise ValueError("--input-filename is required.")

    project_root = Path(__file__).resolve().parents[2]
    paths = get_paths(project_root)

    df = load_feature_table(paths, input_filename)
    if "start_station_id" not in df.columns:
        raise ValueError("Input file must contain start_station_id.")
    if "y_pred" not in df.columns:
        raise ValueError("Input file must contain y_pred.")

    station_df = df.loc[df["start_station_id"].astype(int) == int(station_id)].copy()
    if station_df.empty:
        raise ValueError(
            f"No rows found for station_id={station_id} in {input_filename}."
        )

    if "y_station" in station_df.columns and station_df["y_station"].notna().any():
        m = station_df["y_station"].notna()
        compute_metrics(
            station_df.loc[m, "y_station"],
            station_df.loc[m, "y_pred"],
            f"Station_{station_id}",
        )

    if output_filename:
        out_name = output_filename
    else:
        stem = Path(input_filename).stem
        out_name = f"{stem}_station_{station_id}.csv"

    output_path = paths["DATA_DIR"] / "processed" / out_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    station_df.to_csv(output_path, index=False)
    print(f"Saved station export: {output_path}")
    print(f"Rows: {len(station_df)}")
