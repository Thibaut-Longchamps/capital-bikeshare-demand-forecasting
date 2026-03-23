from pathlib import Path

from bike_demand_forecasting.io import extract_all_zips, merge_all_csv
from bike_demand_forecasting.preprocessing import (
    add_segment_name,
    add_time_features_3segments,
    build_complete_station_segment_panel,
    build_feature_table_3segments,
    build_station_segment_demand_from_csv,
)
from bike_demand_forecasting.utils import get_paths


def main(skip_extract: bool, skip_merge: bool) -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = get_paths(project_root)

    merged_csv_path = paths["DATA_DIR_MERGED"] / "all_merged.csv"

    # 1) Extract raw ZIP files if requested.
    if not skip_extract:
        extract_all_zips(paths["DATA_DIR_2024"], paths["DATA_EXTRACTED_DIR_2024"])
        extract_all_zips(paths["DATA_DIR_2025"], paths["DATA_EXTRACTED_DIR_2025"])
        print("Extraction done.")
    else:
        print("Skip extract.")

    # 2) Merge extracted CSV files or reuse the existing merged file.
    if not skip_merge:
        merged_csv_path = merge_all_csv(
            extract_dir=paths["DATA_EXTRACTED"],
            output_csv=paths["DATA_DIR_MERGED"] / "all_merged",
            encoding="latin-1",
        )
        print(f"Merged CSV: {merged_csv_path}")
    else:
        print(f"Skip merge. Using existing file: {merged_csv_path}")
        if not merged_csv_path.exists():
            raise FileNotFoundError(
                f"Merged file not found: {merged_csv_path}. "
                "Run without --skip-merge first."
            )

    # 3) Aggregate raw trips into the 3-segment target table.
    df_station = build_station_segment_demand_from_csv(data_merged_path=merged_csv_path)

    # 4) Build the complete station x segment panel and enrich it with time features.
    df_panel = build_complete_station_segment_panel(df_station)
    df_panel = add_time_features_3segments(df_panel)
    df_panel = add_segment_name(df_panel)

    # 5) Build the final training feature table.
    df_feat = build_feature_table_3segments(df_panel)

    # 6) Persist all intermediate/final offline datasets.
    processed_dir = paths["DATA_DIR"] / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    station_out = processed_dir / "station_segment_demand.csv"
    panel_out = processed_dir / "station_segment_panel.csv"
    feat_out = processed_dir / "features_3segments.csv"

    df_station.to_csv(station_out, index=False)
    df_panel.to_csv(panel_out, index=False)
    df_feat.to_csv(feat_out, index=False)

    print("Saved:")
    print(station_out)
    print(panel_out)
    print(feat_out)
