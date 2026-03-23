import argparse
from bike_demand_forecasting.recursive_forecast import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive multi-day forecast with dual CatBoost model.")
    parser.add_argument(
        "--features-filename",
        default="features_3segments.csv",
        help="Input features CSV in data/processed.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="catboost_station_3segments_fullfit_v1",
        help="Model artifacts prefix in models/.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Forecast horizon in days (3 segments/day).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional forecast start date/time (aligned to segment starts 00/06/16).",
    )
    parser.add_argument(
        "--output-filename",
        default="predictions_recursive_days.csv",
        help="Output CSV filename in data/processed.",
    )
    parser.add_argument(
        "--station-id",
        type=int,
        default=None,
        help="Optional station id to export as a dedicated station-level forecast file.",
    )
    parser.add_argument(
        "--station-output-filename",
        default=None,
        help="Optional output CSV filename for station-level forecast in data/processed.",
    )
    parser.add_argument(
        "--no-station-output-file",
        action="store_true",
        help="Do not save station-level CSV even if --station-id is provided.",
    )
    args = parser.parse_args()

    main(
        features_filename=args.features_filename,
        artifact_prefix=args.artifact_prefix,
        days=args.days,
        output_filename=args.output_filename,
        start_date=args.start_date,
        station_id=args.station_id,
        station_output_filename=args.station_output_filename,
        save_station_output=not args.no_station_output_file,
    )
