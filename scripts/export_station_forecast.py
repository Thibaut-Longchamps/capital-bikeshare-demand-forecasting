import argparse

from bike_demand_forecasting.station_export import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export one station forecast from an existing prediction CSV."
    )
    parser.add_argument(
        "--input-filename",
        required=True,
        help="Prediction CSV filename located in data/processed.",
    )
    parser.add_argument(
        "--station-id",
        type=int,
        required=True,
        help="Station id to export.",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Optional output CSV filename in data/processed.",
    )
    args = parser.parse_args()

    main(
        input_filename=args.input_filename,
        station_id=args.station_id,
        output_filename=args.output_filename,
    )
