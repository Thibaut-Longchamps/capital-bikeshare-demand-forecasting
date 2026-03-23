import argparse

from bike_demand_forecasting.offline_dataset import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build offline dataset for station 3-segment modeling.")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip ZIP extraction step.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip CSV merge step and use existing data/merged/all_merged.csv.",
    )
    args = parser.parse_args()

    main(
        skip_extract=args.skip_extract,
        skip_merge=args.skip_merge,
    )
