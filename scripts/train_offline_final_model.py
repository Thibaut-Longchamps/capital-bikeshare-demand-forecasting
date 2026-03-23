import argparse

from bike_demand_forecasting.final_training import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train final dual CatBoost model on full dataset.")
    parser.add_argument(
        "--features-filename",
        default="features_3segments.csv",
        help="CSV filename in data/processed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="GridSearchCV parallel jobs.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="catboost_station_3segments_fullfit_v1",
        help="Prefix for saved artifact filenames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing artifacts if they already exist.",
    )
    args = parser.parse_args()

    main(
        features_filename=args.features_filename,
        n_jobs=args.n_jobs,
        artifact_prefix=args.artifact_prefix,
        overwrite=args.overwrite,
    )
