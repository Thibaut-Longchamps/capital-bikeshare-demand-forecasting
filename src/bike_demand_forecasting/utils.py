from pathlib import Path
import pandas as pd


def get_paths(work_dir: Path | None = None) -> dict[str, Path]:
    """
    Return the main project and data directory paths.
    """
    work_dir = work_dir or Path.cwd().parents[0]
    data_dir = work_dir / "data"

    return {
        "WORK_DIR": work_dir,
        "DATA_DIR": data_dir,
        "DATA_DIR_2024": data_dir / "raw" / "2024",
        "DATA_DIR_2025": data_dir / "raw" / "2025",
        "DATA_EXTRACTED": data_dir / "extracted",
        "DATA_EXTRACTED_DIR_2024": data_dir / "extracted" / "2024",
        "DATA_EXTRACTED_DIR_2025": data_dir / "extracted" / "2025",
        "DATA_DIR_MERGED": data_dir / "merged",
        "DATA_DIR_2024_MERGED": data_dir / "merged" / "2024",
        "DATA_DIR_2025_MERGED": data_dir / "merged" / "2025",
    }


def split_dev_test(df, test_pct = 0.2):
    """
    Split a time-ordered dataset into development (train/val) and test sets
    """
    size_test = round(len(df) * test_pct)

    df_dev = df.iloc[:-size_test].copy()
    df_test = df.iloc[-size_test:].copy()

    return df_dev, df_test


def separate_X_y(df):
    """
    Separate x et y for pipeline
    """
    X = df.drop("y", axis = 1)
    y = df["y"]

    return X, y
