from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd


def _resolve_app_timezone() -> ZoneInfo:
    """
    Resolve application timezone for all generated timestamps.

    Example
    -------
    _resolve_app_timezone() -> ZoneInfo(key='Europe/Paris')
    """
    return ZoneInfo("Europe/Paris")


APP_TIMEZONE = _resolve_app_timezone()


def now_local() -> datetime:
    """
    Return current time in the configured app timezone.

    Example
    -------
    now_local() -> datetime(2026, 3, 12, 17, 10, 5, tzinfo=ZoneInfo('Europe/Paris'))
    """
    return datetime.now(APP_TIMEZONE)


def format_timestamp_for_filename(ts: datetime | None = None) -> str:
    """
    Return a filesystem-safe local timestamp.

    Example
    -------
    format_timestamp_for_filename(datetime(2026, 3, 12, 17, 10, 5)) -> '20260312T171005'
    """
    dt = ts or now_local()
    return dt.strftime("%Y%m%dT%H%M%S")


def to_naive_timestamp(ts) -> pd.Timestamp:
    """
    Convert a timestamp-like value to timezone-naive pandas Timestamp.

    Example
    -------
    to_naive_timestamp('2026-03-12 16:00:00') -> Timestamp('2026-03-12 16:00:00')
    """
    out = pd.Timestamp(ts)
    if out.tzinfo is not None:
        out = out.tz_convert(None)
    return out


def align_to_segment_start(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Align to the next segment boundary (ceiling): 00:00, 06:00, 16:00.

    Example
    -------
    align_to_segment_start(Timestamp('2026-03-12 07:10:00')) -> Timestamp('2026-03-12 16:00:00')
    """
    out = to_naive_timestamp(ts)
    day = out.normalize()
    boundaries = [
        day,
        day + pd.Timedelta(hours=6),
        day + pd.Timedelta(hours=16),
        day + pd.Timedelta(days=1),
    ]
    for boundary in boundaries:
        if out <= boundary:
            return boundary
    return day + pd.Timedelta(days=1)


def next_segment_start(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Return next segment start among 00:00, 06:00, 16:00.

    Example
    -------
    next_segment_start(Timestamp('2026-03-12 06:00:00')) -> Timestamp('2026-03-12 16:00:00')
    """
    out = to_naive_timestamp(ts)
    h = int(out.hour)
    if h == 0:
        return out.normalize() + pd.Timedelta(hours=6)
    if h == 6:
        return out.normalize() + pd.Timedelta(hours=16)
    if h == 16:
        return out.normalize() + pd.Timedelta(days=1)
    raise ValueError(f"Unexpected segment hour={h}; expected 0/6/16.")


def get_paths(work_dir: Path | None = None) -> dict[str, Path]:
    """
    Return the main project and data directory paths.

    Example
    -------
    get_paths(Path('/repo'))['DATA_DIR'] -> Path('/repo/data')
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


def split_dev_test(df, test_pct=0.2):
    """
    Split a time-ordered dataset into development (train/val) and test sets

    Example
    -------
    split_dev_test(df, test_pct=0.2) -> (df_dev: first 80%, df_test: last 20%)
    """
    size_test = round(len(df) * test_pct)

    df_dev = df.iloc[:-size_test].copy()
    df_test = df.iloc[-size_test:].copy()

    return df_dev, df_test


def separate_X_y(df):
    """
    Separate x et y for pipeline

    Example
    -------
    separate_X_y(df) -> (X: all columns except 'y', y: df['y'])
    """
    X = df.drop("y", axis=1)
    y = df["y"]

    return X, y
