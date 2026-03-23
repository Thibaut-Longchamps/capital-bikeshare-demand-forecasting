from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def add_holiday_feature_us(df, dt_col="started_at"):
    """
    Add a binary `is_holiday` feature (1/0) based on US federal holidays.
    """
    df = df.copy()
    # Official US federal holiday calendar.
    cal = USFederalHolidayCalendar()
    # Time window covered by the data (normalized to midnight).
    start = df[dt_col].min().normalize()
    end = df[dt_col].max().normalize()
    # Holiday dates within that time window.
    us_holidays = cal.holidays(start=start, end=end)
    # 1 if the row date (ignoring time) is a holiday, else 0.
    df["is_holiday"] = df[dt_col].dt.normalize().isin(us_holidays).astype(int)
    return df

def add_sin_cos_features(df):
    """
    Add cyclical time encodings (sin/cos) for dayofweek, dayofyear, month, and hour.

    Why:
    Temporal variables are periodic (e.g., 23h is close to 0h). Using sin/cos maps each
    value onto a unit circle, preserving cyclical proximity and helping the model capture
    daily/weekly/seasonal patterns.
    """
    df = df.copy()
    # dayofweek
    df["dayofw_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofw_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    # dayofyear
    df["dayofy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    # month
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    # hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_lag_features_by_station(
    df,
    target_col="y_station",
    lags=(24, 168),
    station_col="start_station_id",
    time_col="date",
):
    """
    Add lagged target features computed independently for each station.
    """
    df = df.copy()
    # Keep original row order
    df["_row_order"] = range(len(df))
    # Sort to ensure correct temporal lag inside each station
    df = df.sort_values([station_col, time_col])
    g = df.groupby(station_col, sort=False)[target_col]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)
    # Restore original order
    df = df.sort_values("_row_order").drop(columns="_row_order")
    return df


def add_rolling_features_by_station(
    df,
    target_col="y_station",
    windows=(24, 168),
    shift_steps=24,
    station_col="start_station_id",
    time_col="date",
):
    """
    Add rolling mean/std features computed independently for each station.
    """
    df = df.copy()
    # Keep original order
    df["_row_order"] = range(len(df))
    # Sort for correct time order within each station
    df = df.sort_values([station_col, time_col])
    g = df.groupby(station_col, sort=False)[target_col]
    shifted = g.shift(shift_steps)

    for w in windows:
        roll = (
            shifted.groupby(df[station_col], sort=False)
            .rolling(window=w)
        )
        df[f"roll_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"roll_std_{w}"] = roll.std().reset_index(level=0, drop=True)

    # Restore original order
    df = df.sort_values("_row_order").drop(columns="_row_order")
    return df


def time_train_test_split(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "y_station",
    drop_cols: list[str] | None = None,
    train_ratio: float = 0.8,
):
    """
    Split a time-indexed dataset into train/test sets using a chronological cutoff.

    The function first computes a raw cutoff at `train_ratio` over unique timestamps,
    then aligns the final cutoff to the next Monday 00:00 to keep week boundaries clean.
    Rows strictly before the cutoff are used for training; rows at/after the cutoff are
    used for testing.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test, cutoff)
    """
    if drop_cols is None:
        drop_cols = [target_col, "is_filled_zero"]

    # 80% raw cutoff on unique timestamps
    dates_all = pd.DatetimeIndex(np.sort(df[date_col].dropna().unique()))
    raw_cutoff = dates_all[int(len(dates_all) * train_ratio)]

    # align to next Monday 00:00
    cutoff = raw_cutoff.normalize() - pd.Timedelta(days=raw_cutoff.weekday())
    if raw_cutoff > cutoff:
        cutoff += pd.Timedelta(days=7)

    # split
    mask_train = df[date_col] < cutoff
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = df.loc[mask_train, feature_cols]
    y_train = df.loc[mask_train, target_col]
    X_test = df.loc[~mask_train, feature_cols]
    y_test = df.loc[~mask_train, target_col]

    # sanity checks
    assert X_train[date_col].max() < X_test[date_col].min()
    assert set(X_train[date_col]).isdisjoint(set(X_test[date_col]))
    assert len(X_train) > 0 and len(X_test) > 0

    return X_train, y_train, X_test, y_test, cutoff


def make_cv_splits_by_date(X_sub, n_splits=3, test_size=3 * 30, gap=42):
    dates_sub = np.array(sorted(X_sub["date"].dropna().unique()))
    tscv_sub = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    splits = []
    for tr_idx, va_idx in tscv_sub.split(dates_sub):
        tr_start, tr_end = dates_sub[tr_idx[0]], dates_sub[tr_idx[-1]]
        va_start, va_end = dates_sub[va_idx[0]], dates_sub[va_idx[-1]]

        m_tr = (X_sub["date"] >= tr_start) & (X_sub["date"] <= tr_end)
        m_va = (X_sub["date"] >= va_start) & (X_sub["date"] <= va_end)

        idx_tr = np.flatnonzero(m_tr.to_numpy())
        idx_va = np.flatnonzero(m_va.to_numpy())
        splits.append((idx_tr, idx_va))

    return splits
