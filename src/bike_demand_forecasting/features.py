from pandas.tseries.holiday import USFederalHolidayCalendar

def add_holiday_feature_us(df, dt_col="started_at"):
    """
    Add a binary US federal holiday feature based on the datetime column.
    """
    df = df.copy()
    cal = USFederalHolidayCalendar()

    start = df[dt_col].min().normalize()
    end = df[dt_col].max().normalize()
    us_holidays = cal.holidays(start=start, end=end)

    df["is_holiday"] = df[dt_col].dt.normalize().isin(us_holidays).astype(int)
    return df


def add_lag_features(df, target_col="y", lags=(24, 168)):
    """
    Add lagged target features for the given lag steps.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col="y", windows=(24, 168)):
    """
    Add rolling mean and std features computed from past target values.
    """
    df = df.copy()
    for w in windows:
        # shift(24) to use past values only
        df[f"roll_mean_{w}"] = df[target_col].shift(24).rolling(window=w).mean()
        df[f"roll_std_{w}"] = df[target_col].shift(24).rolling(window=w).std()
    return df