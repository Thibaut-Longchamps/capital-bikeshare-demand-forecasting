import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import pandas as pd


def smape(y_true, y_pred, eps=1e-8):
    """
    Compute smape metric
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))


def bias(y_true, y_pred):
    """
    Compute bias metric
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    return np.mean(y_pred - y_true)



def wape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps)


def compute_metrics(y_test, y_pred, info, mask=None):
    
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    smape_value = smape(y_test, y_pred)
    bias_value = bias(y_test, y_pred)
    wape_value = wape(y_test, y_pred)

    if mask is not None:
        print(f"Nb row {info}: {int(np.asarray(mask).sum())}")
    print(f"MAE {info}: {mae}")
    print(f"MAPE {info}: {mape}")
    print(f"sMAPE {info}: {smape_value}")
    print(f"Bias {info}: {bias_value}")
    print(f"WAPE {info}: {wape_value}")
    


def plot_actual_vs_pred(
    X,
    y_true,
    y_pred,
    date_col="date",
    aggregate=True,
    title="Actual vs Predicted Demand",
):
    import plotly.express as px

    plot_df = pd.DataFrame({
        "date": pd.to_datetime(X[date_col]).values,
        "y_test": pd.Series(y_true).values,
        "y_pred": pd.Series(y_pred).values,
    }).sort_values("date")

    if aggregate:
        plot_df = plot_df.groupby("date", as_index=False)[["y_test", "y_pred"]].sum()

    long_df = plot_df.melt(
        id_vars="date",
        value_vars=["y_test", "y_pred"],
        var_name="series",
        value_name="value"
    )

    fig = px.line(
        long_df,
        x="date",
        y="value",
        color="series",
        title=title
    )
    
    fig.update_traces(selector=dict(name="y_pred"), opacity=0.45)
    
    fig.show()
