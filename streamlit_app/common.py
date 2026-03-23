from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from bike_demand_forecasting.utils import align_to_segment_start as _align_to_segment_start_shared
except ModuleNotFoundError:
    _align_to_segment_start_shared = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MERGED_PATH = PROJECT_ROOT / "data" / "merged" / "all_merged.csv"
COORDS_CACHE_PATH = PROCESSED_DIR / "station_coordinates.csv"
STATION_NAMES_CACHE_PATH = PROCESSED_DIR / "station_names.csv"
SEGMENT_LABELS = {0: "00h-06h", 1: "06h-16h", 2: "16h-00h"}
SEGMENT_ORDER = ["00h-06h", "06h-16h", "16h-00h"]
CARD_BLUE = "#3b82f6"
CARD_GREEN = "#10b981"
CARD_AMBER = "#f59e0b"
CARD_RED = "#ef4444"
CARD_NEUTRAL = "#94a3b8"
PLOTLY_DISCRETE_COLORS = [CARD_BLUE, CARD_GREEN, CARD_AMBER, CARD_RED, CARD_NEUTRAL]
PLOTLY_CONTINUOUS_SCALE = [
    [0.0, CARD_BLUE],
    [0.33, CARD_GREEN],
    [0.66, CARD_AMBER],
    [1.0, CARD_RED],
]


@st.cache_data(show_spinner=False)
def load_predictions(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def normalize_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "start_station_id" in out.columns:
        out["start_station_id"] = pd.to_numeric(out["start_station_id"], errors="coerce").astype("Int64")
    if "segment_id" in out.columns:
        out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce").astype("Int64")
    if "y_pred" in out.columns:
        out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    if "y_station" in out.columns:
        out["y_station"] = pd.to_numeric(out["y_station"], errors="coerce")

    out = out.dropna(subset=["date", "start_station_id", "segment_id", "y_pred"]).copy()
    out["start_station_id"] = out["start_station_id"].astype(int)
    out["segment_id"] = out["segment_id"].astype(int)
    out["segment_label"] = out["segment_id"].map(SEGMENT_LABELS).fillna(out["segment_id"].astype(str))
    out["segment_label"] = pd.Categorical(out["segment_label"], categories=SEGMENT_ORDER, ordered=True)
    out["date_day"] = out["date"].dt.date
    return out.sort_values(["date", "start_station_id"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_station_coordinates() -> pd.DataFrame:
    if COORDS_CACHE_PATH.exists():
        coords = pd.read_csv(COORDS_CACHE_PATH)
        return coords.dropna(subset=["start_station_id", "start_lat", "start_lng"])

    if not MERGED_PATH.exists():
        return pd.DataFrame(columns=["start_station_id", "start_lat", "start_lng", "rows"])

    acc: pd.DataFrame | None = None
    usecols = ["start_station_id", "start_lat", "start_lng"]
    for chunk in pd.read_csv(
        MERGED_PATH,
        usecols=usecols,
        encoding="latin-1",
        chunksize=500_000,
    ):
        chunk["start_station_id"] = pd.to_numeric(chunk["start_station_id"], errors="coerce")
        chunk["start_lat"] = pd.to_numeric(chunk["start_lat"], errors="coerce")
        chunk["start_lng"] = pd.to_numeric(chunk["start_lng"], errors="coerce")
        chunk = chunk.dropna(subset=["start_station_id", "start_lat", "start_lng"])
        if chunk.empty:
            continue

        grouped = chunk.groupby("start_station_id", as_index=True).agg(
            lat_sum=("start_lat", "sum"),
            lng_sum=("start_lng", "sum"),
            rows=("start_lat", "size"),
        )
        if acc is None:
            acc = grouped
        else:
            acc = acc.add(grouped, fill_value=0.0)

    if acc is None or acc.empty:
        return pd.DataFrame(columns=["start_station_id", "start_lat", "start_lng", "rows"])

    coords = acc.reset_index()
    coords["start_lat"] = coords["lat_sum"] / coords["rows"]
    coords["start_lng"] = coords["lng_sum"] / coords["rows"]
    coords["start_station_id"] = coords["start_station_id"].astype(int)
    coords = coords[["start_station_id", "start_lat", "start_lng", "rows"]]
    COORDS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    coords.to_csv(COORDS_CACHE_PATH, index=False)
    return coords


@st.cache_data(show_spinner=False)
def load_station_names() -> pd.DataFrame:
    if STATION_NAMES_CACHE_PATH.exists():
        df_names = pd.read_csv(STATION_NAMES_CACHE_PATH)
        return df_names.dropna(subset=["start_station_id", "start_station_name"])

    if not MERGED_PATH.exists():
        return pd.DataFrame(columns=["start_station_id", "start_station_name", "rows"])

    grouped_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        MERGED_PATH,
        usecols=["start_station_id", "start_station_name"],
        encoding="latin-1",
        chunksize=500_000,
    ):
        chunk["start_station_id"] = pd.to_numeric(chunk["start_station_id"], errors="coerce").astype("Int64")
        chunk["start_station_name"] = chunk["start_station_name"].astype(str).str.strip()
        chunk = chunk.dropna(subset=["start_station_id"])
        chunk = chunk[chunk["start_station_name"] != ""]
        if chunk.empty:
            continue

        part = (
            chunk.groupby(["start_station_id", "start_station_name"], as_index=False)
            .size()
            .rename(columns={"size": "rows"})
        )
        grouped_parts.append(part)

    if not grouped_parts:
        return pd.DataFrame(columns=["start_station_id", "start_station_name", "rows"])

    all_parts = pd.concat(grouped_parts, ignore_index=True)
    agg = (
        all_parts.groupby(["start_station_id", "start_station_name"], as_index=False)["rows"]
        .sum()
        .sort_values(["start_station_id", "rows"], ascending=[True, False])
    )
    best = agg.drop_duplicates(subset=["start_station_id"], keep="first").copy()
    best["start_station_id"] = best["start_station_id"].astype(int)
    STATION_NAMES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(STATION_NAMES_CACHE_PATH, index=False)
    return best


@st.cache_data(show_spinner=False)
def get_station_ids_from_prediction_file(filename: str) -> list[int]:
    if not filename:
        return []
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        return []

    try:
        s = pd.read_csv(file_path, usecols=["start_station_id"])["start_station_id"]
    except Exception:
        return []

    ids = pd.to_numeric(s, errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(ids)


@st.cache_data(show_spinner=False)
def get_feature_date_max(features_filename: str) -> pd.Timestamp | None:
    if not features_filename:
        return None
    file_path = PROCESSED_DIR / features_filename
    if not file_path.exists():
        return None
    try:
        d = pd.to_datetime(pd.read_csv(file_path, usecols=["date"])["date"], errors="coerce").dropna()
    except Exception:
        return None
    if d.empty:
        return None
    return pd.Timestamp(d.max())


@st.cache_data(show_spinner=False)
def get_training_date_max(
    artifact_prefix: str,
    features_filename: str | None = None,
) -> tuple[pd.Timestamp | None, str]:
    """
    Resolve a training data max date for display.

    Source priority:
    1) explicit date fields in model metadata
    2) metadata cutoff (if not `full_data_refit`)
    3) fallback to selected features file date max
    """
    meta_path = MODELS_DIR / f"{artifact_prefix}_dual_meta.joblib"
    if meta_path.exists():
        try:
            meta = joblib.load(meta_path)
        except Exception:
            meta = {}

        for key in ("train_date_max", "data_date_max", "feature_date_max"):
            if key in meta and meta[key]:
                try:
                    return pd.Timestamp(pd.to_datetime(meta[key], errors="raise")), f"meta:{key}"
                except Exception:
                    pass

        cutoff = meta.get("cutoff")
        if cutoff and str(cutoff) != "full_data_refit":
            try:
                return pd.Timestamp(pd.to_datetime(cutoff, errors="raise")), "meta:cutoff"
            except Exception:
                pass

    if features_filename:
        feat_max = get_feature_date_max(features_filename)
        if feat_max is not None:
            return feat_max, "features:fallback"

    return None, "unavailable"


def api_post(base_url: str, endpoint: str, payload: dict) -> tuple[int, dict]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    raw = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=1800) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        detail = {"detail": body}
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            pass
        return exc.code, detail
    except Exception as exc:
        return 0, {"detail": str(exc)}


def validate_start_date_input(value: str) -> tuple[bool, str | None]:
    if not value or not value.strip():
        return True, None
    try:
        pd.to_datetime(value, errors="raise")
        return True, None
    except Exception:
        return False, (
            "Format de date invalide. Formats attendus: "
            "YYYY-MM-DD ou YYYY-MM-DDTHH:MM:SS "
            "(ex. : 2026-03-10T06:00:00)."
        )


def align_to_segment_start_ui(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Align UI-provided datetime to segment start (00:00, 06:00, 16:00).
    """
    if _align_to_segment_start_shared is not None:
        return _align_to_segment_start_shared(ts)

    # Local fallback when src package is unavailable in streamlit runtime.
    out = pd.Timestamp(ts)
    if out.tzinfo is not None:
        out = out.tz_convert(None)
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


def load_processed_df_to_state(filename: str | None, state_key: str) -> None:
    if not filename:
        return
    out_path = PROCESSED_DIR / filename
    if out_path.exists():
        st.session_state[state_key] = normalize_prediction_df(load_predictions(out_path))


def run_api_action(
    base_url: str,
    endpoint: str,
    payload: dict,
    spinner_text: str,
    success_text: str,
) -> dict | None:
    with st.spinner(spinner_text):
        code, data = api_post(base_url, endpoint, payload)

    st.session_state["last_response"] = data
    if code != 200:
        st.error(f"Erreur API ({code})")
        st.json(data)
        return None

    st.success(success_text)
    with st.expander("Réponse API"):
        st.json(data)
    return data


def init_session_state(defaults: dict[str, object]) -> None:
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_metric_css() -> None:
    st.markdown(
        """
        <style>
        .kpi-card {
            border-radius: 14px;
            padding: 12px 14px;
            margin-bottom: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
        }
        .kpi-label {
            font-size: 0.82rem;
            opacity: 0.82;
            margin-bottom: 0.15rem;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .kpi-sub {
            margin-top: 0.25rem;
            font-size: 0.75rem;
            opacity: 0.78;
        }
        .kpi-blue { border-left: 4px solid #3b82f6; }
        .kpi-green { border-left: 4px solid #10b981; }
        .kpi-amber { border-left: 4px solid #f59e0b; }
        .kpi-red { border-left: 4px solid #ef4444; }
        .kpi-neutral { border-left: 4px solid #94a3b8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_metric_value(value: float | int, decimals: int = 0) -> str:
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def render_kpi_cards(items: list[dict], cols: int = 4) -> None:
    if not items:
        return
    rows = [items[i : i + cols] for i in range(0, len(items), cols)]
    for row_items in rows:
        row_cols = st.columns(len(row_items))
        for idx, item in enumerate(row_items):
            with row_cols[idx]:
                st.metric(label=item["label"], value=item["value"])
                sub = item.get("sub")
                if sub:
                    st.caption(str(sub))


def style_plotly_figure(
    fig,
    *,
    height: int = 430,
    show_legend: bool = True,
    legend_title: str | None = None,
) -> None:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        showlegend=show_legend,
        legend_title_text=legend_title,
        margin=dict(l=20, r=20, t=58, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)


def _table_base_styler(df: pd.DataFrame):
    return (
        df.style.set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "rgba(148,163,184,0.16)"),
                        ("color", "#e5e7eb"),
                        ("font-weight", "600"),
                        ("border", "1px solid rgba(148,163,184,0.20)"),
                    ],
                },
                {
                    "selector": "tbody td",
                    "props": [
                        ("background-color", "rgba(15,23,42,0.35)"),
                        ("color", "#dbe4f0"),
                        ("border", "1px solid rgba(148,163,184,0.16)"),
                    ],
                },
            ]
        )
        .hide(axis="index")
        .set_properties(**{"text-align": "left"})
    )


def _severity_cell_style(value) -> str:
    v = str(value).strip().lower()
    if v == "critical":
        return "background-color: rgba(239,68,68,0.20); color: #fecaca; font-weight: 600;"
    if v == "warning":
        return "background-color: rgba(245,158,11,0.20); color: #fde68a; font-weight: 600;"
    return "color: #dbe4f0;"


def _priority_cell_style(value) -> str:
    v = str(value).strip().lower()
    if v in {"critical", "high"}:
        return "background-color: rgba(239,68,68,0.20); color: #fecaca; font-weight: 600;"
    if v in {"warning", "medium"}:
        return "background-color: rgba(245,158,11,0.20); color: #fde68a; font-weight: 600;"
    if v in {"low", "ok"}:
        return "background-color: rgba(16,185,129,0.18); color: #bbf7d0; font-weight: 600;"
    return "color: #dbe4f0;"


def style_alerts_table(df: pd.DataFrame):
    style_df = df.copy()
    styler = _table_base_styler(style_df)
    fmt = {}
    for c in ["start_station_id", "capacity_bikes", "min_buffer", "deficit_qty", "surplus_qty"]:
        if c in style_df.columns:
            fmt[c] = "{:,.0f}"
    for c in ["bikes_available", "pred_out_horizon", "pred_net_out_horizon", "stock_proj"]:
        if c in style_df.columns:
            fmt[c] = "{:,.1f}"
    styler = styler.format(fmt, na_rep="-")
    if "severity" in style_df.columns:
        styler = styler.map(_severity_cell_style, subset=["severity"])
    if "deficit_qty" in style_df.columns:
        styler = styler.bar(subset=["deficit_qty"], color="rgba(239,68,68,0.35)")
    if "surplus_qty" in style_df.columns:
        styler = styler.bar(subset=["surplus_qty"], color="rgba(16,185,129,0.30)")
    return styler


def style_transfer_table(df: pd.DataFrame):
    style_df = df.copy()
    styler = _table_base_styler(style_df)
    fmt = {}
    for c in ["from_station_id", "to_station_id", "qty_bikes"]:
        if c in style_df.columns:
            fmt[c] = "{:,.0f}"
    if "distance_km" in style_df.columns:
        fmt["distance_km"] = "{:,.2f}"
    styler = styler.format(fmt, na_rep="-")
    if "priority" in style_df.columns:
        styler = styler.map(_priority_cell_style, subset=["priority"])
    if "qty_bikes" in style_df.columns:
        styler = styler.bar(subset=["qty_bikes"], color="rgba(59,130,246,0.35)")
    return styler


def style_route_table(df: pd.DataFrame):
    style_df = df.copy()
    styler = _table_base_styler(style_df)
    fmt = {}
    for c in ["step", "from_station_id", "to_station_id", "qty_bikes"]:
        if c in style_df.columns:
            fmt[c] = "{:,.0f}"
    if "distance_km" in style_df.columns:
        fmt["distance_km"] = "{:,.2f}"
    styler = styler.format(fmt, na_rep="-")
    if "priority" in style_df.columns:
        styler = styler.map(_priority_cell_style, subset=["priority"])
    return styler
