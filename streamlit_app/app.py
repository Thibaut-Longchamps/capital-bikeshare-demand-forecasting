from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    # Package import path when the app is launched from the project root.
    from streamlit_app.common import (
        CARD_AMBER,
        CARD_BLUE,
        CARD_GREEN,
        CARD_NEUTRAL,
        CARD_RED,
        PLOTLY_CONTINUOUS_SCALE,
        PLOTLY_DISCRETE_COLORS,
        PROCESSED_DIR,
        SEGMENT_ORDER,
        _format_metric_value,
        align_to_segment_start_ui,
        get_station_ids_from_prediction_file,
        init_session_state,
        inject_metric_css,
        load_processed_df_to_state,
        load_station_coordinates,
        load_station_names,
        render_kpi_cards,
        run_api_action,
        style_alerts_table,
        style_plotly_figure,
        style_route_table,
        style_transfer_table,
        validate_start_date_input,
    )
    from streamlit_app import common as _common
except ModuleNotFoundError:
    # Fallback import path when Streamlit runs from inside streamlit_app/.
    from common import (
        CARD_AMBER,
        CARD_BLUE,
        CARD_GREEN,
        CARD_NEUTRAL,
        CARD_RED,
        PLOTLY_CONTINUOUS_SCALE,
        PLOTLY_DISCRETE_COLORS,
        PROCESSED_DIR,
        SEGMENT_ORDER,
        _format_metric_value,
        align_to_segment_start_ui,
        get_station_ids_from_prediction_file,
        init_session_state,
        inject_metric_css,
        load_processed_df_to_state,
        load_station_coordinates,
        load_station_names,
        render_kpi_cards,
        run_api_action,
        style_alerts_table,
        style_plotly_figure,
        style_route_table,
        style_transfer_table,
        validate_start_date_input,
    )
    import common as _common


def get_training_date_max(artifact_prefix: str, features_filename: str | None = None):
    # Keep the app resilient if the optional metadata helper is unavailable.
    fn = getattr(_common, "get_training_date_max", None)
    if callable(fn):
        return fn(artifact_prefix=artifact_prefix, features_filename=features_filename)
    return None, "unavailable"


def _render_network_overview(df: pd.DataFrame) -> None:
    # Top-level network KPIs plus horizon context for the loaded forecast.
    render_kpi_cards(
        [
            {
                "label": "Stations",
                "value": _format_metric_value(df["start_station_id"].nunique()),
                "tone": "blue",
            },
            {
                "label": "Lignes",
                "value": _format_metric_value(len(df)),
                "tone": "neutral",
            },
            {
                "label": "Prédictions totales",
                "value": _format_metric_value(df["y_pred"].sum()),
                "tone": "green",
            },
            {
                "label": "Moyenne / segment",
                "value": _format_metric_value(df["y_pred"].mean(), decimals=2),
                "tone": "amber",
            },
        ],
        cols=4,
    )

    start_dt = pd.Timestamp(df["date"].min())
    end_dt = pd.Timestamp(df["date"].max())
    n_slots = int(df["date"].nunique())
    time_span = f"{start_dt:%Y-%m-%d %H:%M}  ->  {end_dt:%Y-%m-%d %H:%M}"
    st.caption(f"Horizon : {time_span} | créneaux = {n_slots} (3 segments/jour)")

    seg_kpi = df.groupby("segment_label", as_index=False)["y_pred"].sum().sort_values("segment_label")
    render_kpi_cards(
        [
            {
                "label": f"Segment {row.segment_label}",
                "value": _format_metric_value(row.y_pred),
                "tone": "blue",
            }
            for row in seg_kpi.itertuples(index=False)
        ],
        cols=3,
    )


def _render_network_segment_charts(df: pd.DataFrame) -> None:
    # Segment-level charts summarize demand over the selected forecast horizon.
    ts_seg = (
        df.groupby(["date_day", "segment_label"], as_index=False)["y_pred"]
        .sum()
        .sort_values(["date_day", "segment_label"])
    )
    ts_seg["day_label"] = pd.to_datetime(ts_seg["date_day"]).dt.strftime("%Y-%m-%d")
    day_order = ts_seg["day_label"].drop_duplicates().tolist()
    seg = df.groupby("segment_label", as_index=False)["y_pred"].sum().sort_values("segment_label")

    col1, col2 = st.columns(2)
    with col1:
        fig_ts = px.area(
            ts_seg,
            x="day_label",
            y="y_pred",
            color="segment_label",
            category_orders={"segment_label": SEGMENT_ORDER, "day_label": day_order},
            title="Prédictions réseau par segment (empilées)",
            labels={"y_pred": "Nb vélos loués prédits", "day_label": "Jour"},
            color_discrete_sequence=PLOTLY_DISCRETE_COLORS,
        )
        fig_ts.update_traces(
            hovertemplate="Segment=%{fullData.name}<br>Jour=%{x}<br>Nb vélos loués prédits=%{y:.0f}<extra></extra>"
        )
        style_plotly_figure(fig_ts, height=430, show_legend=True, legend_title="Segment")
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption(
            "`days=1` correspond à 3 segments glissants. "
            "Si le départ est à 16h, l'horizon couvre 16h->06h (2 jours calendaires)."
        )
    with col2:
        seg = seg.copy()
        seg["share_pct"] = (100.0 * seg["y_pred"] / seg["y_pred"].sum()).round(1)
        fig_seg = px.bar(
            seg,
            x="segment_label",
            y="y_pred",
            title="Prédictions par segment horaire",
            labels={"segment_label": "Segment", "y_pred": "Nb vélos loués prédits"},
            color="segment_label",
            color_discrete_sequence=PLOTLY_DISCRETE_COLORS,
            category_orders={"segment_label": SEGMENT_ORDER},
            text="share_pct",
        )
        fig_seg.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_line_color=CARD_NEUTRAL,
            marker_line_width=1.2,
        )
        fig_seg.add_hline(
            y=float(seg["y_pred"].mean()),
            line_dash="dot",
            line_color=CARD_NEUTRAL,
            annotation_text="moyenne des segments",
            annotation_position="top left",
        )
        style_plotly_figure(fig_seg, height=430, show_legend=False)
        st.plotly_chart(fig_seg, use_container_width=True)


def _get_network_station_selection(df: pd.DataFrame) -> tuple[list[int], dict[int, str]]:
    # Keep the station selector stable across reruns while adapting to the current top-N.
    station_totals = (
        df.groupby("start_station_id", as_index=False)["y_pred"]
        .sum()
        .sort_values("y_pred", ascending=False)
    )
    top_n_station = st.slider(
        "Nombre de stations à afficher (plus gros volumes)",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
        key="network_station_top_n",
    )
    top_station_ids = station_totals["start_station_id"].head(int(top_n_station)).astype(int).tolist()

    station_names_df = load_station_names()
    station_name_map = {
        int(row.start_station_id): str(row.start_station_name)
        for row in station_names_df.itertuples(index=False)
    }
    prev_top_n = st.session_state.get("network_station_top_n_prev")
    current_selected = st.session_state.get("network_station_multiselect", top_station_ids)
    current_selected = [int(sid) for sid in current_selected if int(sid) in top_station_ids]
    if prev_top_n != int(top_n_station) or not current_selected:
        current_selected = top_station_ids.copy()
    st.session_state["network_station_multiselect"] = current_selected
    st.session_state["network_station_top_n_prev"] = int(top_n_station)

    selected_station_ids = st.multiselect(
        "Stations affichées",
        options=top_station_ids,
        default=current_selected,
        format_func=lambda sid: f"{sid} - {station_name_map.get(int(sid), 'Nom inconnu')}",
        key="network_station_multiselect",
    )
    st.caption("Liste proposée : stations les plus volumineuses en `y_pred` sur l'horizon sélectionné.")
    return selected_station_ids, station_name_map


def _render_network_station_chart(
    df: pd.DataFrame,
    selected_station_ids: list[int],
    station_name_map: dict[int, str],
) -> None:
    # Show the stacked station-level forecast only for the current station selection.
    st.markdown("**Prédictions réseau par station (empilées)**")
    if selected_station_ids:
        ts_station = (
            df[df["start_station_id"].isin(selected_station_ids)]
            .groupby(["date", "start_station_id"], as_index=False)["y_pred"]
            .sum()
            .sort_values(["date", "start_station_id"])
        )
        ts_station["station_label"] = ts_station["start_station_id"].map(
            lambda sid: f"{int(sid)} - {station_name_map.get(int(sid), 'Nom inconnu')}"
        )
        fig_ts_station = px.area(
            ts_station,
            x="date",
            y="y_pred",
            color="station_label",
            title="Prédictions réseau par station (empilées)",
            labels={"y_pred": "Nb vélos loués prédits", "date": "Date", "station_label": "Station"},
            color_discrete_sequence=PLOTLY_DISCRETE_COLORS,
        )
        style_plotly_figure(fig_ts_station, height=450, show_legend=True, legend_title="Station")
        st.plotly_chart(fig_ts_station, use_container_width=True)
    else:
        st.info("Sélectionnez au moins une station pour afficher le graphique réseau par station.")


def _render_network_heatmap(df: pd.DataFrame) -> None:
    # Compact view of demand intensity by day and segment.
    heat = df.groupby(["date_day", "segment_label"], as_index=False)["y_pred"].sum()
    heat_pivot = (
        heat.pivot(index="segment_label", columns="date_day", values="y_pred")
        .reindex(SEGMENT_ORDER)
        .fillna(0.0)
    )
    fig_heat = px.imshow(
        heat_pivot,
        aspect="auto",
        title="Heatmap réseau (jour x segment)",
        labels={"x": "Jour", "y": "Segment", "color": "Nb vélos loués prédits"},
        color_continuous_scale=PLOTLY_CONTINUOUS_SCALE,
    )
    fig_heat.update_xaxes(side="bottom", tickangle=-40, showgrid=False)
    fig_heat.update_yaxes(showgrid=False)
    style_plotly_figure(fig_heat, height=450, show_legend=False)
    st.plotly_chart(fig_heat, use_container_width=True)


def _render_network_map(df: pd.DataFrame) -> None:
    # Optional geographic view using station coordinates from the merged dataset.
    show_map = st.checkbox("Afficher la carte des stations (latitude/longitude)", value=True)
    if not show_map:
        return

    with st.spinner("Chargement des coordonnées des stations..."):
        coords = load_station_coordinates()
    if coords.empty:
        st.warning("Coordonnées introuvables. Vérifiez `data/merged/all_merged.csv`.")
        return

    station_agg = df.groupby("start_station_id", as_index=False)["y_pred"].sum()
    station_map = station_agg.merge(coords, on="start_station_id", how="left").dropna(
        subset=["start_lat", "start_lng"]
    )
    if station_map.empty:
        st.warning("Aucune station géolocalisée pour la vue courante.")
        return

    station_map["start_station_id"] = station_map["start_station_id"].astype(str)
    fig_map = px.scatter_mapbox(
        station_map,
        lat="start_lat",
        lon="start_lng",
        size="y_pred",
        color="y_pred",
        hover_name="start_station_id",
        hover_data={"y_pred": ":.1f"},
        title="Carte des stations (taille/couleur = volume prédit)",
        zoom=10,
        height=520,
        mapbox_style="open-street-map",
        color_continuous_scale=PLOTLY_CONTINUOUS_SCALE,
    )
    style_plotly_figure(fig_map, height=520, show_legend=False)
    st.plotly_chart(fig_map, use_container_width=True)


def render_network_dashboard(df: pd.DataFrame) -> None:
    # Compose the full network dashboard from smaller visual sections.
    st.subheader("Vue réseau")
    _render_network_overview(df)
    _render_network_segment_charts(df)
    selected_station_ids, station_name_map = _get_network_station_selection(df)
    _render_network_station_chart(df, selected_station_ids, station_name_map)
    _render_network_heatmap(df)
    _render_network_map(df)


def render_station_dashboard(df: pd.DataFrame, station_id: int) -> None:
    # Focused station view with totals, segment breakdowns, and forecast vs actual lines.
    station_df = df.loc[df["start_station_id"] == int(station_id)].copy()
    if station_df.empty:
        st.warning(f"Aucune ligne trouvée pour la station {station_id}.")
        return

    st.subheader(f"Vue station {station_id}")
    render_kpi_cards(
        [
            {
                "label": "Lignes",
                "value": _format_metric_value(len(station_df)),
                "tone": "neutral",
            },
            {
                "label": "Prédictions totales",
                "value": _format_metric_value(station_df["y_pred"].sum()),
                "tone": "green",
            },
            {
                "label": "Moyenne / segment",
                "value": _format_metric_value(station_df["y_pred"].mean(), decimals=2),
                "tone": "blue",
            },
            {
                "label": "Max segment",
                "value": _format_metric_value(station_df["y_pred"].max(), decimals=2),
                "tone": "amber",
            },
        ],
        cols=4,
    )

    seg_station_kpi = (
        station_df.groupby("segment_label", as_index=False)["y_pred"]
        .sum()
        .sort_values("segment_label")
    )
    render_kpi_cards(
        [
            {
                "label": f"Segment {row.segment_label}",
                "value": _format_metric_value(row.y_pred),
                "tone": "blue",
            }
            for row in seg_station_kpi.itertuples(index=False)
        ],
        cols=3,
    )

    series_cols = ["y_pred"]
    if "y_station" in station_df.columns and station_df["y_station"].notna().any():
        series_cols.append("y_station")

    station_plot_df = station_df[["date", "segment_label"] + series_cols].melt(
        id_vars=["date", "segment_label"],
        value_vars=series_cols,
        var_name="series",
        value_name="value",
    )
    station_plot_df["series"] = station_plot_df["series"].replace({"y_pred": "Prévision", "y_station": "Réel"})
    st.caption("Chaque point représente un segment complet : 00h-06h, 06h-16h, 16h-00h.")
    fig_station = px.line(
        station_plot_df,
        x="date",
        y="value",
        color="series",
        custom_data=["segment_label"],
        title=f"Station {station_id} - prédiction vs réel",
        labels={"value": "Nb vélos loués", "date": "Date"},
        color_discrete_sequence=[CARD_BLUE, CARD_AMBER],
    )
    fig_station.update_traces(
        line=dict(width=3),
        mode="lines+markers",
        marker=dict(size=6),
        hovertemplate=(
            "Série=%{fullData.name}<br>"
            "Date=%{x|%Y-%m-%d %H:%M}<br>"
            "Segment=%{customdata[0]}<br>"
            "Nb vélos loués=%{y:.2f}<extra></extra>"
        ),
    )
    style_plotly_figure(fig_station, height=430, show_legend=True, legend_title="Série")
    st.plotly_chart(fig_station, use_container_width=True)

    seg_station = (
        station_df.groupby("segment_label", as_index=False)
        .agg(
            total_pred=("y_pred", "sum"),
            moyenne_pred=("y_pred", "mean"),
        )
        .sort_values("segment_label")
    )
    seg_station["share_pct"] = (100.0 * seg_station["total_pred"] / seg_station["total_pred"].sum()).round(1)
    fig_station_seg = px.bar(
        seg_station,
        x="segment_label",
        y="total_pred",
        color="segment_label",
        category_orders={"segment_label": SEGMENT_ORDER},
        title=f"Station {station_id} - total prédit par segment",
        labels={"segment_label": "Segment", "total_pred": "Nb vélos loués prédits"},
        color_discrete_sequence=PLOTLY_DISCRETE_COLORS,
        text="share_pct",
    )
    fig_station_seg.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_color=CARD_NEUTRAL,
        marker_line_width=1.0,
        hovertemplate=(
            "Segment=%{x}<br>"
            "Nb vélos loués prédits=%{y:.1f}<br>"
            "Part du total station=%{text:.1f}%<extra></extra>"
        ),
    )
    style_plotly_figure(fig_station_seg, height=420, show_legend=False)
    st.plotly_chart(fig_station_seg, use_container_width=True)

    daily_station_segment = (
        station_df.groupby(["date_day", "segment_label"], as_index=False)["y_pred"]
        .sum()
        .sort_values(["date_day", "segment_label"])
    )
    daily_station_segment["share_day_pct"] = (
        100.0
        * daily_station_segment["y_pred"]
        / daily_station_segment.groupby("date_day")["y_pred"].transform("sum")
    ).round(1)
    fig_station_daily = px.bar(
        daily_station_segment,
        x="date_day",
        y="y_pred",
        color="segment_label",
        category_orders={"segment_label": SEGMENT_ORDER},
        barmode="group",
        title=f"Station {station_id} - détail quotidien par segment",
        labels={"date_day": "Jour", "y_pred": "Nb vélos loués prédits", "segment_label": "Segment"},
        color_discrete_sequence=PLOTLY_DISCRETE_COLORS,
        text="share_day_pct",
    )
    fig_station_daily.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_color=CARD_NEUTRAL,
        marker_line_width=0.8,
        hovertemplate=(
            "Jour=%{x}<br>"
            "Segment=%{fullData.name}<br>"
            "Nb vélos loués prédits=%{y:.1f}<br>"
            "Part du jour=%{text:.1f}%<extra></extra>"
        ),
    )
    style_plotly_figure(fig_station_daily, height=420, show_legend=True, legend_title="Segment")
    st.plotly_chart(fig_station_daily, use_container_width=True)

    pivot_station = station_df.pivot_table(
        index="date_day",
        columns="segment_label",
        values="y_pred",
        aggfunc="sum",
    ).reset_index()
    st.dataframe(pivot_station, use_container_width=True)

    st.dataframe(
        station_df[["start_station_id", "date", "segment_id", "segment_label", "y_pred"] + (["y_station"] if "y_station" in station_df.columns else [])],
        use_container_width=True,
    )


def render_rebalancing_results(result: dict) -> None:
    # Render the API response as KPIs, operational tables, and a route map when available.
    summary = result.get("summary", {}) or {}
    files = result.get("files", {}) or {}
    st.subheader("Résumé du rééquilibrage")
    render_kpi_cards(
        [
            {
                "label": "Alertes",
                "value": _format_metric_value(summary.get("alerts_total", 0)),
                "tone": "amber",
            },
            {
                "label": "Critiques",
                "value": _format_metric_value(summary.get("alerts_critical", 0)),
                "tone": "red",
            },
            {
                "label": "Avertissements",
                "value": _format_metric_value(summary.get("alerts_warning", 0)),
                "tone": "amber",
            },
            {
                "label": "Transferts",
                "value": _format_metric_value(summary.get("transfers_total", 0)),
                "tone": "blue",
            },
            {
                "label": "Vélos à déplacer",
                "value": _format_metric_value(summary.get("bikes_to_move_total", 0)),
                "tone": "green",
            },
            {
                "label": "Distance (km)",
                "value": _format_metric_value(summary.get("distance_total_km", 0), decimals=1),
                "tone": "neutral",
            },
        ],
        cols=3,
    )

    st.caption(
        f"Horizon en segments : {summary.get('horizon_segments', '-')}, "
        f"de {summary.get('horizon_start', '-')}"
        f" à {summary.get('horizon_end', '-')}"
    )
    forecast_used = files.get("forecast_file")
    if forecast_used:
        st.caption(f"Fichier de prévision utilisé : {forecast_used}")
    st.caption("Horizon calculé à partir des premiers segments chronologiques du fichier de prévision sélectionné.")
    raw_t = summary.get("transfers_total_raw")
    sel_t = summary.get("transfers_total")
    max_t = summary.get("max_transfers")
    if raw_t is not None and sel_t is not None:
        st.caption(
            f"Actions proposées : {sel_t} sur {raw_t} actions possibles "
            f"(max_transfers={max_t if max_t is not None else 'illimité'})."
        )

    with st.expander("Fichiers générés / utilisés"):
        st.json(
            {
                "generated_inputs": result.get("generated_inputs", {}),
                "files": files,
            }
        )

    alerts_df = pd.DataFrame(result.get("alerts_rows", []))
    transfer_df = pd.DataFrame(result.get("transfer_rows", []))
    route_df = pd.DataFrame(result.get("route_rows", []))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Stations en alerte**")
        if alerts_df.empty:
            st.info("Aucune alerte.")
        else:
            display_cols = [
                c
                for c in [
                    "start_station_id",
                    "severity",
                    "bikes_available",
                    "pred_out_horizon",
                    "pred_net_out_horizon",
                    "stock_proj",
                    "deficit_qty",
                    "surplus_qty",
                    "capacity_bikes",
                    "min_buffer",
                ]
                if c in alerts_df.columns
            ]
            st.dataframe(style_alerts_table(alerts_df[display_cols]), use_container_width=True)

    with col2:
        st.markdown("**Plan de transferts**")
        if transfer_df.empty:
            st.info("Aucun transfert nécessaire.")
        else:
            display_cols = [
                c
                for c in [
                    "from_station_id",
                    "to_station_id",
                    "qty_bikes",
                    "distance_km",
                    "priority",
                ]
                if c in transfer_df.columns
            ]
            st.dataframe(style_transfer_table(transfer_df[display_cols]), use_container_width=True)

    st.markdown("**Ordre de tournée conseillé**")
    if route_df.empty:
        st.info("Aucune tournée à afficher (aucun transfert).")
    else:
        st.dataframe(style_route_table(route_df), use_container_width=True)

    has_geo = {
        "from_lat",
        "from_lng",
        "to_lat",
        "to_lng",
    }.issubset(set(transfer_df.columns))
    if has_geo and not transfer_df.empty:
        points = []
        for idx, row in transfer_df.iterrows():
            if pd.isna(row["from_lat"]) or pd.isna(row["from_lng"]) or pd.isna(row["to_lat"]) or pd.isna(row["to_lng"]):
                continue
            points.append(
                {
                    "line_id": str(idx),
                    "lat": row["from_lat"],
                    "lon": row["from_lng"],
                    "priority": row.get("priority", "unknown"),
                    "hover": f"{int(row['from_station_id'])} -> {int(row['to_station_id'])} | {int(row['qty_bikes'])} vélos",
                }
            )
            points.append(
                {
                    "line_id": str(idx),
                    "lat": row["to_lat"],
                    "lon": row["to_lng"],
                    "priority": row.get("priority", "unknown"),
                    "hover": f"{int(row['from_station_id'])} -> {int(row['to_station_id'])} | {int(row['qty_bikes'])} vélos",
                }
            )
        line_df = pd.DataFrame(points)
        if not line_df.empty:
            fig = px.line_mapbox(
                line_df,
                lat="lat",
                lon="lon",
                line_group="line_id",
                color="priority",
                hover_name="hover",
                title="Transferts inter-stations (trajets proposés)",
                mapbox_style="carto-positron",
                zoom=10,
                height=520,
                color_discrete_sequence=[CARD_RED, CARD_AMBER, CARD_GREEN],
            )
            style_plotly_figure(fig, height=520, show_legend=True, legend_title="Priorité")
            st.plotly_chart(fig, use_container_width=True)


def get_processed_file_choices() -> tuple[list[str], list[str]]:
    # Build the file pickers from data/processed, prioritizing prediction-like outputs.
    all_csv_paths = [p for p in PROCESSED_DIR.rglob("*.csv")]
    all_csv_files = sorted(
        [p.relative_to(PROCESSED_DIR).as_posix() for p in all_csv_paths],
        key=lambda rel: (PROCESSED_DIR / rel).stat().st_mtime if (PROCESSED_DIR / rel).exists() else 0,
        reverse=True,
    )
    prediction_like_files = sorted(
        [
            rel
            for rel in all_csv_files
            if (rel.split("/")[-1].startswith("predictions_recursive") or rel.split("/")[-1].startswith("forecast"))
        ],
        key=lambda rel: (PROCESSED_DIR / rel).stat().st_mtime if (PROCESSED_DIR / rel).exists() else 0,
        reverse=True,
    )
    return all_csv_files, (prediction_like_files or all_csv_files)


def render_predict_tab(api_base_url: str) -> None:
    # Network forecast form: validate inputs, call the API, then load the new CSV in session.
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            features_filename = st.text_input("Fichier des variables d'entrée", value="features_3segments.csv")
            artifact_prefix = st.text_input("Modèle utilisé", value="catboost_station_3segments_fullfit_v1")
            days = st.number_input("Horizon de prévision (jours)", min_value=1, max_value=31, value=7, step=1)
        with col2:
            start_date_input = st.text_input(
                "date de début de prévision",
                value="2026-03-10T06:00:00",
                placeholder="YYYY-MM-DDTHH:MM:SS (ex. : 2026-03-10T06:00:00)",
                help=(
                    "Formats acceptés : YYYY-MM-DD ou YYYY-MM-DDTHH:MM:SS."
                ),
            )
        st.caption(
            "Format attendu pour `start_date` : `YYYY-MM-DD` ou `YYYY-MM-DDTHH:MM:SS` "
            "(ex. : `2026-03-10T06:00:00`)."
        )
        st.caption(
            "La date est alignée sur le prochain début de segment (00h, 06h, 16h). "
            "Ex. : `2026-03-10T18:30:00` -> démarrage à `2026-03-11 00:00:00`."
        )
        train_date_max, _ = get_training_date_max(
            artifact_prefix=artifact_prefix,
            features_filename=features_filename,
        )
        if train_date_max is not None:
            max_allowed = train_date_max + pd.Timedelta(days=7)
            st.caption(
                "Contrainte dynamique : "
                f"`start_date <= {max_allowed:%Y-%m-%d %H:%M:%S}` "
                f"(date max du jeu d'entraînement : {train_date_max:%Y-%m-%d %H:%M:%S})."
            )
        else:
            st.caption("Contrainte : `start_date` ne peut pas dépasser `date_max(features) + 7 jours`.")
        submit_predict = st.form_submit_button("Générer la prévision réseau")

    if submit_predict:
        is_valid_date, date_error = validate_start_date_input(start_date_input)
        if not is_valid_date:
            st.error(date_error)
            st.stop()

        if start_date_input and train_date_max is not None:
            start_ts_aligned = align_to_segment_start_ui(pd.to_datetime(start_date_input))
            max_allowed = pd.Timestamp(train_date_max) + pd.Timedelta(days=7)
            if start_ts_aligned > max_allowed:
                st.error(
                    "La valeur de `start_date` est incompatible avec le modèle sélectionné. "
                    f"Date alignée reçue : {start_ts_aligned:%Y-%m-%d %H:%M:%S} ; "
                    f"maximum autorisé : {max_allowed:%Y-%m-%d %H:%M:%S} "
                    "(date max d'entraînement + 7 jours)."
                )
                st.stop()
        elif start_date_input and train_date_max is None:
            st.error(
                "Impossible de valider la compatibilité de start_date: "
                "date_max d'entraînement indisponible pour ce modèle."
            )
            st.stop()

        payload: dict = {
            "features_filename": features_filename,
            "artifact_prefix": artifact_prefix,
            "days": int(days),
            "start_date": start_date_input,
        }
        data = run_api_action(
            base_url=api_base_url,
            endpoint="/predict/recursive",
            payload=payload,
            spinner_text="Appel API en cours...",
            success_text="Prédiction terminée",
        )
        if data:
            st.session_state["last_prediction_filename"] = data.get("output_filename")
            load_processed_df_to_state(data.get("output_filename"), "network_df")
            st.rerun()


def render_station_tab(api_base_url: str, station_candidate_files: list[str]) -> None:
    # Station export form: start from an existing network forecast and extract one station.
    st.caption(
        "Affiche les informations d'une station spécifique à partir d'un fichier de prédictions réseau déjà généré."
    )
    preferred_idx = 0
    preferred_filename = st.session_state.get("last_prediction_filename")
    if preferred_filename in station_candidate_files:
        preferred_idx = station_candidate_files.index(preferred_filename)

    station_names_df = load_station_names()
    station_name_map = {
        int(row.start_station_id): str(row.start_station_name)
        for row in station_names_df.itertuples(index=False)
    }

    with st.form("export_form"):
        selected_input_filename = st.selectbox(
            "Fichier réseau disponible",
            options=station_candidate_files if station_candidate_files else [""],
            index=preferred_idx if station_candidate_files else 0,
            help="Choisissez un fichier déjà présent dans `data/processed`.",
        )
        station_ids_for_file = get_station_ids_from_prediction_file(selected_input_filename)
        if station_ids_for_file:
            station_id_export = st.selectbox(
                "Station (ID + nom, recherche possible)",
                options=station_ids_for_file,
                index=0,
                format_func=lambda sid: f"{sid} - {station_name_map.get(int(sid), 'Nom inconnu')}",
                help="Vous pouvez saisir du texte dans la liste pour rechercher une station.",
            )
        else:
            st.warning("Impossible de charger les stations depuis ce fichier.")
            station_id_export = None
        submit_export = st.form_submit_button("Charger la station", disabled=station_id_export is None)

    if submit_export:
        payload = {
            "input_filename": selected_input_filename,
            "station_id": int(station_id_export),
        }
        data = run_api_action(
            base_url=api_base_url,
            endpoint="/station/export",
            payload=payload,
            spinner_text="Appel API en cours...",
            success_text="Données station chargées",
        )
        if data:
            load_processed_df_to_state(data.get("output_filename"), "station_df")

    if isinstance(st.session_state.get("station_df"), pd.DataFrame) and not st.session_state["station_df"].empty:
        st.divider()
        st.caption("Détail de la station chargé")
        sid = int(st.session_state["station_df"]["start_station_id"].iloc[0])
        render_station_dashboard(st.session_state["station_df"], sid)


def render_rebalancing_tab(api_base_url: str, rebalancing_candidate_files: list[str]) -> None:
    # Rebalancing form: compute alerts, transfers, and route recommendations from a forecast file.
    st.caption(
        "Construit un plan de transferts inter-stations (alertes de pénurie/surplus + trajet conseillé)."
    )
    with st.form("rebalancing_form"):
        forecast_filename_rebal = st.selectbox(
            "Fichier de prévision réseau",
            options=rebalancing_candidate_files if rebalancing_candidate_files else [""],
            index=0 if rebalancing_candidate_files else 0,
            help="Fichier nommé `forecast` ou `predictions_recursive`, déjà généré dans `data/processed`.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            capacity_filename = st.text_input("Fichier des capacités des stations", value="station_capacity.csv")
            horizon_segments = st.number_input("Horizon (segments)", min_value=1, max_value=12, value=3, step=1)
        with col_b:
            realtime_filename = st.text_input("Fichier d'état en temps réel", value="realtime/station_status_realtime.csv")
        max_transfers = st.number_input(
            "Nombre maximal d'actions terrain (transferts)",
            min_value=1,
            max_value=200,
            value=20,
            step=1,
            help="Le plan final sera limité aux transferts prioritaires les plus pertinents.",
        )
        net_out_ratio = st.slider(
            "Part de sorties nettes (après retours attendus)",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help=(
                "0.35 signifie que 35% de la demande prédite est considérée comme consommation nette "
                "de stock sur l'horizon (le reste est compensé par des retours)."
            ),
        )

        save_outputs = st.checkbox("Sauvegarder les fichiers `alerts`/`plan`/`route`", value=True)
        submit_rebalancing = st.form_submit_button("Calculer le plan de rééquilibrage")

    if submit_rebalancing:
        # Clear the previous result first to avoid showing stale tables on a failed rerun.
        st.session_state["rebalancing_result"] = None
        payload = {
            "forecast_filename": forecast_filename_rebal,
            "capacity_filename": capacity_filename,
            "realtime_filename": realtime_filename,
            "horizon_segments": int(horizon_segments),
            "net_out_ratio": float(net_out_ratio),
            "max_transfers": int(max_transfers),
            "save_outputs": save_outputs,
        }
        data = run_api_action(
            base_url=api_base_url,
            endpoint="/ops/rebalancing/plan",
            payload=payload,
            spinner_text="Calcul du plan en cours...",
            success_text="Plan de rééquilibrage calculé",
        )
        if data:
            st.session_state["rebalancing_result"] = data

    if isinstance(st.session_state.get("rebalancing_result"), dict):
        render_rebalancing_results(st.session_state["rebalancing_result"])


def render_loaded_dashboard() -> None:
    # The bottom section shows the latest loaded network view, or falls back to station-only feedback.
    st.divider()
    network_df = st.session_state.get("network_df")
    station_df = st.session_state.get("station_df")

    if isinstance(network_df, pd.DataFrame) and not network_df.empty:
        render_network_dashboard(network_df)
    elif isinstance(station_df, pd.DataFrame) and not station_df.empty:
        st.info("Vue réseau non chargée. Chargez un fichier ou une prédiction réseau pour afficher la vue complète.")
    else:
        st.info("Chargez une prédiction réseau via l'API ou via un fichier local pour afficher le tableau de bord.")


def main() -> None:
    # Main app entry point: initialize UI state, render the tabs, then display loaded results.
    st.set_page_config(page_title="Prévisions de demande vélo", layout="wide")
    inject_metric_css()
    st.title("Tableau de bord des prévisions de demande vélo")
    st.caption("Vue globale réseau + station, structurée pour lecture rapide")

    init_session_state(
        {
            "network_df": None,
            "station_df": None,
            "last_response": None,
            "rebalancing_result": None,
            "last_prediction_filename": None,
        }
    )

    with st.sidebar:
        st.header("Connexion API")
        api_base_url = st.text_input("URL de base", value="http://api:8000")
        st.caption("Exemple : http://api:8000")

    station_candidate_files, rebalancing_candidate_files = get_processed_file_choices()
    tab_predict, tab_station_detail, tab_rebalancing = st.tabs(
        ["Générer une prévision réseau (API)", "Détail par station (API)", "Rééquilibrage (API)"]
    )

    with tab_predict:
        render_predict_tab(api_base_url)
    with tab_station_detail:
        render_station_tab(api_base_url, station_candidate_files)
    with tab_rebalancing:
        render_rebalancing_tab(api_base_url, rebalancing_candidate_files)

    render_loaded_dashboard()


main()
