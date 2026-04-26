# app/pages/results_page.py
"""Results Page — pipeline outputs with aesthetic charts."""

import os
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import INSIGHTS_DIR, ANNOTATIONS_DIR, PROCESSED_DIR
from app.utils import (
    page_header, render_pipeline, nav_button, metric_card,
    ACCENT, TEXT_PRIMARY, TEXT_MUTED, BG_CARD, BG_DARK,
)

TEAM_COLORS = {0: "#dc2626", 1: "#3b82f6", -1: "#52525b"}
TEAM_NAMES  = {0: "Team A", 1: "Team B", -1: "Unassigned"}

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
    title_font=dict(size=13, color=TEXT_PRIMARY, family="Inter, sans-serif"),
    margin=dict(t=44, b=28, l=36, r=20),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.05)",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=10),
    ),
)


def _layout(**overrides):
    d = dict(_LAYOUT)
    d.update(overrides)
    return d


def _load_csv(name):
    for d in [INSIGHTS_DIR, ANNOTATIONS_DIR]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def _load_summary():
    p = os.path.join(INSIGHTS_DIR, "pipeline_summary.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _find_tracked_video():
    v = st.session_state.get("tracked_video")
    if v and os.path.exists(v):
        return v
    if os.path.exists(PROCESSED_DIR):
        for f in sorted(os.listdir(PROCESSED_DIR)):
            if "tracked" in f and f.endswith(".mp4"):
                return os.path.join(PROCESSED_DIR, f)
    return None


def render():
    page_header("Results", "Possession, player stats, speed analysis, and exports.")
    render_pipeline(done_up_to=3)

    player_df     = _load_csv("player_summary.csv")
    poss_df       = _load_csv("possession_summary.csv")
    track_df      = _load_csv("tracking_enriched.csv")
    summary       = _load_summary()
    tracked_video = _find_tracked_video()

    if player_df is None and poss_df is None and track_df is None:
        st.warning("No results found. Run the analysis pipeline first.")
        _, r = st.columns([3, 1])
        with r:
            nav_button("Go to Analysis", "Analysis")
        return

    # Summary metrics
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Video", summary.get("video", "—")), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Frames", f"{summary.get('total_frames', 0):,}"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Resolution", summary.get("resolution", "—")), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Replays Skipped", str(summary.get("replays_detected", 0))), unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if tracked_video:
        with st.expander("▶  Tracked Video Preview"):
            st.video(tracked_video)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    tab_poss, tab_player, tab_speed, tab_track, tab_dl = st.tabs([
        "Possession", "Players", "Speed", "Tracking", "Downloads",
    ])

    # POSSESSION
    with tab_poss:
        if poss_df is not None and not poss_df.empty:
            poss_df = poss_df.copy()
            poss_df["team"] = poss_df["team_id"].map(TEAM_NAMES)
            colors = [TEAM_COLORS.get(int(t), "#6b7280") for t in poss_df["team_id"]]

            chart_col, data_col = st.columns([3, 1])
            with chart_col:
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=poss_df["team"],
                    values=poss_df["possession_pct"],
                    hole=0.62,
                    marker=dict(colors=colors, line=dict(color=BG_DARK, width=3)),
                    textinfo="label+percent",
                    textfont=dict(size=12, color=TEXT_PRIMARY),
                    hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
                ))
                fig.update_layout(
                    **_layout(height=360, title="Ball Possession"),
                    annotations=[dict(text="Possession", x=0.5, y=0.5,
                                      font=dict(size=12, color=TEXT_MUTED), showarrow=False)],
                )
                st.plotly_chart(fig, use_container_width=True)

            with data_col:
                st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
                for _, row in poss_df.iterrows():
                    st.markdown(metric_card(row["team"], f"{row['possession_pct']:.1f}%"), unsafe_allow_html=True)
                    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

            if player_df is not None and "poss_pct" in player_df.columns:
                top = player_df.nlargest(10, "poss_pct").copy()
                if not top.empty:
                    top["team"] = top["team_id"].map(TEAM_NAMES)
                    top = top.sort_values("poss_pct", ascending=True)
                    fig2 = go.Figure()
                    for tid, tname in TEAM_NAMES.items():
                        sub = top[top["team_id"] == tid]
                        if sub.empty:
                            continue
                        fig2.add_trace(go.Bar(
                            y=sub["object_id"].astype(str), x=sub["poss_pct"],
                            name=tname, orientation="h",
                            marker=dict(color=TEAM_COLORS[tid], opacity=0.85, line=dict(width=0)),
                            hovertemplate="Player %{y}<br>%{x:.1f}%<extra></extra>",
                        ))
                    fig2.update_layout(**_layout(height=340, title="Top 10 Players by Possession"),
                                       barmode="stack", xaxis_title="Possession %", yaxis_title="Player ID")
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No possession data available.")

    # PLAYERS
    with tab_player:
        if player_df is not None and not player_df.empty:
            df = player_df.copy()
            if "team_id" in df.columns:
                df["team"] = df["team_id"].map(TEAM_NAMES)
            if "class_id" in df.columns:
                df["role"] = df["class_id"].map({0: "Ball", 1: "GK", 2: "Player", 3: "Referee"})

            f1, f2 = st.columns(2)
            with f1:
                teams = st.multiselect("Team", df["team"].unique().tolist() if "team" in df.columns else [],
                                       default=df["team"].unique().tolist() if "team" in df.columns else [], key="pl_team")
            with f2:
                roles = st.multiselect("Role", df["role"].unique().tolist() if "role" in df.columns else [],
                                       default=df["role"].unique().tolist() if "role" in df.columns else [], key="pl_role")

            mask = pd.Series(True, index=df.index)
            if teams and "team" in df.columns:
                mask &= df["team"].isin(teams)
            if roles and "role" in df.columns:
                mask &= df["role"].isin(roles)
            filtered = df[mask]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(metric_card("Players", str(len(filtered))), unsafe_allow_html=True)
            with m2:
                if "top_speed_km_h" in filtered.columns and len(filtered) > 0:
                    st.markdown(metric_card("Max Speed", f"{filtered['top_speed_km_h'].max():.1f} km/h"), unsafe_allow_html=True)
            with m3:
                if "avg_speed_km_h" in filtered.columns and len(filtered) > 0:
                    st.markdown(metric_card("Avg Speed", f"{filtered['avg_speed_km_h'].mean():.1f} km/h"), unsafe_allow_html=True)
            with m4:
                if "total_frames" in filtered.columns and len(filtered) > 0:
                    st.markdown(metric_card("Avg Visibility", f"{filtered['total_frames'].mean():.0f} frames"), unsafe_allow_html=True)

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            st.dataframe(filtered, use_container_width=True, hide_index=True, height=380)
        else:
            st.info("No player data available.")

    # SPEED
    with tab_speed:
        if player_df is not None and "top_speed_km_h" in player_df.columns:
            sdf = player_df.copy()
            sdf["team"] = sdf["team_id"].map(TEAM_NAMES)

            col_a, col_b = st.columns(2)
            with col_a:
                fig3 = go.Figure()
                for tid, tname in TEAM_NAMES.items():
                    sub = sdf[sdf["team_id"] == tid]
                    if sub.empty:
                        continue
                    fig3.add_trace(go.Violin(
                        y=sub["top_speed_km_h"], name=tname,
                        box_visible=True, meanline_visible=True,
                        fillcolor=TEAM_COLORS[tid], opacity=0.7,
                        line_color=TEAM_COLORS[tid],
                        hovertemplate="%{y:.1f} km/h<extra></extra>",
                    ))
                fig3.update_layout(**_layout(height=360, title="Speed Distribution by Team"))
                st.plotly_chart(fig3, use_container_width=True)

            with col_b:
                top10 = sdf.nlargest(10, "top_speed_km_h").sort_values("top_speed_km_h")
                fig4 = go.Figure()
                for tid, tname in TEAM_NAMES.items():
                    sub = top10[top10["team_id"] == tid]
                    if sub.empty:
                        continue
                    fig4.add_trace(go.Bar(
                        y=sub["object_id"].astype(str), x=sub["top_speed_km_h"],
                        name=tname, orientation="h",
                        marker=dict(color=TEAM_COLORS[tid], opacity=0.85, line=dict(width=0)),
                        text=sub["top_speed_km_h"].round(1).astype(str) + " km/h",
                        textposition="outside",
                        textfont=dict(size=10, color=TEXT_MUTED),
                        hovertemplate="Player %{y}<br>%{x:.1f} km/h<extra></extra>",
                    ))
                fig4.update_layout(**_layout(height=360, title="Fastest Players"),
                                   barmode="stack", xaxis_title="Top Speed (km/h)", yaxis_title="Player ID")
                st.plotly_chart(fig4, use_container_width=True)

            if "avg_speed_km_h" in sdf.columns:
                fig5 = go.Figure()
                for tid, tname in TEAM_NAMES.items():
                    sub = sdf[sdf["team_id"] == tid]
                    if sub.empty:
                        continue
                    fig5.add_trace(go.Scatter(
                        x=sub["avg_speed_km_h"], y=sub["top_speed_km_h"],
                        mode="markers", name=tname,
                        marker=dict(color=TEAM_COLORS[tid], size=9, opacity=0.8,
                                    line=dict(width=1, color="rgba(255,255,255,0.1)")),
                        text=sub["object_id"].astype(str),
                        hovertemplate="Player %{text}<br>Avg: %{x:.1f} km/h<br>Top: %{y:.1f} km/h<extra></extra>",
                    ))
                fig5.update_layout(**_layout(height=360, title="Speed Profile: Average vs Peak"),
                                   xaxis_title="Average Speed (km/h)", yaxis_title="Top Speed (km/h)")
                st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No speed data available.")

    # TRACKING
    with tab_track:
        if track_df is not None and not track_df.empty:
            st.caption(f"{len(track_df):,} rows — showing first 500")
            st.dataframe(track_df.head(500), use_container_width=True, hide_index=True, height=360)

            if "frame_id" in track_df.columns and "object_id" in track_df.columns:
                per_frame = track_df.groupby("frame_id")["object_id"].nunique().reset_index()
                per_frame.columns = ["frame_id", "objects"]
                fig6 = go.Figure()
                fig6.add_trace(go.Scatter(
                    x=per_frame["frame_id"], y=per_frame["objects"],
                    mode="lines", fill="tozeroy",
                    fillcolor="rgba(220,38,38,0.08)",
                    line=dict(color=ACCENT, width=1.5),
                    hovertemplate="Frame %{x}<br>%{y} objects<extra></extra>",
                ))
                fig6.update_layout(**_layout(height=300, title="Active Tracked Objects Over Time"),
                                   xaxis_title="Frame", yaxis_title="Objects")
                st.plotly_chart(fig6, use_container_width=True)

            if "cx" in track_df.columns and "cy" in track_df.columns:
                fig7 = go.Figure(go.Histogram2dContour(
                    x=track_df["cx"], y=track_df["cy"],
                    colorscale=[[0, "rgba(0,0,0,0)"], [0.3, "rgba(220,38,38,0.2)"],
                                [0.7, "rgba(220,38,38,0.6)"], [1, "#dc2626"]],
                    reversescale=False, showscale=False, ncontours=20,
                    contours=dict(showlines=False),
                ))
                fig7.update_layout(**_layout(height=380, title="Player Position Heatmap"),
                                   xaxis_title="X", yaxis_title="Y", yaxis_autorange="reversed")
                st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("No tracking data available.")

    # DOWNLOADS
    with tab_dl:
        downloads = [
            ("Player Summary", "player_summary.csv", player_df, "Per-player stats: speed, possession, team."),
            ("Possession", "possession_summary.csv", poss_df, "Team-level possession percentages."),
            ("Tracking Data", "tracking_enriched.csv", track_df, "Frame-by-frame tracking with velocity."),
        ]
        cols = st.columns(3)
        for col, (title, fname, df, desc) in zip(cols, downloads):
            with col:
                st.markdown(f"""
                <div class="card" style="text-align:center; margin-bottom:0.8rem;">
                    <div class="card-label">CSV</div>
                    <div class="card-heading">{title}</div>
                    <div class="card-body">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                if df is not None:
                    st.download_button(f"Download {fname}", data=df.to_csv(index=False),
                                       file_name=fname, mime="text/csv",
                                       use_container_width=True, key=f"dl_{fname}")
                else:
                    st.button("Not available", disabled=True, use_container_width=True, key=f"na_{fname}")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        if tracked_video:
            with open(tracked_video, "rb") as vf:
                st.download_button("⬇  Download Tracked Video (MP4)", data=vf.read(),
                                   file_name=os.path.basename(tracked_video),
                                   mime="video/mp4", use_container_width=True, type="primary")
        if summary:
            st.download_button("⬇  Download Pipeline Summary (JSON)",
                               data=json.dumps(summary, indent=2),
                               file_name="pipeline_summary.json",
                               mime="application/json", use_container_width=True)

    st.markdown("---")
    left, center, right = st.columns(3)
    with left:
        nav_button("← Back to Analysis", "Analysis", key="res_back")
    with center:
        if st.button("Start New Analysis", use_container_width=True):
            for k in ["uploaded_video", "uploaded_video_name", "processed_video",
                      "analysis_done", "analysis_results", "tracked_video"]:
                st.session_state.pop(k, None)
            st.session_state.page = "Upload"
            st.rerun()
    with right:
        nav_button("Home", "Home", key="res_home")
