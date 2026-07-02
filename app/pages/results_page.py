# app/pages/results_page.py
"""Results Page — pipeline outputs with aesthetic charts."""

import os
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import INSIGHTS_DIR, PROCESSED_DIR, get_game_list
from app.utils import (
    page_header, render_pipeline, nav_button, metric_card,
    TEXT_PRIMARY, TEXT_MUTED, BG_CARD, BG_DARK,
)

TEAM_COLORS = {0: "#dc2626", 1: "#3b82f6", -1: "#52525b"}
TEAM_NAMES  = {0: "Team A", 1: "Team B", -1: "Unassigned"}

EVENT_LABELS = {
    "pass": "Pass",
    "interception": "Interception",
    "recovery": "Ball Recovery",
    "switch_of_play": "Switch of Play",
    "skill_move": "Skill Move",
    "cross": "Cross",
    "penalty_area_entry": "Penalty Area Entry",
    "final_third_entry": "Final Third Entry",
}


def _team_label(raw):
    if raw == "Team 0":
        return "Team A"
    if raw == "Team 1":
        return "Team B"
    if not raw:
        return "Unknown"
    return raw


def _event_row(e):
    etype = e.get("type", "unknown")
    team = None
    if etype == "pass":
        team = e.get("passer_team")
        detail = f"Player {e.get('passer_id')} → Player {e.get('receiver_id')}"
    elif etype == "interception":
        team = e.get("interceptor_team")
        detail = f"Player {e.get('interceptor_id')} won the ball off Player {e.get('passer_id')}"
    elif etype == "recovery":
        team = e.get("team")
        detail = f"Player {e.get('player_id')} recovered a loose ball"
    elif etype == "switch_of_play":
        team = e.get("passer_team")
        detail = f"Player {e.get('passer_id')} → Player {e.get('receiver_id')} (long switch)"
    elif etype == "skill_move":
        team = e.get("team")
        detail = f"Player {e.get('player_id')} beat a defender"
    elif etype == "cross":
        team = e.get("team")
        detail = f"Player {e.get('player_id')} delivered a cross"
    elif etype == "penalty_area_entry":
        team = e.get("team")
        detail = f"Player {e.get('player_id')} entered the penalty area"
    elif etype == "final_third_entry":
        team = e.get("team")
        detail = f"Player {e.get('player_id')} entered the final third"
    else:
        detail = etype.replace("_", " ").title()

    return {
        "Time": e.get("game_clock") or "",
        "Event": EVENT_LABELS.get(etype, etype.replace("_", " ").title()),
        "Team": _team_label(team),
        "Details": detail,
        "_frame": e.get("frame", 0),
    }

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


from src.exporters.output_schema import OutputFiles, AnalyticsCSVColumns

def _load_csv(name, game_id=None):
    """Load CSV file from game-specific folder or fallback locations."""
    
    # If game_id is provided, try game-specific folder first
    if game_id:
        game_path = os.path.join(INSIGHTS_DIR, game_id)
        p = os.path.join(game_path, name)
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    
    # Fallback to old insights directory
    p = os.path.join(INSIGHTS_DIR, name)
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None


def _load_summary(game_id=None):
    """Load analytics.json metadata from game-specific folder or fallback."""
    
    # If game_id is provided, try game-specific folder first
    if game_id:
        game_path = os.path.join(INSIGHTS_DIR, game_id)
        p = os.path.join(game_path, OutputFiles.ANALYTICS_JSON)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    data = json.load(f)
                    # Extract metadata section for display
                    return data.get("metadata", {})
            except Exception:
                pass
    
    # Fallback to old insights directory
    p = os.path.join(INSIGHTS_DIR, OutputFiles.ANALYTICS_JSON)
    if os.path.exists(p):
        try:
            with open(p) as f:
                data = json.load(f)
                # Extract metadata section for display
                return data.get("metadata", {})
        except Exception:
            pass
    return {}


def _load_events(game_id=None):
    """Load events.json (match events: passes, interceptions, crosses...)."""
    if game_id:
        p = os.path.join(INSIGHTS_DIR, game_id, "events.json")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
    p = os.path.join(INSIGHTS_DIR, "events.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _find_tracked_video(game_id=None):
    """Find tracked video from game-specific folder or fallback locations."""
    # 1. Prefer the path stored in session state (set by analysis_page)
    v = st.session_state.get("tracked_video")
    if v and os.path.exists(v):
        return v
    
    # 2. If game_id is provided, check game-specific folder
    if game_id:
        game_path = os.path.join(INSIGHTS_DIR, game_id)
        canonical = os.path.join(game_path, OutputFiles.ANNOTATED_VIDEO)
        if os.path.exists(canonical):
            return canonical
    
    # 3. Fallback to old locations
    if os.path.exists(PROCESSED_DIR):
        for f in sorted(os.listdir(PROCESSED_DIR)):
            if "tracked" in f and f.endswith(".mp4"):
                return os.path.join(PROCESSED_DIR, f)
    return None


def render():
    page_header("Results", "Possession, player stats, speed analysis, and exports.")
    # render_pipeline(done_up_to=3)
    
    # Game selection
    games = get_game_list()
    
    if games:
        game_options = [(f"{g['video_name']} ({g['status']})", g['game_id']) for g in games]
        selected_option = st.selectbox(
            "Select Game:",
            options=game_options,
            index=0,
            format_func=lambda x: x[0],
            key="game_selector"
        )
        selected_game_id = selected_option[1]
    else:
        st.info("No games found. Run analysis first.")
        selected_game_id = None
    
    # Load data for selected game
    player_df     = _load_csv(OutputFiles.PLAYER_SUMMARY, selected_game_id)
    poss_df       = _load_csv(OutputFiles.POSSESSION_SUMMARY, selected_game_id)
    track_df      = _load_csv(OutputFiles.TRACKING, selected_game_id)
    summary       = _load_summary(selected_game_id)
    events        = _load_events(selected_game_id)
    tracked_video = _find_tracked_video(selected_game_id)

    # Check if any data is available
    if player_df is None and poss_df is None and track_df is None and not events and not tracked_video:
        st.warning("No results found. The analysis may not have completed successfully. Run the analysis pipeline first.")
        if selected_game_id:
            st.info(f"Game folder: {selected_game_id}")
        _, r = st.columns([3, 1])
        with r:
            nav_button("Go to Analysis", "Analysis")
        return

    if tracked_video:
        with st.expander("▶  Tracked Video Preview"):
            st.video(tracked_video)

    # Resolve team heatmaps if available (from game-specific folder)
    heatmap_paths = []
    if selected_game_id:
        game_heatmaps_dir = os.path.join(INSIGHTS_DIR, selected_game_id, "heatmaps")
        if os.path.exists(game_heatmaps_dir):
            heatmap_paths = [
                os.path.join(game_heatmaps_dir, "team_0_heatmap.png"),
                os.path.join(game_heatmaps_dir, "team_1_heatmap.png")
            ]

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    tab_events, tab_poss, tab_heatmap, tab_player, tab_dl = st.tabs([
        "Events", "Possession", "Heatmaps", "Players", "Downloads",
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
                st.plotly_chart(fig, width='stretch')

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
                    st.plotly_chart(fig2, width='stretch')
        else:
            st.info("No possession data available.")

    # EVENTS
    with tab_events:
        if events:
            rows = [_event_row(e) for e in events]
            events_df = pd.DataFrame(rows).sort_values("_frame")

            f1, f2 = st.columns(2)
            with f1:
                event_types = st.multiselect(
                    "Filter by event", sorted(events_df["Event"].unique().tolist()),
                    default=sorted(events_df["Event"].unique().tolist()), key="ev_type_filter",
                )
            with f2:
                team_filter = st.multiselect(
                    "Filter by team", sorted(events_df["Team"].unique().tolist()),
                    default=sorted(events_df["Team"].unique().tolist()), key="ev_team_filter",
                )

            st.caption("Times line up with the video clock — use them to jump to the moment in the recording.")
            timeline = events_df[
                events_df["Event"].isin(event_types) & events_df["Team"].isin(team_filter)
            ][["Time", "Event", "Team", "Details"]]
            st.dataframe(timeline, width='stretch', hide_index=True, height=460)
        else:
            st.info("No match events detected for this game.")

    # HEATMAPS
    with tab_heatmap:
        if any(os.path.exists(p) for p in heatmap_paths):
            c0, c1 = st.columns(2)
            cols = [c0, c1]
            for i, col in enumerate(cols):
                p = heatmap_paths[i]
                with col:
                    if os.path.exists(p):
                        st.image(p, width='stretch', caption=TEAM_NAMES.get(i, f"Team {i}"))
                    else:
                        st.info(f"{TEAM_NAMES.get(i, f'Team {i}')} heatmap not available")
        else:
            st.info("No heatmaps available for this game.")

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
            st.dataframe(filtered, width='stretch', hide_index=True, height=380)
        else:
            st.info("No player data available.")

    # DOWNLOADS
    with tab_dl:
        downloads = [
            ("Player Summary", OutputFiles.PLAYER_SUMMARY, player_df, "Per-player stats: speed, possession, team."),
            ("Possession", OutputFiles.POSSESSION_SUMMARY, poss_df, "Team-level possession percentages."),
            ("Tracking Data", OutputFiles.TRACKING, track_df, "Frame-by-frame tracking with velocity."),
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
                                       width='stretch', key=f"dl_{fname}")
                else:
                    st.button("Not available", disabled=True, width='stretch', key=f"na_{fname}")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        if tracked_video:
            with open(tracked_video, "rb") as vf:
                st.download_button("⬇  Download Tracked Video (MP4)", data=vf.read(),
                                   file_name=os.path.basename(tracked_video),
                                   mime="video/mp4", width='stretch', type="primary")
        if summary:
            st.download_button("⬇  Download Analytics (JSON)",
                               data=json.dumps(summary, indent=2),
                               file_name=OutputFiles.ANALYTICS_JSON,
                               mime="application/json", width='stretch')
        if events:
            st.download_button("⬇  Download Match Events (JSON)",
                               data=json.dumps(events, indent=2),
                               file_name="events.json",
                               mime="application/json", width='stretch')

    st.markdown("---")
    left, center, right = st.columns(3)
    with left:
        nav_button("← Back to Analysis", "Analysis", key="res_back")
    with center:
        if st.button("Start New Analysis", width='stretch'):
            for k in ["uploaded_video", "uploaded_video_name", "processed_video",
                      "analysis_done", "analysis_results", "tracked_video"]:
                st.session_state.pop(k, None)
            st.session_state.page = "Upload"
            st.rerun()
    with right:
        nav_button("Home", "Home", key="res_home")
