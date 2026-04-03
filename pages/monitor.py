"""
Monitor Page — FRAUD.GUARD | AML Monitor
=========================================
Improvements over dashboard.py:
  ✓ Non-blocking auto-refresh via @st.fragment (Streamlit ≥1.37)
  ✓ Sidebar expanded by default (set in app.py)
  ✓ Real system metrics via psutil (optional; graceful fallback)
  ✓ Fixed donut 'Clean' colour (was near-invisible)
  ✓ Fixed latency chart colours (indexed palette, not fragile string-match)
  ✓ Search + multi-select filter on inference feed
  ✓ CSV export button on filtered feed
  ✓ Fixed metric-card delta bug (no more 'Active' coloured red)
  ✓ Accessibility: text symbols alongside colour (▲ HIGH, ◆ MEDIUM, ✓ LOW, ○ CLEAN)
  ✓ Account drill-down (score trend + signal summary)
  ✓ Alert management (status + notes, persisted via alert_store)
  ✓ Manual transaction scoring form (POSTs to inference API)
  ✓ Date-range filter alongside rolling-window slider
"""
import io
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.alert_store import get_alert_state, set_alert_state
from src.metrics_logger import get_metrics_logger

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    'accent':  '#00FF94',
    'high':    '#FF2B2B',
    'medium':  '#FFB800',
    'low':     '#00FF94',
    'clean':   '#555555',
    'info':    '#00BFFF',
}
TIER_COLORS  = {'HIGH': C['high'], 'MEDIUM': C['medium'], 'LOW': C['low'], 'CLEAN': C['clean']}
TIER_SYMBOLS = {'HIGH': '▲', 'MEDIUM': '◆', 'LOW': '✓', 'CLEAN': '○'}
# Indexed colour palette for latency chart engines (no string-parsing)
ENGINE_PALETTE = ['#00FFFF', '#FF6B35', '#00FF94', '#FF2B2B', '#FFB800', '#A78BFA', '#34D399', '#F87171']


@st.cache_resource
def _get_metrics():
    return get_metrics_logger()

metrics = _get_metrics()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Monitor Controls")

    time_window = st.select_slider(
        "Rolling window",
        options=[5, 15, 30, 60, 120, 360],
        value=30,
        format_func=lambda x: f"{x} min",
    )

    st.markdown("**Historical range**")
    date_from = st.date_input("From", value=datetime.utcnow().date() - timedelta(days=7))
    date_to   = st.date_input("To",   value=datetime.utcnow().date())

    auto_refresh = st.toggle("Live Updates", value=True)

    st.divider()
    st.markdown("### 🖥️ System Status")
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem     = psutil.virtual_memory()
        mem_gb  = mem.used  / 1024 ** 3
        mem_tot = mem.total / 1024 ** 3
        _c1, _c2 = st.columns(2)
        _c1.metric("CPU", f"{cpu_pct:.0f}%")
        _c2.metric("RAM", f"{mem_gb:.1f} GB", f"/ {mem_tot:.0f} GB total")
    except ImportError:
        st.caption("Install `psutil` for live system metrics.")

    st.caption("Build v3.0.0")

# ── Page header ────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(
        "# 🛡️ FRAUD.GUARD <span style='font-weight:300;opacity:0.45'>| MONITOR</span>",
        unsafe_allow_html=True,
    )
with col_status:
    now_str = datetime.utcnow().strftime("%H:%M:%S UTC")
    st.markdown(
        f'<div style="text-align:right;padding-top:14px;">'
        f'<span style="color:#00FF94;font-weight:600;">● OPERATIONAL</span>'
        f'<br><span style="color:#555;font-size:0.78rem">{now_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── Manual transaction scoring form ────────────────────────────────────────────
with st.expander("🎯 Manual Transaction Score", expanded=False):
    st.caption("Submit a transaction to the inference API for immediate scoring.")
    with st.form("manual_score_form", clear_on_submit=False):
        _fc1, _fc2, _fc3, _fc4 = st.columns(4)
        _src  = _fc1.text_input("Source Account", placeholder="ACC_0001")
        _tgt  = _fc2.text_input("Target Account", placeholder="ACC_0002")
        _amt  = _fc3.number_input("Amount ($)", min_value=0.01, value=5000.0, step=100.0)
        _chan = _fc4.selectbox("Channel", ["online", "mobile", "branch", "atm"])
        _submitted = st.form_submit_button("Score Transaction", type="primary")

        if _submitted and _src and _tgt:
            try:
                import requests as _req
                _payload = {
                    "source": _src, "target": _tgt,
                    "amount": _amt, "channel": _chan,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                _resp = _req.post(
                    "http://localhost:5000/score/transaction",
                    json=_payload, timeout=3,
                )
                if _resp.ok:
                    _r = _resp.json()
                    _risk = _r.get("risk_level", "UNKNOWN")
                    _score = _r.get("risk_score", 0)
                    _color = TIER_COLORS.get(_risk, "#888")
                    st.markdown(
                        f'<div style="padding:12px;border-radius:8px;'
                        f'border:1px solid {_color};background:rgba(0,0,0,0.4);margin-top:8px;">'
                        f'<b style="color:{_color}">'
                        f'{TIER_SYMBOLS.get(_risk,"●")} {_risk}</b>'
                        f' &nbsp; Score: <b>{_score:.4f}</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"API returned HTTP {_resp.status_code}")
            except Exception as _e:
                st.warning(
                    f"Inference API unavailable ({_e}). "
                    f"Start it with `python inference_api.py`."
                )
        elif _submitted:
            st.warning("Please fill in both Source and Target account fields.")


# ── Live data section (auto-refreshed) ─────────────────────────────────────────

def _render_live():
    """Fetches live data and renders KPIs, charts, and the inference feed."""
    kpi        = metrics.get_kpi_stats(minutes=time_window)
    inferences = metrics.get_recent_inferences(limit=300)
    trends     = metrics.get_latency_trends(limit=200)

    total_tx  = max(kpi["total_transactions"], 1)
    risk_ratio = (kpi["high_risk_count"] + kpi["medium_risk_count"]) / total_tx * 100

    # ── KPI cards ──────────────────────────────────────────────────────────────
    def _kpi_html(label, value, sub=None, value_color="#ffffff"):
        sub_html = (
            f'<div style="font-size:0.8rem;color:#666;margin-top:4px">{sub}</div>'
            if sub else ""
        )
        return (
            f'<div class="glass-card" style="text-align:center">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{value_color}">{value}</div>'
            f'{sub_html}</div>'
        )

    _k1, _k2, _k3, _k4 = st.columns(4)
    _k1.markdown(_kpi_html("Analyzed Accounts", f"{kpi['total_accounts']:,}",
                            "unique accounts"), unsafe_allow_html=True)
    _k2.markdown(_kpi_html("Risk Ratio", f"{risk_ratio:.1f}%",
                            f"▲ {kpi['high_risk_count']} HIGH alerts", C["high"]),
                 unsafe_allow_html=True)
    _k3.markdown(_kpi_html("Throughput", f"{kpi['total_transactions']:,}",
                            "transactions"), unsafe_allow_html=True)
    _k4.markdown(_kpi_html("Avg Latency", f"{kpi['avg_latency_ms']:.0f}<span style='font-size:1rem'>ms</span>",
                            None, C["accent"]), unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────────────────────────
    _cl, _cr = st.columns([2, 1])

    with _cl:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Engine Latency Trends")
        if trends:
            lat_df = pd.DataFrame(trends)
            lat_df["timestamp"] = pd.to_datetime(lat_df["timestamp"])
            lat_df = lat_df.sort_values("timestamp")
            engines = lat_df["engine"].unique().tolist()
            fig_lat = go.Figure()
            for idx, engine in enumerate(engines):
                edf = lat_df[lat_df["engine"] == engine]
                hex_c = ENGINE_PALETTE[idx % len(ENGINE_PALETTE)]
                # Convert hex to rgb for fillcolor
                r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
                fig_lat.add_trace(go.Scatter(
                    x=edf["timestamp"], y=edf["latency_ms"],
                    name=engine, mode="lines",
                    line=dict(width=2, color=hex_c),
                    fill="tozeroy",
                    fillcolor=f"rgba({r},{g},{b},0.07)",
                ))
            fig_lat.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                           title="ms"),
                legend=dict(orientation="h", y=1.18, font=dict(size=11)),
                font=dict(family="system-ui"),
            )
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("No latency data yet — start the inference API to populate.")
        st.markdown("</div>", unsafe_allow_html=True)

    with _cr:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Risk Distribution")
        _values = [
            kpi["high_risk_count"], kpi["medium_risk_count"],
            kpi["low_risk_count"],  kpi["clean_count"],
        ]
        _labels = ["▲ HIGH", "◆ MEDIUM", "✓ LOW", "○ CLEAN"]
        # Clean uses a clearly visible neutral grey (was near-invisible before)
        _pie_colors = [C["high"], C["medium"], C["low"], C["clean"]]
        fig_pie = go.Figure(data=[go.Pie(
            labels=_labels, values=_values, hole=0.6,
            marker=dict(colors=_pie_colors),
            textinfo="label+percent", showlegend=False,
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )])
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            annotations=[dict(
                text=str(sum(_values)), x=0.5, y=0.5,
                font_size=22, font_color="#fff", showarrow=False,
            )],
            font=dict(family="system-ui"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Inference feed ─────────────────────────────────────────────────────────
    st.markdown("### 🚨 Live Inference Feed")

    # Filter controls
    _f1, _f2, _f3 = st.columns([2, 2, 1])
    search_acc  = _f1.text_input("🔍 Account search", placeholder="ACC_0001",
                                  key="mon_search")
    risk_filter = _f2.multiselect(
        "Risk level", ["HIGH", "MEDIUM", "LOW", "CLEAN"],
        default=["HIGH", "MEDIUM", "LOW", "CLEAN"], key="mon_risk_filter",
    )
    n_rows = _f3.number_input("Rows", 10, 300, 50, 10, key="mon_nrows")

    if inferences:
        df = pd.DataFrame(inferences)
        df["Time"]   = df["timestamp"].str[11:19]
        df = df.rename(columns={
            "account_id": "Account", "risk_score": "Score",
            "risk_level": "Level",   "latency_ms": "Latency (ms)",
        })

        # Apply search & risk-level filters
        if search_acc:
            df = df[df["Account"].str.contains(search_acc, case=False, na=False)]
        if risk_filter:
            df = df[df["Level"].isin(risk_filter)]

        # Apply date range (inference timestamp date only)
        if "timestamp" in df.columns:
            df["_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
            df = df[(df["_date"] >= date_from) & (df["_date"] <= date_to)]

        display_df = df[["Time", "Account", "Score", "Level", "Latency (ms)"]].head(
            int(n_rows)
        ).copy()
        display_df["Latency (ms)"] = display_df["Latency (ms)"].round(1)

        # ── CSV export ─────────────────────────────────────────────────────────
        _buf = io.StringIO()
        display_df.to_csv(_buf, index=False)
        st.download_button(
            "⬇ Export filtered feed as CSV",
            data=_buf.getvalue(),
            file_name=f"aml_alerts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        # Add accessibility symbols to Level column BEFORE styling
        display_df["Level"] = display_df["Level"].map(
            lambda v: f"{TIER_SYMBOLS.get(str(v), '●')} {v}"
        )

        def _color_level(val):
            for tier, sym in TIER_SYMBOLS.items():
                if str(val).startswith(sym) or str(val).endswith(tier):
                    return f"color:{TIER_COLORS.get(tier,'#888')};font-weight:600"
            return ""

        st.dataframe(
            display_df.style.map(_color_level, subset=["Level"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", format="%.3f", min_value=0, max_value=1,
                )
            },
        )

        # ── Account drill-down ─────────────────────────────────────────────────
        st.markdown("### 🔎 Account Drill-Down")
        available = df["Account"].dropna().unique().tolist()
        if available:
            selected_acc = st.selectbox(
                "Select account to inspect",
                ["— select —"] + available,
                key="mon_drilldown",
            )
            if selected_acc and selected_acc != "— select —":
                acc_rows = df[df["Account"] == selected_acc].copy()
                # Restore numeric score for trend chart
                acc_rows["Score_raw"] = (
                    pd.to_numeric(df.loc[df["Account"] == selected_acc, "Score"],
                                  errors="coerce")
                )

                _dd1, _dd2 = st.columns([2, 1])

                with _dd1:
                    if len(acc_rows) > 0:
                        fig_trend = px.line(
                            acc_rows.reset_index(drop=True),
                            y="Score_raw",
                            title=f"Risk Score Trend — {selected_acc}",
                            markers=True,
                            template="plotly_dark",
                            color_discrete_sequence=[C["accent"]],
                        )
                        fig_trend.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=240,
                            margin=dict(l=0, r=0, t=40, b=0),
                            yaxis=dict(range=[0, 1], showgrid=True,
                                       gridcolor="rgba(255,255,255,0.07)",
                                       title="Risk Score"),
                            font=dict(family="system-ui"),
                        )
                        fig_trend.add_hline(
                            y=0.75, line_dash="dot", line_color=C["high"],
                            annotation_text="CRITICAL", annotation_font_color=C["high"],
                        )
                        fig_trend.add_hline(
                            y=0.60, line_dash="dot", line_color=C["medium"],
                            annotation_text="HIGH", annotation_font_color=C["medium"],
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)

                with _dd2:
                    if len(acc_rows) > 0:
                        latest   = acc_rows.iloc[0]
                        lvl_raw  = str(latest.get("Level", ""))
                        # Strip accessibility symbol to get raw tier
                        tier_key = lvl_raw.split()[-1] if " " in lvl_raw else lvl_raw
                        tier_color = TIER_COLORS.get(tier_key, "#888")
                        score_disp = latest.get("Score_raw", "—")
                        st.markdown(
                            f'<div class="glass-card" style="text-align:center;'
                            f'border-color:{tier_color}44">'
                            f'<div class="metric-label">Current Risk</div>'
                            f'<div class="metric-value" style="color:{tier_color}">'
                            f'{latest.get("Level","—")}</div>'
                            f'<div style="color:#666;font-size:0.85rem;margin-top:4px">'
                            f'Score: {score_disp:.3f if isinstance(score_disp, float) else score_disp}'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )

                    # Alert management
                    st.markdown("**📋 Alert Status**")
                    alert_info  = get_alert_state(selected_acc)
                    current_st  = alert_info.get("state", "Unreviewed")
                    STATES      = ["Unreviewed", "Investigating", "False Positive", "Escalated"]
                    new_state   = st.selectbox(
                        "Status", STATES,
                        index=STATES.index(current_st) if current_st in STATES else 0,
                        key=f"mon_alert_{selected_acc}",
                    )
                    note_text = st.text_area(
                        "Notes", value=alert_info.get("note", ""),
                        height=90, key=f"mon_note_{selected_acc}",
                        placeholder="Add investigation notes…",
                    )
                    if st.button("💾 Save Status", key=f"mon_save_{selected_acc}"):
                        set_alert_state(selected_acc, new_state, note_text)
                        st.success(f"Saved: {new_state}", icon="✅")
                    if alert_info.get("updated_at"):
                        st.caption(f"Last updated: {alert_info['updated_at'][:19]}")

    else:
        st.info(
            "No inference data yet. "
            "Start the inference API (`python inference_api.py`) and the "
            "simulator (`python transaction_simulator.py`) to see live results."
        )


# ── Wire up auto-refresh ───────────────────────────────────────────────────────
if auto_refresh:
    try:
        # Streamlit ≥1.37: only this fragment reruns every 5 s;
        # sidebar controls remain interactive the whole time.
        _live_fragment = st.fragment(run_every=5)(_render_live)
        _live_fragment()
    except TypeError:
        # Streamlit <1.37: fall back to full-page rerun
        _render_live()
        import time
        time.sleep(5)
        st.rerun()
else:
    _render_live()
    if st.button("🔄 Refresh Now"):
        st.rerun()
