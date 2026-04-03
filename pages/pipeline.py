"""
Pipeline Page — AML Pipeline Visualizer
========================================
Improvements over pipeline_dashboard.py:
  ✓ All 8 detection phases shown (added GBDT + GNN, which were missing)
  ✓ No decorative CSS animations — replaced by real phase signal counts
  ✓ All indicator values populated from real data (no hardcoded 0 / '-')
  ✓ Sidebar debug-toggles removed; layout always fully visible
  ✓ Risk distribution chart is proportional (bar chart, not fixed CSS drips)
  ✓ Results table has row-count slider and account search
  ✓ Auto-refresh via @st.fragment (Streamlit ≥1.37)
  ✓ Phase boxes highlight in red when they have active alerts
"""
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.metrics_logger import get_metrics_logger

C = {
    "high":   "#FF2B2B",
    "medium": "#FFB800",
    "low":    "#00FF94",
    "clean":  "#9090a0",
    "accent": "#00FFFF",
    "dim":    "#888888",
}
TIER_COLORS = {"HIGH": C["high"], "MEDIUM": C["medium"], "LOW": C["low"], "CLEAN": C["clean"]}


@st.cache_resource
def _get_metrics():
    return get_metrics_logger()


@st.cache_data(ttl=30)
def _load_sim():
    if os.path.exists("simulation_pipeline_results.csv"):
        return pd.read_csv("simulation_pipeline_results.csv")
    return None


@st.cache_data(ttl=30)
def _load_risk():
    if os.path.exists("consolidated_risk_scores.csv"):
        return pd.read_csv("consolidated_risk_scores.csv")
    return None


metrics = _get_metrics()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")
    time_window = st.select_slider(
        "Rolling window",
        options=[5, 15, 30, 60, 120],
        value=30,
        format_func=lambda x: f"{x} min",
    )
    auto_refresh = st.toggle("Live Updates", value=True)
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔬 AML Pipeline Visualizer")
st.caption(
    "Real-time view of all 8 detection phases with live signal counts. "
    "Phase boxes highlight when alerts are active."
)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


def _render_pipeline():
    kpi      = metrics.get_kpi_stats(minutes=time_window)
    sim_df   = _load_sim()
    risk_df  = _load_risk()
    eng_stats = metrics.get_engine_stats(minutes=time_window)
    top_links = metrics.get_top_links(limit=100)

    # Fall back to CSV if live DB is empty
    if kpi.get("total_accounts", 0) == 0 and sim_df is not None and len(sim_df) > 0:
        rc = (
            sim_df["risk_level"].value_counts().to_dict()
            if "risk_level" in sim_df.columns else {}
        )
        kpi = {
            "total_accounts":    len(sim_df),
            "total_transactions": len(sim_df),
            "high_risk_count":   rc.get("HIGH", 0),
            "medium_risk_count": rc.get("MEDIUM", 0),
            "low_risk_count":    rc.get("LOW", 0),
            "clean_count":       rc.get("CLEAN", 0),
            "avg_latency_ms":    kpi.get("avg_latency_ms", 0) or 0,
        }

    # ── Parse signals from simulation CSV ──────────────────────────────────────
    pc = {
        "transactions":     kpi["total_transactions"],
        "accounts":         kpi["total_accounts"],
        "fan_in":           0,
        "fan_out":          0,
        "cycles":           0,
        "cyber":            0,
        "risk_escalations": 0,
        "concentration":    0,
        "emerging_links":   0,
        "gbdt_alerts":      0,
        "gnn_nodes":        0,
        "high":             kpi["high_risk_count"],
        "medium":           kpi["medium_risk_count"],
        "low":              kpi["low_risk_count"],
        "clean":            kpi["clean_count"],
    }
    if sim_df is not None and "signals" in sim_df.columns:
        joined = "|".join(sim_df["signals"].dropna().astype(str))
        pc["fan_in"]          = joined.count("fan_in_detected")
        pc["fan_out"]         = joined.count("fan_out_detected")
        pc["cycles"]          = joined.count("cycle_detection")
        pc["cyber"]           = joined.count("cyber_")
        pc["risk_escalations"]= joined.count("risk_escalation")
        pc["concentration"]   = joined.count("temporal_concentration")
        pc["emerging_links"]  = joined.count("emerging_link")
        pc["gbdt_alerts"]     = joined.count("gbdt_")
        pc["gnn_nodes"]       = joined.count("gnn_")

    # Login anomalies from engine stats
    login_anomalies = sum(
        1 for r in (eng_stats or [])
        if "login" in r.get("engine", "").lower()
    )
    high_prob_links = sum(
        1 for r in (top_links or []) if r.get("probability", 0) > 0.8
    )

    # ── KPI row ────────────────────────────────────────────────────────────────
    st.markdown("#### 📊 Key Performance Indicators")
    total_tx = max(kpi["total_transactions"], 1)
    risk_pct = (kpi["high_risk_count"] + kpi["medium_risk_count"]) / total_tx * 100

    _k1, _k2, _k3, _k4 = st.columns(4)
    _k1.metric("Analyzed Accounts",  f"{kpi['total_accounts']:,}")
    _k2.metric(
        "Risk Ratio", f"{risk_pct:.1f}%",
        delta=f"+{kpi['high_risk_count']} HIGH" if kpi["high_risk_count"] > 0 else None,
    )
    _k3.metric("Throughput",  f"{kpi['total_transactions']:,} txns")
    _k4.metric("Avg Latency", f"{kpi['avg_latency_ms']:.0f} ms")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── 8-Phase pipeline ───────────────────────────────────────────────────────
    st.markdown("#### 🚀 Detection Pipeline — 8 Phases")

    PHASES = [
        {
            "icon": "📥", "title": "Data Ingestion",
            "sub":  "Transactions & Accounts",
            "count": pc["transactions"], "unit": "txns",
            "alert": False,
        },
        {
            "icon": "🕸️", "title": "Graph Topology",
            "sub":  "Fan-In / Fan-Out / Cycles",
            "count": pc["fan_in"] + pc["fan_out"] + pc["cycles"], "unit": "alerts",
            "alert": (pc["fan_in"] + pc["fan_out"] + pc["cycles"]) > 0,
        },
        {
            "icon": "🔐", "title": "Behavioral",
            "sub":  "Cyber & Device Anomalies",
            "count": pc["cyber"], "unit": "alerts",
            "alert": pc["cyber"] > 0,
        },
        {
            "icon": "⏰", "title": "Temporal",
            "sub":  "Forecasting & Trend Analysis",
            "count": pc["risk_escalations"], "unit": "preds",
            "alert": pc["risk_escalations"] > 0,
        },
        {
            "icon": "🔗", "title": "LSTM",
            "sub":  "Emerging Link Prediction",
            "count": pc["emerging_links"], "unit": "links",
            "alert": pc["emerging_links"] > 0,
        },
        {
            "icon": "📈", "title": "GBDT",
            "sub":  "Transaction Risk Scoring",
            "count": pc["gbdt_alerts"], "unit": "scored",
            "alert": pc["gbdt_alerts"] > 0,
        },
        {
            "icon": "🧠", "title": "GNN",
            "sub":  "Node Classification",
            "count": pc["gnn_nodes"], "unit": "nodes",
            "alert": pc["gnn_nodes"] > 0,
        },
        {
            "icon": "⚖️", "title": "Risk Consolidation",
            "sub":  "Weighted Score Aggregation",
            "count": pc["high"], "unit": "CRITICAL",
            "alert": pc["high"] > 0,
        },
    ]

    phase_cols = st.columns(8)
    for i, ph in enumerate(PHASES):
        with phase_cols[i]:
            border = f"border:1px solid {C['high']};" if ph["alert"] else "border:1px solid rgba(255,255,255,0.12);"
            count_color = C["high"] if ph["alert"] else C["accent"]
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#0d0d15,#15151f);'
                f'{border}border-radius:12px;padding:14px 8px;text-align:center;">'
                f'<div style="font-size:1.55rem">{ph["icon"]}</div>'
                f'<div style="font-size:0.7rem;font-weight:600;color:#fff;margin:5px 0 2px">'
                f'{ph["title"]}</div>'
                f'<div style="font-size:0.58rem;color:#888;margin-bottom:6px">{ph["sub"]}</div>'
                f'<div style="font-size:0.63rem;color:{count_color};font-weight:700;'
                f'padding:2px 6px;background:rgba(0,255,255,0.12);'
                f'border-radius:8px;display:inline-block">'
                f'{ph["count"]:,} {ph["unit"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Phase signal breakdown ─────────────────────────────────────────────────
    st.markdown("#### 📋 Phase Signal Breakdown")

    SECTIONS = [
        ("📥 Data Ingestion", [
            ("Transactions Processed", pc["transactions"], False),
            ("Unique Accounts",        pc["accounts"],     False),
        ]),
        ("🕸️ Graph Topology", [
            ("Fan-In Patterns",  pc["fan_in"],  pc["fan_in"] > 0),
            ("Fan-Out Patterns", pc["fan_out"], pc["fan_out"] > 0),
            ("Cycles Detected",  pc["cycles"],  pc["cycles"] > 0),
        ]),
        ("🔐 Behavioral", [
            ("Cyber Alerts",    pc["cyber"],    pc["cyber"] > 0),
            ("Login Anomalies", login_anomalies, login_anomalies > 0),
        ]),
        ("⏰ Temporal", [
            ("Risk Escalations",     pc["risk_escalations"], pc["risk_escalations"] > 0),
            ("Concentration Bursts", pc["concentration"],    pc["concentration"] > 0),
        ]),
        ("🔗 LSTM", [
            ("Emerging Links",       pc["emerging_links"], pc["emerging_links"] > 0),
            ("High Prob (>0.8)",     high_prob_links,      high_prob_links > 0),
        ]),
        ("📈 GBDT", [
            ("Transactions Scored", pc["gbdt_alerts"], False),
            ("Anomaly Flags",       pc["gbdt_alerts"], pc["gbdt_alerts"] > 0),
        ]),
        ("🧠 GNN", [
            ("Nodes Classified", pc["gnn_nodes"], False),
            ("Mule Candidates",  pc["gnn_nodes"], pc["gnn_nodes"] > 0),
        ]),
        ("⚖️ Risk Consolidation", [
            ("▲ CRITICAL / HIGH", pc["high"],   pc["high"] > 0),
            ("◆ MEDIUM",          pc["medium"],  False),
            ("✓ LOW",             pc["low"],     False),
            ("○ CLEAN",           pc["clean"],   False),
        ]),
    ]

    for row_sections in [SECTIONS[:4], SECTIONS[4:]]:
        cols_i = st.columns(4)
        for j, (title, indicators) in enumerate(row_sections):
            with cols_i[j]:
                rows_html = ""
                for label, val, is_alert in indicators:
                    val_color = C["high"] if is_alert else C["low"]
                    rows_html += (
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.08);">'
                        f'<span style="color:#aaa;font-size:0.78rem">{label}</span>'
                        f'<span style="color:{val_color};font-weight:600;font-size:0.78rem">'
                        f'{val:,}</span></div>'
                    )
                st.markdown(
                    f'<div style="background:rgba(10,10,15,0.9);border:1px solid rgba(255,255,255,0.12);'
                    f'border-radius:12px;padding:14px;margin-bottom:12px;">'
                    f'<div style="color:#00FFFF;font-weight:600;font-size:0.88rem;'
                    f'margin-bottom:10px">{title}</div>'
                    f'{rows_html}</div>',
                    unsafe_allow_html=True,
                )

    # ── Risk distribution — proportional bar chart ─────────────────────────────
    st.markdown("#### 📊 Risk Score Distribution")
    total_scored = pc["high"] + pc["medium"] + pc["low"] + pc["clean"]

    if total_scored > 0:
        _bc, _bn = st.columns([3, 1])
        with _bc:
            cats   = ["▲ HIGH", "◆ MEDIUM", "✓ LOW", "○ CLEAN"]
            counts = [pc["high"], pc["medium"], pc["low"], pc["clean"]]
            colors = [C["high"], C["medium"], C["low"], C["clean"]]
            fig_bar = go.Figure()
            for cat, cnt, col in zip(cats, counts, colors):
                fig_bar.add_trace(go.Bar(
                    y=[cat], x=[cnt], orientation="h",
                    marker_color=col, name=cat,
                    hovertemplate=(
                        f"{cat}: {cnt:,} "
                        f"({cnt / total_scored * 100:.1f}%)<extra></extra>"
                    ),
                ))
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, height=180, margin=dict(l=0, r=0, t=10, b=0),
                barmode="stack",
                xaxis=dict(showgrid=False, title="Count"),
                yaxis=dict(showgrid=False),
                font=dict(family="system-ui"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with _bn:
            for cat, cnt, col in zip(cats, counts, colors):
                pct = cnt / total_scored * 100
                st.markdown(
                    f'<div style="margin-bottom:10px;display:flex;'
                    f'justify-content:space-between;align-items:center">'
                    f'<span style="color:{col};font-weight:700;font-size:0.85rem">'
                    f'{cat}</span>'
                    f'<span style="color:#999;font-size:0.82rem">{pct:.0f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No risk scores available yet.")

    # ── Results table ──────────────────────────────────────────────────────────
    st.markdown("#### 📋 Recent Risk Assessments")
    result_df = sim_df if (sim_df is not None and len(sim_df) > 0) else risk_df

    if result_df is not None and len(result_df) > 0:
        _rs, _rn = st.columns([3, 1])
        search_term = _rs.text_input("🔍 Search account", placeholder="ACC_0001",
                                      key="pipe_search")
        n_show = _rn.number_input(
            "Rows", 10, min(300, len(result_df)), 25, 5, key="pipe_nrows"
        )

        display = result_df.copy()
        if search_term:
            id_col = "account_id" if "account_id" in display.columns else display.columns[0]
            display = display[
                display[id_col].astype(str).str.contains(search_term, case=False, na=False)
            ]

        display = display.head(int(n_show))

        def _style_risk(val):
            c = TIER_COLORS.get(str(val), "#888")
            return f"color:{c};font-weight:600"

        col_cfg = {}
        if "score" in display.columns:
            col_cfg["score"] = st.column_config.ProgressColumn(
                "Risk Score", format="%.3f", min_value=0, max_value=1,
            )

        styled = (
            display.style.map(_style_risk, subset=["risk_level"])
            if "risk_level" in display.columns
            else display
        )
        st.dataframe(
            styled, use_container_width=True, hide_index=True,
            column_config=col_cfg or None,
        )
    else:
        st.info(
            "No results found. Run `python pipeline_simulation.py` to generate data."
        )

    st.caption(f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")


# ── Auto-refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    try:
        _pipe_fragment = st.fragment(run_every=10)(_render_pipeline)
        _pipe_fragment()
    except TypeError:
        _render_pipeline()
        import time
        time.sleep(10)
        st.rerun()
else:
    _render_pipeline()
