"""
FRAUD.GUARD | AML Detection Platform
=====================================
Unified multi-page Streamlit entry point.

Run with:
    streamlit run app.py

Pages:
    🛡️  Monitor       — Live inference feed, KPIs, latency charts, manual scoring
    🔬  Pipeline      — 8-phase pipeline visualisation with real signal counts
    🕸️  Graph Explorer — Interactive Plotly transaction network
    🔍  Account Detail — Per-account drill-down, timeline, and alert management
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

st.set_page_config(
    page_title="FRAUD.GUARD | AML Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS injected once for all pages ────────────────────────────────────
st.markdown("""
<style>
    /* Font stack: Outfit from Google Fonts with full offline fallback */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #050505 60%);
        font-family: 'Outfit', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }

    /* ── Glassmorphism card ─────────────────────────────────────────────── */
    .glass-card {
        background: rgba(20, 20, 25, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }

    /* ── Typography ─────────────────────────────────────────────────────── */
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8888aa;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.1;
    }

    /* ── Risk badges (colour + text for colour-blind accessibility) ──────── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    .badge-CRITICAL { background: rgba(255,43,43,0.15);  color: #FF2B2B; border: 1px solid #FF2B2B44; }
    .badge-HIGH     { background: rgba(255,107,53,0.15); color: #FF6B35; border: 1px solid #FF6B3544; }
    .badge-MEDIUM   { background: rgba(255,184,0,0.15);  color: #FFB800; border: 1px solid #FFB80044; }
    .badge-LOW      { background: rgba(0,255,148,0.15);  color: #00FF94; border: 1px solid #00FF9444; }
    .badge-CLEAN    { background: rgba(100,100,100,0.15);color: #888888; border: 1px solid #55555544; }

    /* ── Section header ─────────────────────────────────────────────────── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
        margin: 24px 0 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(0,255,255,0.2);
    }

    /* ── Custom scrollbar ───────────────────────────────────────────────── */
    ::-webkit-scrollbar       { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #555; }
</style>
""", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
monitor_page = st.Page("pages/monitor.py",        title="Monitor",        icon="🛡️", default=True)
pipeline_page = st.Page("pages/pipeline.py",       title="Pipeline",       icon="🔬")
graph_page    = st.Page("pages/graph_explorer.py", title="Graph Explorer", icon="🕸️")
account_page  = st.Page("pages/account_detail.py", title="Account Detail", icon="🔍")

pg = st.navigation([monitor_page, pipeline_page, graph_page, account_page])
pg.run()
