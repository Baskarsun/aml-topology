"""
AML Pipeline Visualization Dashboard - Animated Edition
========================================================
An animated dashboard visualizing the AML detection pipeline flow
with transactions flowing through each phase, key indicators,
and risk scores dripping out at the end.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.metrics_logger import get_metrics_logger

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AML Pipeline Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== COLOR SCHEME ====================
colors = {
    'bg': '#050505',
    'card': '#0a0a0a',
    'border': '#1a1a1a',
    'text': '#E0E0E0',
    'accent': '#00FFFF',      # Cyan - Predicted links
    'high_risk': '#FF2B2B',   # Red
    'medium_risk': '#FFB800', # Orange
    'low_risk': '#00FF94',    # Green
    'clean': '#444444',       # Gray
    'edge_normal': '#555555',
    'phase_active': '#00FF94',
    'phase_inactive': '#333333',
    'glow_cyan': 'rgba(0, 255, 255, 0.6)',
    'particle': '#00FFFF'
}

# ==================== CUSTOM CSS WITH ANIMATIONS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    .stApp {
        background-color: #050505;
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00FFFF, #00FF94);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* ========== ANIMATED PIPELINE CONTAINER ========== */
    .pipeline-container {
        position: relative;
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0f18 100%);
        border: 1px solid #1a1a2e;
        border-radius: 20px;
        padding: 30px 20px;
        margin: 20px 0;
        overflow: hidden;
    }
    
    .pipeline-track {
        position: relative;
        height: 120px;
        background: linear-gradient(90deg, 
            rgba(0,255,255,0.05) 0%, 
            rgba(0,255,148,0.05) 50%, 
            rgba(255,43,43,0.05) 100%);
        border-radius: 10px;
        margin: 20px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 20px;
    }
    
    /* ========== PHASE BOXES ========== */
    .phase-box {
        position: relative;
        background: linear-gradient(135deg, #0d0d15, #15151f);
        border: 1px solid #2a2a3e;
        border-radius: 12px;
        padding: 15px 12px;
        text-align: center;
        width: 130px;
        z-index: 10;
        transition: all 0.3s ease;
    }
    
    .phase-box:hover {
        border-color: #00FFFF;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.3);
        transform: translateY(-3px);
    }
    
    .phase-icon {
        font-size: 1.8rem;
        margin-bottom: 5px;
    }
    
    .phase-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 3px;
    }
    
    .phase-indicator {
        font-size: 0.65rem;
        color: #00FFFF;
        font-weight: 700;
        padding: 3px 8px;
        background: rgba(0, 255, 255, 0.1);
        border-radius: 10px;
        margin-top: 5px;
    }
    
    /* ========== FLOWING PARTICLES ========== */
    @keyframes flowRight {
        0% { left: 0%; opacity: 0; }
        5% { opacity: 1; }
        95% { opacity: 1; }
        100% { left: 100%; opacity: 0; }
    }
    
    .particle {
        position: absolute;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: radial-gradient(circle, #00FFFF 0%, rgba(0,255,255,0.3) 70%);
        box-shadow: 0 0 15px #00FFFF, 0 0 30px rgba(0,255,255,0.5);
        top: 50%;
        transform: translateY(-50%);
        animation: flowRight 4s linear infinite;
        z-index: 5;
    }
    
    .particle:nth-child(2) { animation-delay: 0.5s; background: radial-gradient(circle, #FFB800 0%, rgba(255,184,0,0.3) 70%); box-shadow: 0 0 15px #FFB800; }
    .particle:nth-child(3) { animation-delay: 1s; }
    .particle:nth-child(4) { animation-delay: 1.5s; background: radial-gradient(circle, #00FF94 0%, rgba(0,255,148,0.3) 70%); box-shadow: 0 0 15px #00FF94; }
    .particle:nth-child(5) { animation-delay: 2s; }
    .particle:nth-child(6) { animation-delay: 2.5s; background: radial-gradient(circle, #FF2B2B 0%, rgba(255,43,43,0.3) 70%); box-shadow: 0 0 15px #FF2B2B; }
    .particle:nth-child(7) { animation-delay: 3s; }
    .particle:nth-child(8) { animation-delay: 3.5s; background: radial-gradient(circle, #FFB800 0%, rgba(255,184,0,0.3) 70%); box-shadow: 0 0 15px #FFB800; }
    
    /* ========== CONNECTING LINES ========== */
    .flow-line {
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, 
            rgba(0,255,255,0.3) 0%, 
            rgba(0,255,148,0.5) 50%, 
            rgba(255,184,0,0.3) 100%);
        z-index: 1;
    }
    
    /* ========== RISK DRIP ANIMATION ========== */
    @keyframes dripDown {
        0% { transform: translateY(0) scale(1); opacity: 1; }
        50% { transform: translateY(40px) scale(0.9); opacity: 0.8; }
        100% { transform: translateY(80px) scale(0.7); opacity: 0; }
    }
    
    .drip-container {
        position: relative;
        height: 100px;
        margin-top: 20px;
    }
    
    .risk-drip {
        position: absolute;
        width: 16px;
        height: 16px;
        border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
        animation: dripDown 2s ease-in infinite;
    }
    
    .drip-high { background: #FF2B2B; box-shadow: 0 0 10px #FF2B2B; animation-delay: 0s; left: 10%; }
    .drip-medium { background: #FFB800; box-shadow: 0 0 10px #FFB800; animation-delay: 0.4s; left: 30%; }
    .drip-low { background: #00FF94; box-shadow: 0 0 10px #00FF94; animation-delay: 0.8s; left: 50%; }
    .drip-clean { background: #444444; box-shadow: 0 0 10px #444444; animation-delay: 1.2s; left: 70%; }
    .drip-high-2 { background: #FF2B2B; box-shadow: 0 0 10px #FF2B2B; animation-delay: 1.6s; left: 90%; }
    
    /* ========== KPI CARDS ========== */
    .kpi-row {
        display: flex;
        gap: 15px;
        margin: 20px 0;
    }
    
    .kpi-card {
        flex: 1;
        background: linear-gradient(135deg, #0a0a0f, #12121a);
        border: 1px solid #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        border-color: #00FFFF;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.15);
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00FFFF;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }
    
    .kpi-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 5px;
    }
    
    /* ========== INDICATOR PANEL ========== */
    .indicator-panel {
        background: rgba(10, 10, 15, 0.9);
        border: 1px solid #1a1a2e;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .indicator-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .indicator-row:last-child {
        border-bottom: none;
    }
    
    .indicator-name {
        color: #888;
        font-size: 0.85rem;
    }
    
    .indicator-value {
        color: #00FF94;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .indicator-value.alert {
        color: #FF2B2B;
    }
    
    /* ========== GLASS CARD ========== */
    .glass-card {
        background: rgba(10, 10, 10, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid #222;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* ========== LEGEND ========== */
    .legend-container {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        color: #888;
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    /* ========== SECTION HEADER ========== */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #fff;
        margin: 30px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(0, 255, 255, 0.2);
    }
    
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data(ttl=30)
def load_simulation_results():
    """Load simulation pipeline results."""
    if os.path.exists('simulation_pipeline_results.csv'):
        return pd.read_csv('simulation_pipeline_results.csv')
    return None

@st.cache_data(ttl=30)
def load_risk_scores():
    """Load consolidated risk scores."""
    if os.path.exists('consolidated_risk_scores.csv'):
        return pd.read_csv('consolidated_risk_scores.csv')
    return None

@st.cache_resource
def get_logger():
    return get_metrics_logger()

def get_kpi_stats(minutes=30):
    """Get KPI stats from metrics logger."""
    try:
        metrics = get_logger()
        return metrics.get_kpi_stats(minutes=minutes)
    except Exception:
        return {
            'total_accounts': 0,
            'total_transactions': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0,
            'clean_count': 0,
            'avg_latency_ms': 0
        }

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🔬 AML Pipeline Visualizer</h1>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9326/9326938.png", width=50)
    st.markdown("### Pipeline Controls")
    
    show_animation = st.checkbox("Show Animation", value=True)
    show_indicators = st.checkbox("Show Phase Indicators", value=True)
    show_kpis = st.checkbox("Show KPI Metrics", value=True)
    show_drips = st.checkbox("Show Risk Drips", value=True)
    
    st.markdown("---")
    st.markdown("### Time Window")
    time_window = st.select_slider(
        "Minutes",
        options=[5, 15, 30, 60, 120],
        value=30
    )
    
    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ==================== LOAD DATA ====================
kpi_stats = get_kpi_stats(minutes=time_window)
sim_results = load_simulation_results()
risk_df = load_risk_scores()

# Calculate KPIs from simulation results if metrics.db has no recent data
if sim_results is not None and len(sim_results) > 0:
    # Count risk levels from CSV
    risk_counts = sim_results['risk_level'].value_counts().to_dict() if 'risk_level' in sim_results.columns else {}
    
    # If database has zeros, use CSV data
    if kpi_stats.get('total_accounts', 0) == 0:
        kpi_stats = {
            'total_accounts': len(sim_results),
            'total_transactions': len(sim_results),
            'high_risk_count': risk_counts.get('HIGH', 0),
            'medium_risk_count': risk_counts.get('MEDIUM', 0),
            'low_risk_count': risk_counts.get('LOW', 0),
            'clean_count': risk_counts.get('CLEAN', 0),
            'avg_latency_ms': 25.0  # Default estimate
        }

# Calculate phase indicators from results
phase_indicators = {
    'transactions': kpi_stats.get('total_transactions', 0),
    'accounts': kpi_stats.get('total_accounts', 0),
    'fan_in': 0,
    'fan_out': 0,
    'cycles': 0,
    'cyber_alerts': 0,
    'risk_escalations': 0,
    'concentration': 0,
    'emerging_links': 0,
    'high_risk': kpi_stats.get('high_risk_count', 0),
    'medium_risk': kpi_stats.get('medium_risk_count', 0),
    'low_risk': kpi_stats.get('low_risk_count', 0),
    'clean': kpi_stats.get('clean_count', 0)
}

# Parse signals from results if available
if sim_results is not None and 'signals' in sim_results.columns:
    all_signals = '|'.join(sim_results['signals'].dropna().astype(str))
    phase_indicators['fan_in'] = all_signals.count('fan_in_detected')
    phase_indicators['fan_out'] = all_signals.count('fan_out_detected')
    phase_indicators['cycles'] = all_signals.count('cycle_detection')
    phase_indicators['cyber_alerts'] = all_signals.count('cyber_')  # matches cyber_bruteforce, cyber_unknown, etc.
    phase_indicators['risk_escalations'] = all_signals.count('risk_escalation')
    phase_indicators['concentration'] = all_signals.count('temporal_concentration')
    phase_indicators['emerging_links'] = all_signals.count('emerging_link')

# ==================== KPI METRICS ROW ====================
if show_kpis:
    st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    risk_ratio = 0
    if kpi_stats['total_transactions'] > 0:
        risk_ratio = (kpi_stats['high_risk_count'] + kpi_stats['medium_risk_count']) / kpi_stats['total_transactions'] * 100
    
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-value">{kpi_stats['total_accounts']:,}</div>
            <div class="kpi-label">Analyzed Accounts</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color: #FF2B2B;">{risk_ratio:.1f}%</div>
            <div class="kpi-label">Risk Ratio</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpi_stats['total_transactions']:,}</div>
            <div class="kpi-label">Throughput</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color: #00FF94;">{kpi_stats['avg_latency_ms']:.0f}<span style="font-size:1rem">ms</span></div>
            <div class="kpi-label">Avg Latency</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== ANIMATED PIPELINE FLOW ====================
st.markdown('<div class="section-header">🚀 Pipeline Flow Animation</div>', unsafe_allow_html=True)

# Phase data with indicators
phases = [
    {"icon": "📥", "title": "Data Ingestion", "indicator": f"{phase_indicators['transactions']} txns"},
    {"icon": "🕸️", "title": "Graph Topology", "indicator": f"{phase_indicators['fan_in'] + phase_indicators['fan_out']} alerts"},
    {"icon": "🔐", "title": "Behavioral", "indicator": f"{phase_indicators['cyber_alerts']} cyber"},
    {"icon": "⏰", "title": "Temporal", "indicator": f"{phase_indicators['risk_escalations']} pred"},
    {"icon": "🧠", "title": "LSTM", "indicator": f"{phase_indicators['emerging_links']} links"},
    {"icon": "📈", "title": "Risk Score", "indicator": f"{phase_indicators['high_risk']} high"},
]

# Use Streamlit columns for phases (more reliable rendering)
phase_cols = st.columns(6)
for i, phase in enumerate(phases):
    with phase_cols[i]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0d0d15, #15151f); 
                    border: 1px solid #2a2a3e; border-radius: 12px; 
                    padding: 15px 10px; text-align: center;">
            <div style="font-size: 1.8rem;">{phase['icon']}</div>
            <div style="font-size: 0.75rem; font-weight: 600; color: #fff; margin: 5px 0;">{phase['title']}</div>
            <div style="font-size: 0.65rem; color: #00FFFF; font-weight: 700;
                        padding: 3px 8px; background: rgba(0, 255, 255, 0.1);
                        border-radius: 10px; display: inline-block;">{phase['indicator']}</div>
        </div>
        """, unsafe_allow_html=True)

# Animated particles track (below phases)
if show_animation:
    particles_html = ''.join([f'<div class="particle"></div>' for _ in range(8)])
    st.markdown(f"""
    <div class="pipeline-container">
        <div class="pipeline-track" style="height: 60px; margin-top: 10px;">
            <div class="flow-line"></div>
            {particles_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== PHASE INDICATORS DETAIL ====================
if show_indicators:
    st.markdown('<div class="section-header">📋 Phase Indicators Detail</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="indicator-panel">
            <h4 style="color:#00FFFF; margin-bottom:10px;">📥 Data Ingestion</h4>
            <div class="indicator-row">
                <span class="indicator-name">Transactions</span>
                <span class="indicator-value">{}</span>
            </div>
            <div class="indicator-row">
                <span class="indicator-name">Accounts</span>
                <span class="indicator-value">{}</span>
            </div>
        </div>
        """.format(phase_indicators['transactions'], phase_indicators['accounts']), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="indicator-panel">
            <h4 style="color:#00FFFF; margin-bottom:10px;">🕸️ Graph Topology</h4>
            <div class="indicator-row">
                <span class="indicator-name">Fan-In Patterns</span>
                <span class="indicator-value {}">{}</span>
            </div>
            <div class="indicator-row">
                <span class="indicator-name">Fan-Out Patterns</span>
                <span class="indicator-value {}">{}</span>
            </div>
            <div class="indicator-row">
                <span class="indicator-name">Cycles Detected</span>
                <span class="indicator-value {}">{}</span>
            </div>
        </div>
        """.format(
            'alert' if phase_indicators['fan_in'] > 0 else '', phase_indicators['fan_in'],
            'alert' if phase_indicators['fan_out'] > 0 else '', phase_indicators['fan_out'],
            'alert' if phase_indicators['cycles'] > 0 else '', phase_indicators['cycles']
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="indicator-panel">
            <h4 style="color:#00FFFF; margin-bottom:10px;">🔐 Behavioral Analysis</h4>
            <div class="indicator-row">
                <span class="indicator-name">Cyber Alerts</span>
                <span class="indicator-value {}">{}</span>
            </div>
            <div class="indicator-row">
                <span class="indicator-name">Login Anomalies</span>
                <span class="indicator-value">0</span>
            </div>
        </div>
        """.format('alert' if phase_indicators['cyber_alerts'] > 0 else '', phase_indicators['cyber_alerts']), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="indicator-panel">
            <h4 style="color:#00FFFF; margin-bottom:10px;">⏰ Temporal Prediction</h4>
            <div class="indicator-row">
                <span class="indicator-name">Risk Escalations</span>
                <span class="indicator-value">{}</span>
            </div>
            <div class="indicator-row">
                <span class="indicator-name">Concentration Bursts</span>
                <span class="indicator-value">{}</span>
            </div>
        </div>
        """.format(phase_indicators['risk_escalations'], phase_indicators['concentration']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="indicator-panel">
            <h4 style="color:#00FFFF; margin-bottom:10px;">🧠 LSTM Link Prediction</h4>
            <div class="indicator-row">
                <span class="indicator-name">Emerging Links</span>
                <span class="indicator-value">{}</span>
            </div>
            <div class="indicator-row">
                <span class="indicator-name">High Probability</span>
                <span class="indicator-value">-</span>
            </div>
        </div>
        """.format(phase_indicators['emerging_links']), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="indicator-panel">
            <h4 style="color:#00FFFF; margin-bottom:10px;">📈 Risk Consolidation</h4>
            <div class="indicator-row">
                <span class="indicator-name">🔴 High Risk</span>
                <span class="indicator-value alert">{}</span>
            </div>
            <div class="indicator-row">
                <span class="indicator-name">🟡 Medium Risk</span>
                <span class="indicator-value" style="color:#FFB800">{}</span>
            </div>
            <div class="indicator-row">
                <span class="indicator-name">🟢 Low Risk</span>
                <span class="indicator-value">{}</span>
            </div>
            <div class="indicator-row">
                <span class="indicator-name">⚪ Clean</span>
                <span class="indicator-value" style="color:#888">{}</span>
            </div>
        </div>
        """.format(
            phase_indicators['high_risk'],
            phase_indicators['medium_risk'],
            phase_indicators['low_risk'],
            phase_indicators['clean']
        ), unsafe_allow_html=True)

# ==================== RISK SCORE DRIPPING ANIMATION ====================
if show_drips:
    st.markdown('<div class="section-header">💧 Risk Score Output</div>', unsafe_allow_html=True)
    
    # Legend
    st.markdown("""
    <div class="legend-container">
        <div class="legend-item"><div class="legend-dot" style="background:#FF2B2B;"></div> High Risk</div>
        <div class="legend-item"><div class="legend-dot" style="background:#FFB800;"></div> Medium Risk</div>
        <div class="legend-item"><div class="legend-dot" style="background:#00FF94;"></div> Low Risk</div>
        <div class="legend-item"><div class="legend-dot" style="background:#444444;"></div> Clean</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dripping animation
    st.markdown("""
    <div class="pipeline-container">
        <div style="text-align:center; color:#888; margin-bottom:10px;">Risk Scores Dripping from Consolidation Phase</div>
        <div class="drip-container">
            <div class="risk-drip drip-high"></div>
            <div class="risk-drip drip-medium"></div>
            <div class="risk-drip drip-low"></div>
            <div class="risk-drip drip-clean"></div>
            <div class="risk-drip drip-high-2"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== RESULTS TABLE ====================
st.markdown('<div class="section-header">📊 Recent Risk Assessments</div>', unsafe_allow_html=True)

if sim_results is not None and len(sim_results) > 0:
    # Style the dataframe
    display_df = sim_results.head(15).copy()
    
    def style_risk_level(val):
        if val == 'HIGH':
            return f'color: {colors["high_risk"]}; font-weight: 600'
        elif val == 'MEDIUM':
            return f'color: {colors["medium_risk"]}; font-weight: 600'
        elif val == 'LOW':
            return f'color: {colors["low_risk"]}; font-weight: 600'
        return f'color: {colors["clean"]}'
    
    st.dataframe(
        display_df.style.map(style_risk_level, subset=['risk_level']) if 'risk_level' in display_df.columns else display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.ProgressColumn(
                "Risk Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            )
        } if 'score' in display_df.columns else None
    )
elif risk_df is not None and len(risk_df) > 0:
    st.dataframe(risk_df.head(15), use_container_width=True, hide_index=True)
else:
    st.info("No results found. Run `python pipeline_simulation.py` to generate data.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #555; font-size: 0.8rem;'>"
    "AML Pipeline Visualizer • Animated Edition • Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)
