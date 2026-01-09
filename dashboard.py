"""
AML Detection Dashboard - Premium Edition
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import sys
import os

# Ensure workspace imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.metrics_logger import get_metrics_logger

# ==================== PAGE CONFIG & STYLING ====================
st.set_page_config(
    page_title="FRAUD.GUARD | AML Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Theme Colors
colors = {
    'background': '#0E1117',
    'card_bg': 'rgba(255, 255, 255, 0.05)',
    'text': '#E0E0E0',
    'accent': '#00FF94',  # Neon Green
    'high_risk': '#FF2B2B', # Neon Red
    'medium_risk': '#FFB800', # Neon Orange
    'low_risk': '#00FF94',
    'clean': '#333333'
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Reset */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #050505 60%);
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(20, 20, 25, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8888aa;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
    }
    
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Status Indicators */
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 10px currentColor;
    }
    
    /* Tables */
    .stDataFrame {
        border: none !important;
    }
    
    .dataframe {
        background-color: transparent !important;
        color: #cccccc !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_resource
def get_logger():
    return get_metrics_logger()

metrics = get_logger()

# ==================== SIDEBAR (Controls) ====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9326/9326938.png", width=60)
    st.markdown("### SYSTEM CONTROLS")
    
    time_window = st.select_slider(
        "Time Window (Min)",
        options=[5, 15, 30, 60, 120, 360],
        value=30
    )
    
    auto_refresh = st.toggle("Live Updates", value=True)
    refresh_interval = st.number_input("Refresh Rate (s)", 2, 60, 5)
    
    if st.button("Manual Refresh", type="primary"):
        st.rerun()
        
    st.divider()
    
    st.markdown("### SYSTEM STATUS")
    col_sys_1, col_sys_2 = st.columns(2)
    with col_sys_1:
        st.metric("CPU", "12%", "-2%")
    with col_sys_2:
        st.metric("Mem", "4.2GB", "+0.1%")
        
    st.markdown("Build v2.4.0-rc1")

# ==================== MAIN LAYOUT ====================

# HEADER
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# 🛡️ FRAUD.GUARD <span style='font-weight:300; opacity: 0.5'>| INSIGHTS</span>", unsafe_allow_html=True)
with col_status:
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 10px;">
            <span class="status-dot" style="color: #00FF94;"></span>
            <span style="color: #00FF94; font-weight: 600;">SYSTEM OPERATIONAL</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

# Fetch Data
kpi_stats = metrics.get_kpi_stats(minutes=time_window)
recent_inferences = metrics.get_recent_inferences(limit=100)
engine_stats = metrics.get_engine_stats(minutes=time_window)

# --- KPI ROW ---
col1, col2, col3, col4 = st.columns(4)

def metric_card(title, value, delta=None, color="#ffffff"):
    delta_html = ""
    if delta:
        delta_color = "#00FF94" if delta.startswith("+") else "#FF2B2B"
        delta_html = f'<div class="metric-delta" style="color: {delta_color}">{delta}</div>'
    
    return f"""
    <div class="glass-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value" style="color: {color}">{value}</div>
        {delta_html}
    </div>
    """

with col1:
    st.markdown(metric_card("Analyzed Accounts", f"{kpi_stats['total_accounts']:,}", "Active"), unsafe_allow_html=True)

with col2:
    risk_ratio = 0
    if kpi_stats['total_transactions'] > 0:
        risk_ratio = (kpi_stats['high_risk_count'] + kpi_stats['medium_risk_count']) / kpi_stats['total_transactions'] * 100
    st.markdown(metric_card("Risk Ratio", f"{risk_ratio:.1f}%", f"{kpi_stats['high_risk_count']} Alerts", colors['high_risk']), unsafe_allow_html=True)

with col3:
    st.markdown(metric_card("Throughput", f"{kpi_stats['total_transactions']:,}", "Transactions"), unsafe_allow_html=True)

with col4:
    st.markdown(metric_card("Avg Latency", f"{kpi_stats['avg_latency_ms']:.0f}<span style='font-size:1rem'>ms</span>", None, colors['accent']), unsafe_allow_html=True)

# --- CHARTS ROW ---
col_chart_1, col_chart_2 = st.columns([2, 1])

with col_chart_1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🕸️ Latency & Network Activity")
    
    latency_trends = metrics.get_latency_trends(limit=100)
    if latency_trends:
        lat_df = pd.DataFrame(latency_trends)
        
        fig = go.Figure()
        
        # Add a glowy line for each engine
        for engine in lat_df['engine'].unique():
            eng_data = lat_df[lat_df['engine'] == engine]
            fig.add_trace(go.Scatter(
                x=eng_data['timestamp'],
                y=eng_data['latency_ms'],
                name=engine,
                mode='lines',
                line=dict(width=3),
                fill='tozeroy',
                fillcolor=f"rgba({100 if 'graph' in engine else 0}, {255 if 'forest' in engine else 100}, 200, 0.1)"
            ))
            
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'family': "Outfit"},
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No latency data available")
    st.markdown('</div>', unsafe_allow_html=True)

with col_chart_2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Risk Distribution")
    
    labels = ['High', 'Medium', 'Low', 'Clean']
    values = [
        kpi_stats['high_risk_count'],
        kpi_stats['medium_risk_count'],
        kpi_stats['low_risk_count'],
        kpi_stats['clean_count']
    ]
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=[colors['high_risk'], colors['medium_risk'], colors['low_risk'], colors['card_bg']]),
        textinfo='label+percent',
        showlegend=False
    )])
    
    fig_donut.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Outfit"},
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        annotations=[dict(text=str(sum(values)), x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- INFERNECE FEED ROW ---
st.markdown("### 🚨 Live Inference Feed")

if recent_inferences:
    inf_data = []
    for inf in recent_inferences[:20]:
        inf_data.append({
            "Time": inf.get('timestamp', '')[11:19],
            "Account": inf.get('account_id'),
            "Risk Score": inf.get('risk_score'),
            "Level": inf.get('risk_level'),
            "Latency": f"{inf.get('latency_ms', 0):.1f}ms"
        })
    
    df = pd.DataFrame(inf_data)
    
    # Custom coloring for table using Pandas Styler
    def color_risk(val):
        color = colors['text']
        if val == 'HIGH': color = colors['high_risk']
        elif val == 'MEDIUM': color = colors['medium_risk']
        elif val == 'LOW': color = colors['low_risk']
        return f'color: {color}; font-weight: 600'

    st.dataframe(
        df.style.map(color_risk, subset=['Level']),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            )
        }
    )
else:
    st.info("Waiting for live data...")

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
