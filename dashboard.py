"""
AML Detection Dashboard - Real-Time Monitoring

Streamlit-based interactive dashboard for monitoring AML inference API.
Displays real-time metrics, risk statistics, and investigation tools.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import sys
import os

# Ensure workspace imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.metrics_logger import get_metrics_logger

# Page configuration
st.set_page_config(
    page_title="AML Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize metrics logger
@st.cache_resource
def get_logger():
    return get_metrics_logger()

metrics = get_logger()

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
    }
    .medium-risk {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("🔍 AML Detection System - Real-Time Dashboard")
st.markdown("**Monitoring Multi-Engine Risk Detection Pipeline**")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    # Time window selector
    time_window = st.selectbox(
        "Time Window",
        [5, 15, 30, 60, 120],
        index=2,
        format_func=lambda x: f"Last {x} minutes"
    )
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", 2, 30, 5)
    
    # Manual refresh button
    if st.button("🔄 Refresh Now"):
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Dashboard Sections")
    st.markdown("- **Global Metrics**: System throughput")
    st.markdown("- **Risk Overview**: KPI statistics")
    st.markdown("- **Investigation**: Detailed logs")
    
    st.markdown("---")
    st.markdown("### 🎯 Risk Levels")
    st.markdown("🔴 **HIGH** (≥0.7): Block/Verify")
    st.markdown("🟡 **MEDIUM** (0.4-0.7): Monitor")
    st.markdown("🟢 **LOW** (0.0-0.4): Log")
    st.markdown("⚪ **CLEAN** (0.0): Allow")

# Fetch data
kpi_stats = metrics.get_kpi_stats(minutes=time_window)
engine_stats = metrics.get_engine_stats(minutes=time_window)
recent_inferences = metrics.get_recent_inferences(limit=100)
top_links = metrics.get_top_links(limit=10)

# ==================== SECTION A: GLOBAL INGESTION METRICS ====================
st.header("📥 Global Ingestion Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Accounts Scanned",
        f"{kpi_stats['total_accounts']:,}",
        delta=None,
        help="Unique accounts processed in time window"
    )

with col2:
    st.metric(
        "Live Transactions",
        f"{kpi_stats['total_transactions']:,}",
        delta=None,
        help="Total transactions analyzed"
    )

with col3:
    total_events = sum([len(json.loads(inf.get('component_scores', '{}')).keys()) 
                       for inf in recent_inferences if inf.get('component_scores')])
    st.metric(
        "Cyber Events",
        f"{total_events:,}",
        delta=None,
        help="Event sequences processed"
    )

with col4:
    st.metric(
        "Avg Latency",
        f"{kpi_stats['avg_latency_ms']:.1f} ms",
        delta=None,
        help="Average inference time"
    )

st.markdown("### 🔧 Engine Throughput")

# Engine activity table
if engine_stats:
    engine_df = pd.DataFrame(engine_stats)
    engine_df['avg_latency_ms'] = engine_df['avg_latency_ms'].round(2)
    engine_df['max_latency_ms'] = engine_df['max_latency_ms'].round(2)
    engine_df.columns = ['Engine', 'Operations', 'Avg Latency (ms)', 'Max Latency (ms)']
    
    st.dataframe(
        engine_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("⏳ No engine activity in selected time window")

# Latency trend chart
st.markdown("### ⏱️ Latency Monitor")

latency_trends = metrics.get_latency_trends(limit=50)
if latency_trends:
    latency_df = pd.DataFrame(latency_trends)
    latency_df['timestamp'] = pd.to_datetime(latency_df['timestamp'])
    
    fig_latency = px.line(
        latency_df,
        x='timestamp',
        y='latency_ms',
        color='engine',
        title='Inference Latency by Engine',
        labels={'latency_ms': 'Latency (ms)', 'timestamp': 'Time', 'engine': 'Engine'},
        line_shape='spline'
    )
    fig_latency.update_layout(height=350)
    st.plotly_chart(fig_latency, use_container_width=True)
else:
    st.info("⏳ No latency data available")

st.markdown("---")

# ==================== SECTION B: RISK OVERVIEW & KEY STATISTICS ====================
st.header("📊 Risk Overview & Key Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "🔴 High Risk",
        kpi_stats['high_risk_count'],
        help="Scores ≥ 0.7"
    )

with col2:
    st.metric(
        "🟡 Medium Risk",
        kpi_stats['medium_risk_count'],
        help="Scores 0.4-0.7"
    )

with col3:
    st.metric(
        "🟢 Low Risk",
        kpi_stats['low_risk_count'],
        help="Scores 0.0-0.4"
    )

with col4:
    st.metric(
        "⚪ Clean",
        kpi_stats['clean_count'],
        help="Score = 0.0"
    )

with col5:
    total_alerts = kpi_stats['high_risk_count'] + kpi_stats['medium_risk_count']
    st.metric(
        "🚨 Active Alerts",
        total_alerts,
        help="High + Medium risk"
    )

# Risk level distribution chart
st.markdown("### 🎯 Risk Level Distribution")

risk_data = {
    'Risk Level': ['High', 'Medium', 'Low', 'Clean'],
    'Count': [
        kpi_stats['high_risk_count'],
        kpi_stats['medium_risk_count'],
        kpi_stats['low_risk_count'],
        kpi_stats['clean_count']
    ],
    'Color': ['#c62828', '#ef6c00', '#2e7d32', '#757575']
}

fig_donut = go.Figure(data=[go.Pie(
    labels=risk_data['Risk Level'],
    values=risk_data['Count'],
    hole=0.4,
    marker=dict(colors=risk_data['Color']),
    textinfo='label+value+percent',
    textposition='outside'
)])

fig_donut.update_layout(
    title='Risk Distribution',
    height=400,
    showlegend=True
)

st.plotly_chart(fig_donut, use_container_width=True)

# Financial impact estimate
st.markdown("### 💰 Financial Impact Estimate")

if recent_inferences:
    # Calculate total amount at risk from recent inferences
    total_at_risk = 0
    for inf in recent_inferences:
        if inf.get('risk_level') in ['HIGH', 'MEDIUM']:
            # Try to extract amount from component scores
            try:
                comp_scores = json.loads(inf.get('component_scores', '{}'))
                # This is a rough estimate - in production, you'd extract actual transaction amounts
                total_at_risk += 1000  # Placeholder amount
            except:
                pass
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated Amount at Risk", f"${total_at_risk:,}")
    with col2:
        st.metric("Suspected Accounts", kpi_stats['high_risk_count'])
    with col3:
        avg_risk = kpi_stats['total_transactions'] / max(1, kpi_stats['total_accounts'])
        st.metric("Avg Transactions/Account", f"{avg_risk:.1f}")

st.markdown("---")

# ==================== SECTION C: INTERACTIVE INVESTIGATION AREA ====================
st.header("🔬 Interactive Investigation Area")

tab1, tab2, tab3 = st.tabs(["📋 Recent Inferences", "🔗 Link Predictions", "📄 Raw Response"])

with tab1:
    st.markdown("### Recent Risk Assessments")
    
    if recent_inferences:
        # Convert to DataFrame
        inf_df = []
        for inf in recent_inferences[:50]:
            try:
                comp_scores = json.loads(inf.get('component_scores', '{}'))
                inf_df.append({
                    'Timestamp': inf.get('timestamp', 'N/A')[:19],
                    'Account ID': inf.get('account_id', 'N/A'),
                    'Endpoint': inf.get('endpoint', 'N/A'),
                    'Risk Score': round(inf.get('risk_score', 0.0) or 0.0, 3),
                    'Risk Level': inf.get('risk_level', 'N/A'),
                    'Latency (ms)': round(inf.get('latency_ms', 0.0) or 0.0, 1),
                    'Status': inf.get('status', 'N/A')
                })
            except Exception as e:
                continue
        
        if inf_df:
            df = pd.DataFrame(inf_df)
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                risk_filter = st.multiselect(
                    "Filter by Risk Level",
                    ['HIGH', 'MEDIUM', 'LOW', 'CLEAN'],
                    default=['HIGH', 'MEDIUM', 'LOW', 'CLEAN']
                )
            with col2:
                status_filter = st.multiselect(
                    "Filter by Status",
                    ['success', 'error'],
                    default=['success', 'error']
                )
            
            # Apply filters
            filtered_df = df[
                (df['Risk Level'].isin(risk_filter)) &
                (df['Status'].isin(status_filter))
            ]
            
            # Style the dataframe
            def highlight_risk(row):
                if row['Risk Level'] == 'HIGH':
                    return ['background-color: #ffebee'] * len(row)
                elif row['Risk Level'] == 'MEDIUM':
                    return ['background-color: #fff3e0'] * len(row)
                elif row['Risk Level'] == 'LOW':
                    return ['background-color: #e8f5e9'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                filtered_df.style.apply(highlight_risk, axis=1),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "aml_inferences.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("⏳ No valid inference data to display")
    else:
        st.info("⏳ No recent inferences in selected time window")

with tab2:
    st.markdown("### Top 10 Emerging Links")
    
    if top_links:
        link_df = pd.DataFrame(top_links)
        link_df['timestamp'] = pd.to_datetime(link_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        link_df['probability'] = link_df['probability'].round(3)
        link_df.columns = ['ID', 'Timestamp', 'Source Account', 'Target Account', 'Probability', 'Risk Score']
        
        st.dataframe(
            link_df[['Timestamp', 'Source Account', 'Target Account', 'Probability', 'Risk Score']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("⏳ No link predictions available")

with tab3:
    st.markdown("### Raw JSON Response Inspector")
    
    if recent_inferences:
        # Account selector
        account_ids = [inf.get('account_id', 'N/A') for inf in recent_inferences[:50]]
        unique_accounts = list(set([acc for acc in account_ids if acc != 'N/A']))
        
        if unique_accounts:
            selected_account = st.selectbox(
                "Select Account to Inspect",
                unique_accounts
            )
            
            # Find matching inference
            matching_infs = [inf for inf in recent_inferences 
                           if inf.get('account_id') == selected_account]
            
            if matching_infs:
                selected_inf = matching_infs[0]
                
                # Display formatted JSON
                json_response = {
                    'account_id': selected_inf.get('account_id'),
                    'timestamp': selected_inf.get('timestamp'),
                    'endpoint': selected_inf.get('endpoint'),
                    'risk_score': selected_inf.get('risk_score'),
                    'risk_level': selected_inf.get('risk_level'),
                    'component_scores': json.loads(selected_inf.get('component_scores', '{}')),
                    'latency_ms': selected_inf.get('latency_ms'),
                    'status': selected_inf.get('status')
                }
                
                st.json(json_response)
                
                # Copy to clipboard
                st.code(json.dumps(json_response, indent=2), language='json')
        else:
            st.info("⏳ No accounts with valid IDs found")
    else:
        st.info("⏳ No recent inferences available")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"**Dashboard last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"**Time Window:** Last {time_window} minutes | "
    f"**Total Records:** {len(recent_inferences)}"
)
