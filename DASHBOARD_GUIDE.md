# AML Dashboard Quick Start Guide

**Real-Time Monitoring Dashboard for AML Detection System**

## Overview

The AML Dashboard is a Streamlit-based web interface that provides real-time monitoring of your inference API. It visualizes:

- **Global Ingestion Metrics**: Throughput, latency, and engine activity
- **Risk Overview**: KPI statistics and risk distribution
- **Interactive Investigation**: Detailed logs and raw responses

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Transaction Simulator (transaction_simulator.py)      │
│  • Generates synthetic transactions                    │
│  • Sends to inference API at configurable rate         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│  Flask Inference API (src/inference_api.py)            │
│  • Processes transactions with 5 ML models             │
│  • Logs metrics to SQLite database                     │
│  • Returns JSON risk scores                            │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│  Metrics Database (metrics.db)                         │
│  • SQLite database with 4 tables                       │
│  • Stores inference logs, engine stats, KPIs          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│  Streamlit Dashboard (dashboard.py)                    │
│  • Real-time web UI on port 8501                       │
│  • Auto-refresh every 5 seconds                        │
│  • Interactive charts and tables                       │
└─────────────────────────────────────────────────────────┘
```

## Quick Start (3 Steps)

### Step 1: Start the Inference API

```bash
python -m src.inference_api
```

**Expected output:**
```
Loading inference models...
✓ GBDT model loaded
✓ Sequence Detector model loaded
✓ LSTM Link Predictor model loaded
✓ Risk Consolidator loaded
Inference engine ready. 4 models loaded.

 * Serving Flask app 'inference_api'
 * Running on http://127.0.0.1:5000
```

**Keep this terminal open!** The API must be running for the dashboard to work.

### Step 2: Start the Transaction Simulator (New Terminal)

```bash
python transaction_simulator.py --rate 2.0
```

**Expected output:**
```
🚀 Starting transaction simulator...
📡 API URL: http://localhost:5000/score/consolidate
⚡ Rate: 2.0 transactions/second
⏱️  Duration: Infinite (Ctrl+C to stop)

🟢 ACC_3421 | Risk: 0.125 (LOW) | Profile: normal | Total: 1
⚪ ACC_7892 | Risk: 0.000 (CLEAN) | Profile: normal | Total: 2
🟡 ACC_4561 | Risk: 0.543 (MEDIUM) | Profile: suspicious | Total: 3
🔴 ACC_2109 | Risk: 0.823 (HIGH) | Profile: high_risk | Total: 4
```

**Keep this running!** It generates live data for the dashboard.

### Step 3: Launch the Dashboard (New Terminal)

```bash
streamlit run dashboard.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

**Open your browser** to http://localhost:8501

## Dashboard Features

### 📥 Section A: Global Ingestion Metrics

**Top Metrics Cards:**
- **Total Accounts Scanned**: Unique accounts processed
- **Live Transactions**: Total transactions analyzed
- **Cyber Events**: Event sequences processed
- **Avg Latency**: Average inference time in milliseconds

**Engine Throughput Table:**
Shows activity for each detection engine:
- GNN (Graph Neural Network)
- GBDT (Gradient Boosting)
- Sequence Detector (LSTM)
- LSTM Link Predictor
- Consolidator

**Latency Monitor Chart:**
Real-time line chart showing inference latency by engine over time.

### 📊 Section B: Risk Overview & Key Statistics

**Risk Level Cards:**
- 🔴 **High Risk** (Score ≥ 0.7): Immediate action required
- 🟡 **Medium Risk** (Score 0.4-0.7): Monitor closely
- 🟢 **Low Risk** (Score 0.0-0.4): Log for review
- ⚪ **Clean** (Score 0.0): No risk detected
- 🚨 **Active Alerts**: High + Medium combined

**Risk Distribution Donut Chart:**
Visual breakdown of risk levels as percentages.

**Financial Impact Estimate:**
- Estimated Amount at Risk
- Suspected Accounts count
- Avg Transactions per Account

### 🔬 Section C: Interactive Investigation Area

**Tab 1: Recent Inferences**
- Table of last 50 inferences
- Color-coded by risk level
- Filterable by risk level and status
- Downloadable as CSV
- Shows: Timestamp, Account ID, Risk Score, Latency

**Tab 2: Link Predictions**
- Top 10 emerging links predicted by LSTM
- Source → Target account pairs
- Formation probability scores

**Tab 3: Raw Response Inspector**
- Select any account to view full JSON response
- See component scores breakdown
- Inspect model outputs
- Copy formatted JSON

## Dashboard Controls (Sidebar)

**Time Window:**
- Last 5 minutes
- Last 15 minutes
- Last 30 minutes (default)
- Last 60 minutes
- Last 120 minutes

**Auto-Refresh:**
- Toggle on/off
- Configurable interval (2-30 seconds, default: 5s)

**Manual Refresh:**
- Click "🔄 Refresh Now" button

## Command-Line Options

### Transaction Simulator Options

```bash
# Default (2 transactions/second, infinite duration)
python transaction_simulator.py

# Custom rate (5 transactions/second)
python transaction_simulator.py --rate 5.0

# Custom duration (run for 60 seconds)
python transaction_simulator.py --duration 60

# Custom API URL
python transaction_simulator.py --url http://api.example.com:5000/score/consolidate

# Combine options
python transaction_simulator.py --rate 10.0 --duration 120
```

### Dashboard Options

```bash
# Default port 8501
streamlit run dashboard.py

# Custom port
streamlit run dashboard.py --server.port 8080

# Disable file watcher (for production)
streamlit run dashboard.py --server.fileWatcherType none

# Set theme
streamlit run dashboard.py --theme.base light
```

## Typical Workflow

1. **Start API** (Terminal 1):
   ```bash
   python -m src.inference_api
   ```

2. **Start Simulator** (Terminal 2):
   ```bash
   python transaction_simulator.py --rate 3.0
   ```
   
3. **Launch Dashboard** (Terminal 3):
   ```bash
   streamlit run dashboard.py
   ```

4. **View Dashboard** in browser at http://localhost:8501

5. **Monitor** real-time metrics as transactions flow through

6. **Investigate** suspicious accounts in the Investigation tab

7. **Export** data using CSV download button

## Performance Tips

### For High-Volume Scenarios

**Increase Simulator Rate:**
```bash
python transaction_simulator.py --rate 10.0
```

**Optimize Dashboard Refresh:**
- Reduce auto-refresh interval to 2 seconds for faster updates
- Increase time window to 60 minutes to see more data

**Database Cleanup:**
The metrics logger automatically cleans data older than 7 days. To manually clear:
```python
from src.metrics_logger import get_metrics_logger
metrics = get_metrics_logger()
metrics.clear_old_data(days=1)  # Clear data older than 1 day
```

### For Demo/Presentation

**Moderate Rate:**
```bash
python transaction_simulator.py --rate 2.0
```

**Shorter Time Window:**
- Set dashboard to "Last 15 minutes"
- Auto-refresh every 5 seconds
- Focus on Recent Inferences tab

**Generate High-Risk Events:**
The simulator automatically generates:
- 70% normal transactions (low/clean risk)
- 20% suspicious transactions (medium risk)
- 10% high-risk transactions (high risk)

## Troubleshooting

### Issue: Dashboard shows "No data"

**Solution:**
1. Check that Flask API is running (http://localhost:5000/health should return JSON)
2. Check that simulator is sending transactions (you should see colored output)
3. Wait 10-15 seconds for data to populate
4. Click "🔄 Refresh Now" button

### Issue: "Connection refused" error

**Solution:**
1. Ensure Flask API is running first
2. Check API is on correct port (default: 5000)
3. Update simulator URL if needed:
   ```bash
   python transaction_simulator.py --url http://localhost:5000/score/consolidate
   ```

### Issue: Dashboard is slow

**Solution:**
1. Reduce time window (e.g., last 5 minutes)
2. Increase refresh interval to 10-15 seconds
3. Reduce simulator rate:
   ```bash
   python transaction_simulator.py --rate 1.0
   ```

### Issue: Models not loading

**Solution:**
1. Train models first:
   ```bash
   python main.py
   ```
2. Check models/ directory contains:
   - lgb_model.txt
   - lstm_link_predictor.pt
   - consolidation_config.json
   - (optional) gnn_model.pt, sequence_detector_model.pt

### Issue: Streamlit not found

**Solution:**
```bash
pip install streamlit plotly
```

## Database Schema

The `metrics.db` SQLite database contains:

**Table: inference_logs**
- id, timestamp, account_id, endpoint, engine
- latency_ms, risk_score, risk_level, component_scores
- status, error

**Table: engine_stats**
- id, timestamp, engine, operation
- count, latency_ms

**Table: kpi_aggregates**
- id, timestamp, total_accounts, total_transactions
- high/medium/low/clean_count, total_amount_at_risk, avg_latency_ms

**Table: link_predictions**
- id, timestamp, source_account, target_account
- probability, risk_score

## Production Deployment

### Docker Deployment

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./metrics.db:/app/metrics.db
    command: python -m src.inference_api

  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./metrics.db:/app/metrics.db
    command: streamlit run dashboard.py --server.port 8501
    depends_on:
      - api

  simulator:
    build: .
    command: python transaction_simulator.py --rate 5.0
    depends_on:
      - api
```

Run: `docker-compose up`

### Cloud Deployment

**AWS:**
- Deploy API to ECS/Fargate or EC2
- Use RDS for metrics database (upgrade from SQLite)
- Host dashboard on Elastic Beanstalk or EC2
- Use Application Load Balancer for scaling

**GCP:**
- Deploy API to Cloud Run or GKE
- Use Cloud SQL for metrics
- Host dashboard on Cloud Run
- Use Cloud Load Balancing

**Azure:**
- Deploy API to Container Instances or AKS
- Use Azure SQL Database
- Host dashboard on App Service
- Use Azure Load Balancer

## Next Steps

1. ✅ **Customize Metrics**: Add custom KPIs in metrics_logger.py
2. ✅ **Add Alerts**: Integrate with email/Slack for high-risk events
3. ✅ **Export Reports**: Add scheduled CSV/PDF report generation
4. ✅ **Historical Analysis**: Add time-series analysis charts
5. ✅ **User Authentication**: Add login/RBAC for production use

## Support

For issues or questions:
- Check INFERENCE_API_GUIDE.md for API documentation
- Review SYSTEM_ARCHITECTURE.md for system overview
- Inspect metrics.db with DB Browser for SQLite

---

**Dashboard Version**: 1.0.0  
**Last Updated**: January 9, 2026  
**Status**: ✅ Production Ready
