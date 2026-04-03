# AML Topology Detection System — Run Manual

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Quick Start (5 minutes)](#3-quick-start-5-minutes)
4. [Running the Unified Dashboard](#4-running-the-unified-dashboard)
5. [Running the Full Detection Pipeline](#5-running-the-full-detection-pipeline)
6. [Training the Models](#6-training-the-models)
7. [Running Inference on Real Data](#7-running-inference-on-real-data)
8. [Starting the REST API](#8-starting-the-rest-api)
9. [Running the Live Demo Stack](#9-running-the-live-demo-stack)
10. [Adversarial Retraining](#10-adversarial-retraining)
11. [Configuration Reference](#11-configuration-reference)
12. [Output Files Reference](#12-output-files-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Component Reference Card](#14-component-reference-card)

---

## 1. Prerequisites

### Required

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | 3.12.x confirmed working |
| pip | ≥ 23 | `python -m pip install --upgrade pip` |
| 4 GB RAM | — | 8 GB recommended when running all models |
| 2 GB disk | — | For model artefacts and SQLite logs |

### Optional

| Requirement | Purpose |
|---|---|
| CUDA-capable GPU | Faster adversarial PPO training (`--device cuda`) |
| Redis | Distributed GNN embedding cache (file cache works without it) |
| `psutil` | Live CPU / RAM display in the dashboard sidebar |

---

## 2. Installation

### Step 1 — Clone / navigate to the project

```bash
cd aml-topology
```

### Step 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `networkx`, `pandas`, `numpy`, `matplotlib`, `torch`, `lightgbm`,
`flask`, `streamlit>=1.37`, `plotly`, `psutil`, `redis`, `gymnasium`, and others.

### Step 4 — Verify the installation

```bash
python -c "import torch, streamlit, lightgbm, plotly, networkx, flask; print('All OK')"
```

Expected output: `All OK`

---

## 3. Quick Start (5 minutes)

Run these three commands in order to go from nothing to a live dashboard.

```bash
# 1. Train all models (generates models/ artefacts)
python train.py

# 2. Run the full detection pipeline (generates CSV results)
python main.py

# 3. Launch the unified dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **If models are already trained** (the `models/` directory contains `.pt` and `.txt` files),
> skip step 1 and go straight to step 2 or 3.

---

## 4. Running the Unified Dashboard

The dashboard is a multi-page Streamlit application with four views.

### Launch

```bash
streamlit run app.py
```

Default URL: **http://localhost:8501**

### Pages

| Page | Icon | What it shows |
|---|---|---|
| Monitor | 🛡️ | Live inference feed, KPI cards, latency chart, risk distribution donut, account drill-down, alert management, manual scoring form |
| Pipeline | 🔬 | All 8 detection phases with real signal counts, risk distribution bar chart, full results table |
| Graph Explorer | 🕸️ | Interactive Plotly transaction network coloured by risk tier |
| Account Detail | 🔍 | Per-account signal breakdown, transaction timeline, counterparty graph, alert status |

### Dashboard controls (sidebar)

| Control | Page | Purpose |
|---|---|---|
| Rolling window | Monitor, Pipeline | How many minutes of live data to show |
| From / To date | Monitor, Account Detail | Historical date range for charts and timeline |
| Live Updates toggle | Monitor, Pipeline | Enable / disable auto-refresh (every 5 s / 10 s) |
| Risk Tier filter | Graph Explorer | Show only selected risk tiers in the network |
| Max nodes slider | Graph Explorer | Cap on displayed nodes (highest-risk first) |
| Layout seed | Graph Explorer | Changes the spring-layout arrangement |
| Highlight Account | Graph Explorer | Draws a white border around a specific node |

### Alert management workflow

1. Go to **Monitor → Account Drill-Down** or **Account Detail** page.
2. Select an account from the dropdown.
3. Change **Status** to one of: `Unreviewed` / `Investigating` / `False Positive` / `Escalated`.
4. Add **Investigation Notes** if needed.
5. Click **💾 Save Status**.

States are persisted to `alert_state.db` and survive page reloads and restarts.

### Manual transaction scoring

On the **Monitor** page, expand **🎯 Manual Transaction Score**:

1. Enter Source Account, Target Account, Amount, and Channel.
2. Click **Score Transaction**.
3. The live inference API at `http://localhost:5000` is called; the risk level and score are shown inline.

> The inference API must be running (`python -m src.inference_api`). See [Section 8](#8-starting-the-rest-api).

---

## 5. Running the Full Detection Pipeline

`main.py` runs the complete 5-phase detection pipeline end-to-end using synthetic data.

```bash
python main.py
```

### What it does

| Phase | Action |
|---|---|
| 1 — Data Ingestion | Generates synthetic transaction graph with injected fan-in, fan-out, and cycle patterns |
| 2 — Spatial Analysis | Fan-in, fan-out, cycle detection, centrality calculation |
| 3 — Temporal Analysis | Baselines, volume acceleration, behavioural shift, risk escalation forecast |
| 4 — ML Analysis | LSTM link prediction, GBDT scoring, GNN node classification |
| 5 — Visualisation | Graph PNG saved, risk scores ranked and printed |

### Outputs

| File | Contents |
|---|---|
| `aml_network_graph.png` | Static transaction network image |
| `simulation_pipeline_results.csv` | Per-account risk scores and signal flags |
| `consolidated_risk_scores.csv` | Final ranked risk scores with tier labels |
| `metrics.db` | SQLite log of inference calls (read by the dashboard) |

---

## 6. Training the Models

`train.py` trains all four ML models. Run this once before inference.

### Train everything

```bash
python train.py
```

### Train a specific model

```bash
python train.py --model lstm          # LSTM link predictor only
python train.py --model gbdt          # LightGBM GBDT only
python train.py --model gnn           # Graph Neural Network only
python train.py --model sequence      # Sequence detector only
python train.py --model adversarial   # PPO adversarial agent only
python train.py --model config        # Regenerate consolidation_config.json only
```

### Common options

| Flag | Default | Description |
|---|---|---|
| `--epochs N` | 15 | Training epochs for LSTM and GNN |
| `--samples N` | 5000 | Number of synthetic samples for GBDT training |
| `--adversarial N` | 0 | Episodes for adversarial PPO training (0 = skip) |
| `--device` | `cpu` | `cpu` or `cuda` (for adversarial training) |

### Examples

```bash
# Longer training run for better LSTM accuracy
python train.py --model lstm --epochs 50

# Larger GBDT dataset
python train.py --model gbdt --samples 20000

# Train all models with 100 adversarial episodes on GPU
python train.py --adversarial 100 --device cuda

# Quick smoke-test run
python train.py --epochs 3 --samples 500
```

### Model output files

After training, the `models/` directory contains:

| File | Model | Description |
|---|---|---|
| `lstm_link_predictor.pt` | LSTM | PyTorch checkpoint |
| `lstm_metadata.json` | LSTM | Feature list, config, training metrics |
| `lgb_model.txt` | GBDT | LightGBM saved model (text format) |
| `gbdt_metadata.json` | GBDT | Feature names, threshold, training metrics |
| `gnn_adversarial.pt` | GNN | PyTorch checkpoint |
| `gnn_adversarial_metadata.json` | GNN | Node feature config, training metrics |
| `adversarial_agent.pt` | PPO | Agent policy weights |
| `adversarial_metadata.json` | PPO | Action space config |
| `consolidation_config.json` | Consolidator | Phase weights for risk aggregation |

---

## 7. Running Inference on Real Data

`inference.py` loads persisted models and scores a transaction file.

### Run with synthetic data (default)

```bash
python inference.py
```

### Run with your own transaction CSV

```bash
python inference.py --input your_transactions.csv
```

Your CSV must contain at minimum: `source`, `target`, `amount`, `timestamp`.
Optional columns: `mcc`, `payment_type`, `channel`.

### Custom output location

```bash
python inference.py --input data.csv --output scored_results.csv
```

### Skip database logging

```bash
python inference.py --no-db
```

### Full options reference

```
python inference.py --help

  --input FILE     Input CSV (default: generate synthetic data)
  --output FILE    Output CSV (default: inference_results.csv)
  --no-db          Skip writing to metrics.db
```

### Pipeline simulation (for dashboard demo data)

```bash
python pipeline_simulation.py
```

This generates `simulation_pipeline_results.csv` used by the Pipeline and
Account Detail pages of the dashboard.

---

## 8. Starting the REST API

The inference API is a Flask server that exposes scoring endpoints.

### Start the server

```bash
python -m src.inference_api
```

Default address: **http://localhost:5000**

> The API loads all models at startup. Expect a 10–30 second initialisation delay.

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/score/transaction` | POST | GBDT per-transaction fraud score |
| `/score/consolidate` | POST | Multi-phase consolidated risk score |
| `/analyze/sequence` | POST | Sequence pattern detection |
| `/predict/links` | POST | LSTM emerging-link probability |
| `/graph/analyze` | POST | GNN node classification |

### Example: score a single transaction

```bash
curl -X POST http://localhost:5000/score/transaction \
  -H "Content-Type: application/json" \
  -d '{
    "source": "ACC_0001",
    "target": "ACC_0042",
    "amount": 9500,
    "channel": "online",
    "timestamp": "2026-04-03T14:30:00"
  }'
```

Example response:

```json
{
  "risk_score": 0.812,
  "risk_level": "HIGH",
  "signals": ["structuring_threshold", "high_velocity"],
  "latency_ms": 18.4
}
```

### Example: consolidated multi-phase score

```bash
curl -X POST http://localhost:5000/score/consolidate \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "ACC_0042",
    "transactions": [
      {"source": "ACC_0001", "target": "ACC_0042", "amount": 9500, "timestamp": "2026-04-03T14:30:00"},
      {"source": "ACC_0003", "target": "ACC_0042", "amount": 9200, "timestamp": "2026-04-03T15:10:00"}
    ]
  }'
```

---

## 9. Running the Live Demo Stack

To run the full live demo (API + transaction stream + dashboard together), use
the launcher script:

```bash
python launch_dashboard.py
```

It will:
1. Check dependencies and trained models.
2. Ask for a transaction rate (default: 2.0 tx/sec).
3. Start the Flask API on port 5000.
4. Start the transaction simulator.
5. Start the Streamlit dashboard on port 8501.

Press **Ctrl+C** to stop all three processes cleanly.

### Manual equivalent (three separate terminals)

**Terminal 1 — API server:**
```bash
python -m src.inference_api
```

**Terminal 2 — Transaction simulator:**
```bash
python transaction_simulator.py --rate 2.0
```

**Terminal 3 — Dashboard:**
```bash
streamlit run app.py
```

### Transaction simulator options

```bash
python transaction_simulator.py --rate 5.0      # 5 transactions/second
python transaction_simulator.py --rate 0.5      # 1 transaction every 2 seconds
```

---

## 10. Adversarial Retraining

The adversarial agent generates evasion patterns that are used to harden the
GNN and GBDT detectors.

### Single retraining round

```bash
python adversarial_retrain.py
```

### Multiple rounds

```bash
python adversarial_retrain.py --rounds 5
```

### Full options

| Flag | Default | Description |
|---|---|---|
| `--rounds N` | 1 | Number of adversarial training rounds |
| `--episodes N` | 50 | PPO episodes per round for pattern generation |
| `--gnn-epochs N` | 10 | GNN retraining epochs per round |
| `--mix-ratio R` | 0.5 | Fraction of adversarial data in training (0–1) |
| `--replay-capacity N` | 1000 | Experience replay buffer size |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--seed N` | 42 | Random seed |
| `--output DIR` | `models/` | Where to save the retrained GNN |

### Example: 3 rounds of adversarial hardening on GPU

```bash
python adversarial_retrain.py --rounds 3 --episodes 100 --device cuda
```

---

## 11. Configuration Reference

### `configs/rules.yml` — Rule toggles

Controls which detection rules are active and their contribution weight.

```yaml
fan_in_smurf:
  enabled: true
  weight: 1.0           # Increase to make fan-in patterns more influential

impossible_travel:
  enabled: false        # Set true to enable geographic impossibility checks
  weight: 1.0
```

**To add a new rule:** add an entry matching a pattern key in `src/pattern_library.py`,
set `enabled: true`, and assign a `weight` between 0.1 and 2.0.

---

### `models/consolidation_config.json` — Risk phase weights

Controls how much each detection phase contributes to the final risk score.
Weights are normalised internally; they do not need to sum to 1.

```json
{
  "weights": {
    "spatial":    0.20,
    "behavioral": 0.10,
    "temporal":   0.35,
    "lstm":       0.25,
    "cyber":      0.10
  },
  "multi_flag_boost": 0.15
}
```

**To emphasise LSTM predictions:** raise `"lstm"` to `0.40` and reduce `"temporal"` to `0.20`.  
**`multi_flag_boost`:** a 15% score uplift applied when ≥ 2 phases flag the same account.

---

### Temporal predictor thresholds

These are set in code (`src/temporal_predictor.py`) or passed via `main.py`.
Three preset profiles:

| Profile | `lookback_days` | `threshold_sigma` | `deviation_threshold` | `early_warning_threshold` | Use when |
|---|---|---|---|---|---|
| Conservative | 60 | 3.0 | 3.0 | 0.8 | Minimise false positives |
| Balanced (default) | 30 | 2.5 | 2.0 | 0.6 | General use |
| Aggressive | 14 | 2.0 | 1.5 | 0.4 | Minimise false negatives |

---

## 12. Output Files Reference

| File | Generated by | Contents |
|---|---|---|
| `transactions.csv` | `main.py`, `pipeline_simulation.py` | Raw synthetic transaction records |
| `simulation_pipeline_results.csv` | `main.py`, `pipeline_simulation.py` | Per-account results with risk level and signals |
| `consolidated_risk_scores.csv` | `inference.py`, `main.py` | Final ranked risk scores |
| `aml_network_graph.png` | `main.py` | Static graph PNG (spatial phase) |
| `metrics.db` | Inference API, `inference.py` | SQLite: inference logs, latency, engine stats |
| `alert_state.db` | Dashboard (alert management) | SQLite: per-account investigation states and notes |
| `outputs/rule_explanations.csv` | Pipeline | Neuro-symbolic rule explanations |
| `outputs/hetero_rule_explanations.csv` | Pipeline | Heterogeneous graph rule explanations |
| `models/*.pt` | `train.py` | PyTorch model checkpoints |
| `models/*.json` | `train.py` | Model metadata and feature configs |
| `models/lgb_model.txt` | `train.py` | LightGBM GBDT model |

---

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

You are running a script from the wrong directory. Always run from the project root:

```bash
cd path/to/aml-topology
python main.py      # correct
```

---

### `FileNotFoundError: models/lgb_model.txt not found`

The models have not been trained yet. Run:

```bash
python train.py
```

---

### Dashboard shows "No inference data yet"

The inference API is not running, or has not processed any transactions.

1. Start the API: `python -m src.inference_api`
2. Start the simulator: `python transaction_simulator.py`
3. Or run inference directly: `python inference.py`

---

### Dashboard auto-refresh freezes the page (Streamlit <1.37)

The `@st.fragment(run_every=...)` feature requires Streamlit ≥1.37. Check your version:

```bash
python -c "import streamlit; print(streamlit.__version__)"
```

Upgrade if needed:

```bash
pip install "streamlit>=1.37"
```

The app includes an automatic fallback to the blocking `time.sleep + st.rerun` pattern
for older versions, so it will still work — just with less smooth behaviour.

---

### `ConnectionRefusedError` when using Manual Score form

The inference API is not running. Open a separate terminal and run:

```bash
python -m src.inference_api
```

Wait for the message `API Server running on http://localhost:5000` before using the form.

---

### Graph Explorer shows "No transaction data found"

`transactions.csv` does not exist yet. Generate it:

```bash
python main.py
# or
python pipeline_simulation.py
```

---

### GNN or LSTM model not loaded (API warns "model unavailable")

Ensure the `.pt` files exist in `models/` and that `torch` is installed:

```bash
ls models/*.pt
python -c "import torch; print(torch.__version__)"
```

If missing, retrain: `python train.py --model gnn` or `python train.py --model lstm`.

---

### Port 5000 already in use

Another process is bound to port 5000. Either kill it:

```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS / Linux
lsof -ti:5000 | xargs kill -9
```

Or change the API port at the bottom of `src/inference_api.py`:

```python
app.run(host='0.0.0.0', port=5001, debug=False)   # change 5000 → 5001
```

And update `API_URL` in `transaction_simulator.py` to match.

---

### `psutil not found` in dashboard sidebar

This is optional. Install it to see live CPU/RAM metrics:

```bash
pip install psutil
```

Without it, the system status panel shows an install prompt and everything else works normally.

---

### Redis connection errors in logs

Redis is optional and only used for the distributed GNN embedding cache.
The system falls back to file-based caching automatically. No action required.
To suppress the warning, ensure Redis is running locally (`redis-server`) or
disable the Redis backend in `src/gnn_embedding_cache.py`.

---

## 14. Component Reference Card

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMMAND QUICK REFERENCE                          │
├──────────────────────────────────────┬──────────────────────────────┤
│ Task                                 │ Command                      │
├──────────────────────────────────────┼──────────────────────────────┤
│ Install dependencies                 │ pip install -r requirements  │
│                                      │   .txt                       │
├──────────────────────────────────────┼──────────────────────────────┤
│ Train all models                     │ python train.py              │
│ Train LSTM only                      │ python train.py --model lstm │
│ Train GBDT only                      │ python train.py --model gbdt │
│ Train GNN only                       │ python train.py --model gnn  │
│ Train adversarial agent              │ python train.py              │
│                                      │   --model adversarial        │
├──────────────────────────────────────┼──────────────────────────────┤
│ Full pipeline (synthetic data)       │ python main.py               │
│ Inference on custom CSV              │ python inference.py          │
│                                      │   --input data.csv           │
│ Pipeline simulation (for dashboard)  │ python pipeline_simulation   │
│                                      │   .py                        │
├──────────────────────────────────────┼──────────────────────────────┤
│ Start REST API                       │ python -m src.inference_api  │
│ Start transaction simulator          │ python transaction_simulator │
│                                      │   .py --rate 2.0             │
├──────────────────────────────────────┼──────────────────────────────┤
│ Launch unified dashboard             │ streamlit run app.py         │
│ Launch full demo stack               │ python launch_dashboard.py   │
├──────────────────────────────────────┼──────────────────────────────┤
│ Adversarial retraining               │ python adversarial_retrain   │
│                                      │   .py --rounds 3             │
├──────────────────────────────────────┼──────────────────────────────┤
│ Health check                         │ curl http://localhost:5000   │
│                                      │   /health                    │
└──────────────────────────────────────┴──────────────────────────────┘

Ports:   Dashboard → 8501    Inference API → 5000
Data:    metrics.db          alert_state.db
Models:  models/*.pt         models/lgb_model.txt
Results: simulation_pipeline_results.csv
         consolidated_risk_scores.csv
```

---

*AML Topology Detection System — Run Manual v3.0.0*
