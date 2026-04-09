# AML Topology Detection System — Run Manual (Security Edition)

> **This document supersedes `RUN_MANUAL.md` for all API and demo stack usage.**  
> It reflects the Phase 1 security controls added in April 2026:  
> API key authentication, Pydantic input validation, rate limiting, audit logging, and secrets management.  
> Sections that are unchanged from the original manual are noted with *(unchanged)*.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Setting Up Secrets (.env)](#3-setting-up-secrets-env) ← **NEW — must do before starting the API**
4. [Quick Start (5 minutes)](#4-quick-start-5-minutes)
5. [Running the Unified Dashboard](#5-running-the-unified-dashboard)
6. [Running the Full Detection Pipeline](#6-running-the-full-detection-pipeline)
7. [Training the Models](#7-training-the-models)
8. [Running Inference on Real Data](#8-running-inference-on-real-data)
9. [Starting the REST API](#9-starting-the-rest-api) ← **Significantly changed**
10. [Running the Live Demo Stack](#10-running-the-live-demo-stack) ← **Changed**
11. [Adversarial Retraining](#11-adversarial-retraining)
12. [Configuration Reference](#12-configuration-reference) ← **New .env section added**
13. [Output Files Reference](#13-output-files-reference) ← **New audit.db entry**
14. [Troubleshooting](#14-troubleshooting) ← **New security error entries**
15. [Component Reference Card](#15-component-reference-card) ← **Updated curl commands**

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
| nginx | TLS termination in production (see `configs/nginx_example.conf`) |

---

## 2. Installation

### Step 1 — Clone / navigate to the project

```bash
cd aml-topology
```

### Step 2 — Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This now installs the original packages plus the Phase 1 security additions:
`pydantic>=2.0`, `Flask-Limiter>=3.5`, `Flask-Talisman>=1.1`, `python-dotenv>=1.0`.

### Step 4 — Verify the installation

```bash
python3 -c "
import lightgbm, plotly, networkx, flask
import pydantic, flask_limiter, dotenv
print('All OK')
"
```

Expected output: `All OK`

> **Note:** `torch` is optional. If not installed, the LSTM and Sequence Detector models
> will be skipped but GBDT scoring and GNN still work.

---

## 3. Setting Up Secrets (.env)

> **This step is required before starting the REST API.** The API will refuse all requests
> (HTTP 401) if `AML_API_KEYS` is not set.

### Step 1 — Copy the example file

```bash
cp .env.example .env
```

### Step 2 — Generate a strong API key

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Example output: `a3f8c1d2e4b7...` (64-character hex string)

### Step 3 — Edit `.env`

Open `.env` and replace the placeholders:

```env
# One or more API keys, comma-separated.
# Clients must send one of these in the X-API-Key header.
AML_API_KEYS=your-generated-key-here

# Set to "production" to enforce HTTPS via Flask-Talisman.
AML_ENV=development

# SQLite paths (defaults are fine for local use)
AML_DB_PATH=metrics.db
AML_AUDIT_DB_PATH=audit.db
```

> **Multiple keys:** To support multiple clients or key rotation, list several keys:
> `AML_API_KEYS=key-for-client-a,key-for-analyst-b,key-for-dashboard`

> **Security:** `.env` is listed in `.gitignore` and must never be committed to version
> control. It contains the same secrets as a password — treat it accordingly.

---

## 4. Quick Start (5 minutes)

Run these commands in order to go from nothing to a live dashboard.

```bash
# 1. Set up secrets (required — see Section 3)
cp .env.example .env
# Edit .env and set AML_API_KEYS to a real key

# 2. Train all models (generates models/ artefacts)
python3 train.py

# 3. Run the full detection pipeline (generates CSV results)
python3 main.py

# 4. Launch the unified dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **If models are already trained** (the `models/` directory contains `.pt` and `.txt`
> files), skip step 2 and go straight to step 3 or 4.

> **To also use the API and manual scoring form**, start the API in a separate terminal:
> ```bash
> python3 -m src.inference_api
> ```
> See [Section 9](#9-starting-the-rest-api) for full API instructions.

---

## 5. Running the Unified Dashboard

*(Unchanged from RUN_MANUAL.md — the dashboard itself does not require an API key in
its Streamlit interface. The Manual Transaction Scoring form on the Monitor page calls
the inference API and will need the API running with a valid key — see note below.)*

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

### Manual transaction scoring from the dashboard

The Monitor page **Manual Transaction Score** form calls `http://localhost:5000`.
For this to work:

1. The API must be running (`python3 -m src.inference_api`).
2. The API must have `AML_API_KEYS` set in `.env`.
3. The dashboard's API call must include the key — configure `API_KEY` in the dashboard
   environment or check `pages/monitor.py` for where the API URL is set.

---

## 6. Running the Full Detection Pipeline

*(Unchanged)*

```bash
python3 main.py
```

---

## 7. Training the Models

*(Unchanged)*

```bash
python3 train.py                          # all models
python3 train.py --model gbdt             # GBDT only
python3 train.py --model lstm --epochs 50 # longer LSTM training
python3 train.py --model gnn
python3 train.py --model sequence
python3 train.py --model adversarial
```

---

## 8. Running Inference on Real Data

*(Unchanged)*

```bash
python3 inference.py
python3 inference.py --input your_transactions.csv
python3 inference.py --input data.csv --output scored_results.csv --no-db
```

---

## 9. Starting the REST API

> **This section is significantly changed from RUN_MANUAL.md.**
> All endpoints now require an `X-API-Key` header. All curl examples below
> include the header. Replace `YOUR_KEY` with the key you set in `.env`.

### Start the server

```bash
# Ensure .env exists with AML_API_KEYS set (Section 3)
python3 -m src.inference_api
```

Default address: **http://localhost:5000**

The server loads all models at startup (10–30 seconds), then prints:

```
API Server running on http://localhost:5000
```

If `AML_API_KEYS` is not set, the server still starts but every request returns 401.

---

### Authentication

Every request must include the `X-API-Key` header:

```
X-API-Key: YOUR_KEY
```

| Scenario | HTTP Status |
|----------|------------|
| Header missing | 401 Unauthorized |
| Key invalid / wrong | 401 Unauthorized |
| Key valid | Request proceeds |

Every auth failure is written to the audit log (`metrics.db → audit_log` table).

---

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — model status |
| `/score/transaction` | POST | GBDT per-transaction fraud score |
| `/score/sequence` | POST | Sequence pattern detection |
| `/score/consolidate` | POST | Multi-phase consolidated risk score |
| `/batch/score` | POST | Score up to 1,000 transactions in one call |
| `/models/info` | GET | Loaded model metadata |

> **Change from RUN_MANUAL.md:** The old manual listed `/analyze/sequence`,
> `/predict/links`, and `/graph/analyze`. These paths were superseded by
> `/score/sequence` and `/score/consolidate` in the current implementation.

---

### Health check

```bash
curl -H "X-API-Key: YOUR_KEY" http://localhost:5000/health
```

Response:

```json
{
  "status": "healthy",
  "timestamp": "2026-04-04T20:00:00",
  "models_loaded": {
    "gbdt": true,
    "sequence": true,
    "lstm": false,
    "gnn": false,
    "consolidator": true
  }
}
```

---

### Score a single transaction

```bash
curl -X POST http://localhost:5000/score/transaction \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 9500,
    "mcc": "4829",
    "payment_type": "wire",
    "device_change": 1,
    "ip_risk": 0.7,
    "count_1h": 8,
    "sum_24h": 45000,
    "uniq_payees_24h": 6,
    "country": "RU"
  }'
```

**Validated fields:**

| Field | Type | Constraint | Required |
|-------|------|-----------|----------|
| `amount` | float | `> 0` | Yes |
| `mcc` | string | max 10 chars | No |
| `payment_type` | string | max 50 chars | No |
| `device_change` | int | 0 or 1 | No |
| `ip_risk` | float | 0.0 – 1.0 | No |
| `count_1h` | int | ≥ 0 | No |
| `sum_24h` | float | ≥ 0 | No |
| `uniq_payees_24h` | int | ≥ 0 | No |
| `country` | string | max 10 chars | No |

Example response:

```json
{
  "gbdt_score": 0.812,
  "gbdt_risk_level": "HIGH",
  "risk_score": 0.812,
  "risk_level": "HIGH",
  "model": "gbdt",
  "timestamp": "2026-04-04T20:00:00"
}
```

**If the payload fails validation (e.g. negative amount):**

```bash
curl -X POST http://localhost:5000/score/transaction \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"amount": -500}'
```

Returns **HTTP 422**:

```json
{
  "error": "Validation error",
  "detail": [
    {
      "loc": ["amount"],
      "msg": "Input should be greater than 0",
      "type": "greater_than"
    }
  ]
}
```

---

### Score an event sequence

```bash
curl -X POST http://localhost:5000/score/sequence \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"events": ["login_success", "view_account", "transfer", "logout"]}'
```

**Validated fields:**

| Field | Type | Constraint | Required |
|-------|------|-----------|----------|
| `events` | list of strings | 1 – 500 items | Yes |

---

### Consolidated multi-phase risk score

```bash
curl -X POST http://localhost:5000/score/consolidate \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "ACC_0042",
    "transaction": {
      "amount": 9500,
      "mcc": "4829",
      "payment_type": "wire",
      "country": "RU"
    },
    "events": ["login_success", "transfer", "logout"]
  }'
```

**Validated fields:**

| Field | Type | Constraint | Required |
|-------|------|-----------|----------|
| `account_id` | string | max 100 chars | No |
| `transaction` | object | see transaction fields above | Yes |
| `events` | list of strings | max 500 items | No |

Example response:

```json
{
  "account_id": "ACC_0042",
  "consolidated_risk_score": 0.74,
  "risk_level": "HIGH",
  "recommendation": "Block or require additional verification",
  "component_scores": {
    "gbdt": {"score": 0.81, "weight": 0.3, "status": "success"}
  },
  "gnn_enhanced": false,
  "timestamp": "2026-04-04T20:00:00"
}
```

---

### Batch scoring

Score up to 1,000 transactions in a single request:

```bash
curl -X POST http://localhost:5000/batch/score \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "account_id": "ACC_0001",
        "transaction": {"amount": 500, "mcc": "5411"},
        "events": ["login_success", "transfer"]
      },
      {
        "account_id": "ACC_0002",
        "transaction": {"amount": 95000, "mcc": "4829", "country": "NG"},
        "events": []
      }
    ]
  }'
```

**Validated fields:**

| Field | Type | Constraint |
|-------|------|-----------|
| `transactions` | list | 1 – 1,000 items |
| Each item | object | same schema as `/score/consolidate` |

---

### Rate limits

| Endpoint | Limit |
|----------|-------|
| All endpoints (global) | 200 requests / minute / IP |
| `/score/consolidate` | 30 requests / minute / IP |
| `/batch/score` | 30 requests / minute / IP |

When a limit is exceeded, the server returns **HTTP 429**:

```json
{
  "error": "Rate limit exceeded",
  "detail": "200 per 1 minute"
}
```

The response also includes a `Retry-After` header indicating when to retry.
Every 429 event is written to the audit log.

---

### HTTP status code reference

| Code | Meaning | Common cause |
|------|---------|-------------|
| 200 | OK | Valid key + valid payload |
| 401 | Unauthorized | Missing or wrong `X-API-Key` header |
| 422 | Unprocessable Entity | Payload fails Pydantic validation (bad field value or type) |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Model not loaded, or unhandled exception |

---

## 10. Running the Live Demo Stack

> **Changed from RUN_MANUAL.md.** The API now requires authentication. You must set
> `AML_API_KEYS` in `.env` before running `launch_dashboard.py` or the transaction
> simulator.

### Ensure `.env` is set up first

```bash
cat .env   # confirm AML_API_KEYS is set
```

### Launch the full stack

```bash
python3 launch_dashboard.py
```

This starts:
1. The Flask inference API on port 5000 (reads `AML_API_KEYS` from `.env`)
2. The transaction simulator (sends requests to the API with the key)
3. The Streamlit dashboard on port 8501

Press **Ctrl+C** to stop all three processes cleanly.

### Manual equivalent (three separate terminals)

**Terminal 1 — API server:**

```bash
python3 -m src.inference_api
# Must have .env present with AML_API_KEYS set
```

**Terminal 2 — Transaction simulator:**

```bash
python3 transaction_simulator.py --rate 2.0
# If the simulator calls the API, ensure it includes the X-API-Key header.
# Set AML_API_KEYS in the environment or check transaction_simulator.py for
# the API_KEY configuration variable.
```

**Terminal 3 — Dashboard:**

```bash
streamlit run app.py
```

### Transaction simulator options

```bash
python3 transaction_simulator.py --rate 5.0   # 5 tx/second
python3 transaction_simulator.py --rate 0.5   # 1 tx every 2 seconds
```

---

## 11. Adversarial Retraining

*(Unchanged)*

```bash
python3 adversarial_retrain.py
python3 adversarial_retrain.py --rounds 5 --episodes 100 --device cuda
```

---

## 12. Configuration Reference

### `.env` — Secrets and environment (NEW)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AML_API_KEYS` | Yes | — | Comma-separated API keys. All API requests must include one in `X-API-Key`. |
| `AML_ENV` | No | `development` | Set to `production` to enable HTTPS enforcement via Flask-Talisman. |
| `AML_DB_PATH` | No | `metrics.db` | Path to the operational metrics SQLite database. |
| `AML_AUDIT_DB_PATH` | No | `audit.db` | Path to the audit log database (referenced in `.env`; currently metrics and audit share `metrics.db`). |

---

### `configs/rules.yml` — Rule toggles *(unchanged)*

```yaml
fan_in_smurf:
  enabled: true
  weight: 1.0

impossible_travel:
  enabled: false
  weight: 1.0
```

---

### `models/consolidation_config.json` — Risk phase weights *(unchanged)*

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

---

### `configs/nginx_example.conf` — TLS reverse proxy (NEW)

For production deployments, terminate TLS at nginx and proxy to the Flask API on port 5000.

```bash
# Copy and adapt the example config
cp configs/nginx_example.conf /etc/nginx/sites-available/aml-api
# Edit: set server_name and ssl_certificate paths
nginx -t && systemctl reload nginx
```

Set `AML_ENV=production` in `.env` to also enable Flask-Talisman's HTTPS enforcement at
the application layer.

---

### Temporal predictor profiles *(unchanged)*

| Profile | `lookback_days` | `threshold_sigma` | Use when |
|---|---|---|---|
| Conservative | 60 | 3.0 | Minimise false positives |
| Balanced (default) | 30 | 2.5 | General use |
| Aggressive | 14 | 2.0 | Minimise false negatives |

---

## 13. Output Files Reference

| File | Generated by | Contents |
|---|---|---|
| `transactions.csv` | `main.py`, `pipeline_simulation.py` | Raw synthetic transaction records |
| `simulation_pipeline_results.csv` | `main.py`, `pipeline_simulation.py` | Per-account results with risk level and signals |
| `consolidated_risk_scores.csv` | `inference.py`, `main.py` | Final ranked risk scores |
| `aml_network_graph.png` | `main.py` | Static graph PNG (spatial phase) |
| `metrics.db` | Inference API, `inference.py` | SQLite: inference logs, latency, engine stats, **audit_log table** |
| `audit.db` | Inference API (if `AML_AUDIT_DB_PATH` set separately) | Dedicated audit log for compliance use |
| `alert_state.db` | Dashboard (alert management) | SQLite: per-account investigation states and notes |
| `outputs/rule_explanations.csv` | Pipeline | Neuro-symbolic rule explanations |
| `models/*.pt` | `train.py` | PyTorch model checkpoints |
| `models/*.json` | `train.py` | Model metadata and feature configs |
| `models/lgb_model.txt` | `train.py` | LightGBM GBDT model |

### Querying the audit log

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('metrics.db')
conn.row_factory = sqlite3.Row
rows = conn.execute(
    'SELECT event_time, event_type, client_ip, endpoint, outcome '
    'FROM audit_log ORDER BY id DESC LIMIT 20'
).fetchall()
for r in rows:
    print(f'[{r[\"event_type\"]:15}] {r[\"endpoint\"]:30} {r[\"outcome\"]:10} {r[\"event_time\"]}')
"
```

---

## 14. Troubleshooting

### `HTTP 401 Unauthorized` from the API

The `X-API-Key` header is missing or the key value is wrong.

**Check 1 — Is `.env` present and populated?**

```bash
cat .env | grep AML_API_KEYS
```

If empty or missing, follow [Section 3](#3-setting-up-secrets-env).

**Check 2 — Is the key in your request matching exactly?**

Keys are case-sensitive and whitespace-sensitive. Compare character-by-character if needed.

**Check 3 — Did the server load the key at startup?**

Look for this in the server output on startup:

```
WARNING: AML_API_KEYS is not set.
```

If you see this, the `.env` file was not found or `AML_API_KEYS` is blank. Fix `.env` and restart the API.

---

### `HTTP 422 Unprocessable Entity`

The request body failed schema validation. The response body tells you exactly which field failed:

```json
{
  "error": "Validation error",
  "detail": [{"loc": ["amount"], "msg": "Input should be greater than 0"}]
}
```

Common causes:

| Error | Fix |
|-------|-----|
| `amount` ≤ 0 | Send a positive number |
| `events` is empty `[]` | Provide at least one event string, or omit the field |
| `transactions` list is empty | Provide at least one transaction item in batch requests |
| `ip_risk` outside 0–1 | Clamp to the 0.0–1.0 range |
| `device_change` not 0 or 1 | Send only `0` or `1` |

---

### `HTTP 429 Too Many Requests`

You have exceeded the rate limit. The response includes a `Retry-After` header.

```bash
# Check the header
curl -v -X POST http://localhost:5000/score/consolidate \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"transaction":{"amount":100}}' 2>&1 | grep -i "retry-after"
```

Wait the indicated number of seconds, then retry. Limits reset on a rolling 1-minute window.

---

### `ModuleNotFoundError: No module named 'src'` *(unchanged)*

Run from the project root:

```bash
cd path/to/aml-topology
python3 main.py
```

---

### `FileNotFoundError: models/lgb_model.txt not found` *(unchanged)*

```bash
python3 train.py
```

---

### Dashboard shows "No inference data yet" *(unchanged)*

```bash
python3 -m src.inference_api    # terminal 1
python3 transaction_simulator.py  # terminal 2
```

---

### `ConnectionRefusedError` when using Manual Score form *(unchanged)*

The inference API is not running:

```bash
python3 -m src.inference_api
```

---

### Port 5000 already in use *(unchanged)*

```bash
# macOS / Linux
lsof -ti:5000 | xargs kill -9

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

---

### Redis connection errors in logs *(unchanged)*

Redis is optional; the system falls back to file-based caching automatically.

---

### The audit log is not recording entries

The audit log is stored in `metrics.db` under the `audit_log` table. If the table is
missing, the database was created before Phase 1. Delete `metrics.db` and restart the
API to rebuild the schema with the new table, or run:

```bash
python3 -c "
from src.metrics_logger import MetricsLogger
MetricsLogger('metrics.db')  # re-runs _init_db, applies migrations
print('Schema updated')
"
```

---

## 15. Component Reference Card

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     COMMAND QUICK REFERENCE (Security Edition)          │
├───────────────────────────────────────┬─────────────────────────────────┤
│ Task                                  │ Command                         │
├───────────────────────────────────────┼─────────────────────────────────┤
│ Set up secrets (REQUIRED FIRST)       │ cp .env.example .env            │
│                                       │ # Edit .env → set AML_API_KEYS  │
├───────────────────────────────────────┼─────────────────────────────────┤
│ Install dependencies                  │ pip install -r requirements.txt │
├───────────────────────────────────────┼─────────────────────────────────┤
│ Train all models                      │ python3 train.py                │
│ Train GBDT only                       │ python3 train.py --model gbdt   │
│ Train LSTM only                       │ python3 train.py --model lstm   │
│ Train GNN only                        │ python3 train.py --model gnn    │
├───────────────────────────────────────┼─────────────────────────────────┤
│ Full pipeline (synthetic data)        │ python3 main.py                 │
│ Inference on custom CSV               │ python3 inference.py            │
│                                       │   --input data.csv              │
├───────────────────────────────────────┼─────────────────────────────────┤
│ Start REST API (requires .env)        │ python3 -m src.inference_api    │
│ Launch full demo stack                │ python3 launch_dashboard.py     │
│ Start transaction simulator           │ python3 transaction_simulator.py│
│                                       │   --rate 2.0                    │
├───────────────────────────────────────┼─────────────────────────────────┤
│ Launch unified dashboard              │ streamlit run app.py            │
├───────────────────────────────────────┼─────────────────────────────────┤
│ Adversarial retraining                │ python3 adversarial_retrain.py  │
│                                       │   --rounds 3                    │
├───────────────────────────────────────┼─────────────────────────────────┤
│ Health check (with auth)              │ curl -H "X-API-Key: YOUR_KEY"   │
│                                       │   http://localhost:5000/health  │
│ Score transaction (with auth)         │ curl -X POST                    │
│                                       │   -H "X-API-Key: YOUR_KEY"      │
│                                       │   -H "Content-Type: app/json"   │
│                                       │   -d '{"amount":5000}'          │
│                                       │   .../score/transaction         │
│ Query audit log                       │ sqlite3 metrics.db              │
│                                       │   "SELECT * FROM audit_log      │
│                                       │    ORDER BY id DESC LIMIT 20"   │
└───────────────────────────────────────┴─────────────────────────────────┘

Ports:    Dashboard → 8501    Inference API → 5000
Secrets:  .env (AML_API_KEYS) — never commit this file
Data:     metrics.db (inference + audit_log)    alert_state.db
Models:   models/*.pt    models/lgb_model.txt
Results:  simulation_pipeline_results.csv
          consolidated_risk_scores.csv
```

---

## What Changed from RUN_MANUAL.md — Summary

| Section | Change |
|---------|--------|
| 2. Installation | New packages in `requirements.txt`: pydantic, Flask-Limiter, Flask-Talisman, python-dotenv |
| **3. Secrets (.env)** | **Entirely new section — must be done before starting API** |
| 5. Dashboard | Added note about Manual Score form requiring API key |
| **9. REST API** | All `curl` examples updated to include `-H "X-API-Key: YOUR_KEY"`. HTTP 401/422/429 responses documented. `/analyze/sequence`, `/predict/links`, `/graph/analyze` replaced by `/score/sequence`, `/score/consolidate`. Endpoint field validation table added. |
| 10. Live Demo Stack | Added `.env` prerequisite note |
| 12. Configuration | New `.env` variables table; new nginx section |
| 13. Output Files | Added `audit.db`; audit log query snippet |
| 14. Troubleshooting | New entries: 401, 422, 429, audit log missing |
| 15. Reference Card | curl commands updated with `-H "X-API-Key"`; `.env` setup added as first step |

---

*AML Topology Detection System — Run Manual (Security Edition) v4.0.0*  
*Supersedes RUN_MANUAL.md v3.0.0*
