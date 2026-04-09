# Captivus Implementation Plan

**Date:** 2026-04-04  
**Based on:** CAPTIVUS_GAP_ANALYSIS.md  
**Scope:** All four Captivus deliverables — Security, Accuracy Report, ML Documentation, Scalability

---

## Delivery Phases

| Phase | Focus | Deliverable | Effort |
|-------|-------|-------------|--------|
| **Phase 1** | Security foundations | Hardened API + audit log | 7–10 days |
| **Phase 2** | Accuracy report | Evaluation harness + test report | 5–8 days |
| **Phase 3** | ML documentation | Model governance document + CV | 3–5 days |
| **Phase 4** | Scalability | Load test + TPS report | 3–5 days |

---

## Phase 1: Security Architecture

**Goal:** Close all critical security gaps so the system is safe to demo or pilot with Captivus.  
**Deliverable:** Hardened API + `SECURITY_ARCHITECTURE.md`

---

### Task 1.1 — API Key Authentication

**File:** `src/inference_api.py`  
**Approach:** Add a `require_api_key` decorator that checks the `X-API-Key` request header against keys stored in an environment variable. Apply it to all six endpoints.

**Steps:**
1. Add `python-dotenv` to `requirements.txt`
2. Create `.env.example` with `AML_API_KEYS=key1,key2` (comma-separated, never committed)
3. In `inference_api.py`, at app startup, read `AML_API_KEYS` from `os.environ`
4. Write `require_api_key` decorator:
   ```python
   from functools import wraps
   def require_api_key(f):
       @wraps(f)
       def decorated(*args, **kwargs):
           key = request.headers.get("X-API-Key", "")
           if key not in VALID_KEYS:
               return jsonify({"error": "Unauthorized"}), 401
           return f(*args, **kwargs)
       return decorated
   ```
5. Apply decorator to: `/health`, `/score/transaction`, `/analyze/sequence`, `/predict/links`, `/graph/analyze`, `/consolidate/risks`
6. Add `X-API-Key` header to `src/inference_client.py` so existing client keeps working

**Verification:** `curl /health` without key → 401. With valid key → 200.

---

### Task 1.2 — Input Validation with Pydantic

**File:** `src/inference_api.py`, new file `src/api_schemas.py`  
**Approach:** Define Pydantic v2 `BaseModel` schemas for every endpoint's request body. Validate before any processing.

**Steps:**
1. Add `pydantic>=2.0` to `requirements.txt`
2. Create `src/api_schemas.py` with these models:

   ```python
   from pydantic import BaseModel, Field
   from typing import List, Optional

   class TransactionRequest(BaseModel):
       account_id: str
       amount: float = Field(..., gt=0)
       mcc: Optional[str] = None
       counterparty_id: Optional[str] = None
       timestamp: Optional[str] = None

   class SequenceRequest(BaseModel):
       events: List[str] = Field(..., min_length=1, max_length=500)

   class ConsolidateRequest(BaseModel):
       account_id: str
       transactions: List[TransactionRequest] = Field(..., max_length=1000)
       events: Optional[List[str]] = []
   ```

3. In each endpoint, parse with `Model.model_validate(request.get_json())` inside a try/except `ValidationError` → return 400
4. Pass the validated model's `.model_dump()` to existing inference logic (no downstream changes needed)

**Verification:** POST `/score/transaction` with `amount: -5` → 422 with field error. Valid payload → 200.

---

### Task 1.3 — Rate Limiting

**File:** `src/inference_api.py`  
**Approach:** Add Flask-Limiter. Apply per-IP limits globally, stricter limits on expensive endpoints.

**Steps:**
1. Add `Flask-Limiter` to `requirements.txt`
2. Initialise limiter at app creation:
   ```python
   from flask_limiter import Limiter
   from flask_limiter.util import get_remote_address
   limiter = Limiter(get_remote_address, app=app, default_limits=["200/minute"])
   ```
3. Apply tighter limit on heavy endpoints:
   ```python
   @limiter.limit("30/minute")
   def consolidate_endpoint(): ...
   ```
4. Return standard 429 response with `Retry-After` header on limit breach

**Verification:** Fire 201 rapid requests to `/health` → 201st returns 429.

---

### Task 1.4 — Audit Logging

**File:** `src/metrics_logger.py`  
**Approach:** Extend the existing SQLite `inference_logs` table with security-relevant columns. Add a new append-only `audit_log` table that is never updated or deleted.

**Steps:**
1. Add columns to `inference_logs`: `requester_key_hash TEXT`, `input_payload_hash TEXT`, `client_ip TEXT`
   - Store SHA-256 of the API key (never the key itself)
   - Store SHA-256 of the raw request JSON body
2. Create new table `audit_log`:
   ```sql
   CREATE TABLE IF NOT EXISTS audit_log (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       event_time TEXT NOT NULL,
       event_type TEXT NOT NULL,      -- 'inference', 'auth_failure', 'rate_limit'
       client_ip TEXT,
       key_hash TEXT,
       endpoint TEXT,
       input_hash TEXT,
       outcome TEXT,
       detail TEXT
   );
   ```
3. Add `log_audit_event(event_type, client_ip, key_hash, endpoint, input_hash, outcome, detail)` method to `MetricsLogger`
4. Call `log_audit_event` from:
   - `require_api_key` decorator on every request (success and failure)
   - Rate limit handler (429 events)
   - Any unhandled exception handler (500 events)
5. Add SQLite WAL mode (`PRAGMA journal_mode=WAL`) to prevent corruption under concurrent writes
6. Never expose a DELETE or UPDATE path on `audit_log` — log rows are write-only

**Verification:** Make authenticated request → row appears in `audit_log` with non-null `input_hash`. Make unauthenticated request → row appears with `event_type='auth_failure'`.

---

### Task 1.5 — HTTPS / TLS

**Approach:** The Flask dev server does not support production TLS. Document the recommended production deployment behind an nginx reverse proxy. For local testing, add optional self-signed cert support.

**Steps:**
1. Create `configs/nginx_example.conf` — minimal nginx config with TLS termination:
   ```nginx
   server {
       listen 443 ssl;
       ssl_certificate /etc/ssl/certs/aml.crt;
       ssl_certificate_key /etc/ssl/private/aml.key;
       location / { proxy_pass http://127.0.0.1:5000; }
   }
   ```
2. Add `Flask-Talisman` to `requirements.txt` — forces HTTPS headers and sets HSTS
3. Enable Talisman conditionally when `AML_ENV=production` in env:
   ```python
   if os.getenv("AML_ENV") == "production":
       from flask_talisman import Talisman
       Talisman(app, force_https=True)
   ```
4. Document TLS setup in `SECURITY_ARCHITECTURE.md`

---

### Task 1.6 — Secrets Management (.env)

**Steps:**
1. Create `.env.example`:
   ```
   AML_API_KEYS=change-me-key1,change-me-key2
   AML_ENV=development
   AML_DB_PATH=metrics.db
   AML_AUDIT_DB_PATH=audit.db
   ```
2. Add `.env` to `.gitignore` (verify it is not already tracked)
3. Load at app entry point: `from dotenv import load_dotenv; load_dotenv()`
4. Replace all hard-coded paths/secrets in `inference_api.py` and `metrics_logger.py` with `os.getenv()` calls with safe defaults

---

### Task 1.7 — Write SECURITY_ARCHITECTURE.md

**Sections:**
1. Threat Model — actors (external attacker, rogue analyst, compromised host), attack surfaces (API, disk, network)
2. Controls Implemented — auth (API key), input validation (Pydantic), rate limiting, audit log, TLS (production)
3. Controls Not Yet Implemented — RBAC, OAuth2/JWT, field-level encryption, HSM key management
4. Audit Trail — schema description, what is logged, retention policy recommendation (7 years for AML records per FATF)
5. Compliance Posture — GDPR data minimisation (node IDs anonymised), AML record-keeping, right-to-deletion procedure
6. Deployment Checklist — steps required before production go-live

---

## Phase 2: Accuracy Report (90%+ Claim Substantiation)

**Goal:** Produce a signed-off, reproducible test report with raw metrics per model.  
**Deliverable:** `evaluate_models.py` + `MODEL_EVALUATION_REPORT.md` + `outputs/` plots

---

### Task 2.1 — Build Labelled Evaluation Dataset

**File:** new `scripts/generate_eval_dataset.py`  
**Approach:** Use the existing `TransactionSimulator` (`src/simulator.py`) but retain ground-truth fraud labels in the output. Save as a fixed, versioned CSV so results are reproducible.

**Steps:**
1. Inspect `src/simulator.py` to confirm it marks synthetic fraud nodes/transactions
2. Write `scripts/generate_eval_dataset.py`:
   - Generate 50,000 transactions with `seed=2025` (fixed for reproducibility)
   - Keep 80% for model training context; reserve 20% (10,000 txs) as the **held-out test set**
   - Save test set to `data/eval_dataset.csv` with columns: all features + `label` (0/1) + `account_id`
   - Save a `data/eval_dataset_metadata.json` recording: seed, n_samples, fraud_rate, generation date
3. Commit `data/eval_dataset_metadata.json` but add `data/*.csv` to `.gitignore` (large file)
4. Add a checksum (`sha256`) of the CSV to the metadata JSON so future runs can verify the dataset hasn't changed

---

### Task 2.2 — Build Evaluation Harness

**File:** new `evaluate_models.py` (project root)  
**Approach:** Load each model in isolation, score the held-out test set, compute standard metrics using the manual metric functions already in `gbdt_detector.py`.

**Structure:**
```
evaluate_models.py
  ├── load_eval_dataset()          → pd.DataFrame with label column
  ├── evaluate_gbdt(df)            → MetricsResult
  ├── evaluate_gnn(df)             → MetricsResult
  ├── evaluate_lstm(df)            → MetricsResult
  ├── evaluate_sequence(df)        → MetricsResult
  ├── evaluate_combined_pipeline(df) → MetricsResult
  ├── generate_plots(results)      → saves to outputs/
  └── write_report(results)        → writes MODEL_EVALUATION_REPORT.md
```

**MetricsResult dataclass:**
```python
@dataclass
class MetricsResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    avg_precision: float       # area under PR curve
    confusion_matrix: list     # [[TN, FP], [FN, TP]]
    threshold: float
    n_test: int
    n_fraud: int
    timestamp: str
```

**Steps:**
1. Reuse `accuracy_score_manual`, `precision_score_manual`, `recall_score_manual`, `roc_auc_manual` from `gbdt_detector.py` — import them rather than re-implementing
2. Add `f1_score_manual` and `avg_precision_manual` (PR-AUC) to `gbdt_detector.py` (2 small functions)
3. For GBDT: featurize test set using existing `featurize()` function, call `score_transaction()` per row
4. For GNN: build graph from test transactions, call GNN node embeddings, classify with saved model
5. For LSTM: build pair sequences from test set, run LSTM link predictor
6. For Sequence Detector: encode event sequences from test transactions, run detector
7. Combined pipeline: call `InferenceEngine.consolidate_risks()` for each test account, compare consolidated risk label vs ground truth
8. Save raw numbers to `outputs/evaluation_results.json`

---

### Task 2.3 — Generate Plots

**File:** inside `evaluate_models.py`, using `matplotlib`

**Plots to generate (save to `outputs/`):**

| Plot | Filename |
|------|----------|
| ROC curve (all models on one chart) | `outputs/roc_curves.png` |
| Precision-Recall curve (all models) | `outputs/pr_curves.png` |
| Confusion matrix heatmap per model | `outputs/confusion_matrix_<model>.png` |
| F1 / Precision / Recall bar chart | `outputs/metrics_comparison.png` |

**Steps:**
1. After computing all `MetricsResult` objects, call `generate_plots(results)`
2. Use `matplotlib` (already likely installed); no new dependencies needed
3. Each plot saved at 150 dpi, labelled with model name, test date, and dataset size

---

### Task 2.4 — Write MODEL_EVALUATION_REPORT.md

**File:** `MODEL_EVALUATION_REPORT.md` (auto-generated by `evaluate_models.py`)

**Sections:**
1. Executive Summary — headline metrics table (one row per model)
2. Test Dataset — source, size, fraud rate, checksum, generation date
3. Per-Model Results — for each model: architecture summary, metrics table, confusion matrix, ROC/PR curve reference
4. Combined Pipeline Results — consolidated score performance
5. Accuracy Claim Assessment — explicitly address the "90%+" claim with the correct metric (e.g. "GBDT achieves F1=0.87 at threshold=0.5; AUROC=0.94")
6. Limitations — synthetic data caveat, generalisation risk, recommended real-data validation steps

**Template approach:** `evaluate_models.py` builds the markdown string programmatically and writes it — so the report is always in sync with the latest run.

---

## Phase 3: ML Documentation (Model Governance)

**Goal:** Produce an auditor-grade model governance document.  
**Deliverable:** `MODEL_GOVERNANCE.md` + cross-validation metrics in `train.py`

---

### Task 3.1 — Add K-Fold Cross-Validation to GBDT Training

**File:** `src/gbdt_detector.py`, `train.py`  
**Approach:** The GBDT is the primary classifer for the 90% claim. Add k=5 stratified cross-validation to its training routine and persist fold results.

**Steps:**
1. In `train_gbdt()` in `src/gbdt_detector.py`, add optional `cv=5` parameter
2. When `cv > 1`, loop over 5 stratified folds using `train_test_split_manual` with different seeds (or implement a simple stratified k-fold splitter)
3. Record per-fold: accuracy, precision, recall, F1, AUC
4. Compute mean ± std across folds
5. Save to `models/gbdt_cv_results.json`:
   ```json
   {
     "n_folds": 5,
     "folds": [...],
     "mean_accuracy": 0.91,
     "std_accuracy": 0.02,
     "mean_f1": 0.87,
     "std_f1": 0.03
   }
   ```
6. Print summary table to stdout during training

---

### Task 3.2 — Write MODEL_GOVERNANCE.md

**File:** `MODEL_GOVERNANCE.md`

**Sections:**

**1. Overview**
- Purpose of the document (regulatory and client due diligence)
- Models covered: GBDT, GNN, LSTM Link Predictor, Sequence Detector, PPO Adversarial Agent

**2. Training Data**
- Source: synthetic data generated by `src/simulator.py` / `transaction_simulator.py`
- Rationale for synthetic data: no access to real labelled AML transactions; synthetic allows controlled fraud topology injection
- Fidelity: what real-world distributions it approximates (log-normal amount distributions, Zipf-like account frequency, FATF typologies)
- Limitations: model may not generalise to unseen real-world fraud patterns; recommended pilot with labelled real data before production
- Comparison to public datasets: IEEE-CIS Fraud Detection (card-not-present, different features), Elliptic Bitcoin Dataset (graph-based, closest analogue), PaySim (synthetic mobile money — most similar methodology)

**3. Validation Methodology (per model)**

| Model | Split | CV | Metric |
|-------|-------|----|--------|
| GBDT | 80/20 stratified | 5-fold | F1, AUROC |
| GNN | 80/20 node-level | — | Per-class accuracy |
| LSTM | 80/20 temporal | — | AUC on link pairs |
| Sequence | 80/20 | — | Sequence-level accuracy |

**4. Hyperparameter Choices (with rationale)**
- GBDT: `num_leaves=31` (balanced bias/variance for tabular data), `learning_rate=0.05` (conservative for financial data)
- GNN: `hidden=64` (sufficient capacity for 12 input features), `num_layers=2` (empirically stable)
- LSTM: `hidden_size=128`, `num_layers=2` (standard for sequence tasks of this length)
- PPO: standard Schulman et al. 2017 hyperparameters

**5. Known Limitations**
- Synthetic training data — real-world drift unknown
- Class imbalance: ~3% fraud rate used (real AML may be 0.1%)
- Cold-start: LSTM requires ≥14 days history per account
- Cycle detection bounded at `max_k=6` — deep layering schemes may evade

**6. Recommended Real-Data Validation Steps**
- Pilot on 3 months of labelled historical transactions from Captivus
- Retrain GBDT with real labels; freeze GNN topology weights, fine-tune classifier head
- Re-evaluate against this document's methodology

---

## Phase 4: Scalability — Load Test and TPS Report

**Goal:** Measure and document transactions per second under realistic and peak load.  
**Deliverable:** `scripts/load_test.py` + `LOAD_TEST_REPORT.md`

---

### Task 4.1 — Define Peak Load Target

Before writing the test, document the assumed SLA:

| Metric | Target |
|--------|--------|
| Normal sustained load | 100 TPS (transactions/second) |
| Peak load | 500 TPS |
| Max acceptable latency (p99) | 500ms |
| Max acceptable error rate | <1% at peak |

These are reasonable assumptions for a financial crime monitoring system; confirm with Captivus if different.

---

### Task 4.2 — Write Load Test Script

**File:** `scripts/load_test.py`  
**Approach:** Pure Python load tester using `threading` and `requests` (no external tool dependency). Optionally supports Locust if installed.

**Structure:**
```python
class LoadTester:
    def __init__(self, base_url, api_key, n_workers, duration_sec)
    def _worker(self, results_queue)          # fires requests in a loop
    def run(self)                             # launches threads, collects results
    def report(self, results) -> dict         # computes p50/p95/p99, TPS, error rate
    def write_report(self, report, path)      # writes LOAD_TEST_REPORT.md
```

**Test scenarios:**
1. **Baseline** — 1 worker, 60 seconds → establish single-thread TPS
2. **Sustained** — 10 concurrent workers, 120 seconds → normal load
3. **Peak** — 50 concurrent workers, 60 seconds → peak load
4. **Stress** — 100 concurrent workers, 30 seconds → find break point

**Endpoints tested:**
- `POST /score/transaction` — lightweight, should be fastest
- `POST /consolidate/risks` — heavyweight, full pipeline

**Metrics captured per scenario:**
- Total requests, successful requests, error count, error rate %
- Latency: min, mean, p50, p95, p99, max (ms)
- Achieved TPS (requests/second)
- Memory and CPU (via `psutil`, sampled every 5 seconds)

**Steps:**
1. Add `psutil` and `requests` to `requirements.txt` (requests likely already present)
2. Write `scripts/load_test.py` with the structure above
3. Include a `--dry-run` mode that fires 10 requests to verify connectivity before the full test
4. Save raw per-request timings to `outputs/load_test_raw.csv`
5. Save summary to `outputs/load_test_summary.json`

---

### Task 4.3 — Write LOAD_TEST_REPORT.md

**File:** `LOAD_TEST_REPORT.md` (auto-generated by load test script)

**Sections:**
1. Test Environment — hardware (CPU, RAM, OS), Python version, model versions tested
2. Test Methodology — scenarios, endpoints, duration, tool used
3. Results Table — one row per scenario with all metrics
4. Bottleneck Analysis — which phase is the slowest (GBDT scoring, GNN embedding, cycle detection)
5. Scale-Out Strategy:
   - Horizontal: run multiple API worker processes behind a load balancer (gunicorn multi-worker)
   - Vertical: GPU acceleration for GNN and LSTM (PyTorch CUDA)
   - Caching: Redis embedding cache for repeated account lookups
   - Async: Celery task queue for full-pipeline consolidation (fire-and-forget with webhook callback)
6. Recommendations — what changes would be needed to reach 500 TPS sustainably

---

## File Change Summary

| File | Action | Phase |
|------|--------|-------|
| `src/inference_api.py` | Add auth decorator, rate limiting, Pydantic validation, audit log calls | 1 |
| `src/api_schemas.py` | **New** — Pydantic request models for all endpoints | 1 |
| `src/metrics_logger.py` | Add `audit_log` table + `log_audit_event()` + WAL mode | 1 |
| `.env.example` | **New** — template env file | 1 |
| `configs/nginx_example.conf` | **New** — nginx TLS config example | 1 |
| `SECURITY_ARCHITECTURE.md` | **New** — security overview document | 1 |
| `requirements.txt` | Add: pydantic, Flask-Limiter, Flask-Talisman, python-dotenv, psutil | 1 & 4 |
| `scripts/generate_eval_dataset.py` | **New** — generates labelled held-out test set | 2 |
| `evaluate_models.py` | **New** — ML evaluation harness, auto-generates report and plots | 2 |
| `src/gbdt_detector.py` | Add `f1_score_manual()`, `avg_precision_manual()` functions | 2 |
| `MODEL_EVALUATION_REPORT.md` | **Auto-generated** by `evaluate_models.py` | 2 |
| `outputs/roc_curves.png` | **Auto-generated** | 2 |
| `outputs/pr_curves.png` | **Auto-generated** | 2 |
| `outputs/metrics_comparison.png` | **Auto-generated** | 2 |
| `src/gbdt_detector.py` | Add k=5 CV to `train_gbdt()` | 3 |
| `MODEL_GOVERNANCE.md` | **New** — auditor-grade model governance document | 3 |
| `scripts/load_test.py` | **New** — load test script | 4 |
| `LOAD_TEST_REPORT.md` | **Auto-generated** by load test script | 4 |

---

## Dependencies to Add (requirements.txt)

```
pydantic>=2.0
Flask-Limiter>=3.5
Flask-Talisman>=1.1
python-dotenv>=1.0
psutil>=5.9
```

---

## Acceptance Criteria (per Captivus requirement)

| Requirement | Done when... |
|-------------|-------------|
| ML documentation | `MODEL_GOVERNANCE.md` exists, covers all 5 models, includes CV results in `models/gbdt_cv_results.json`, references `eval_dataset_metadata.json` |
| 90%+ accuracy report | `evaluate_models.py` runs to completion, `MODEL_EVALUATION_REPORT.md` exists with per-model F1/AUROC/confusion matrix, 3 plots in `outputs/`, `evaluation_results.json` saved |
| Scalability | `scripts/load_test.py --dry-run` passes, full test completes, `LOAD_TEST_REPORT.md` exists with TPS figure for `/score/transaction` and `/consolidate/risks` |
| Security | All API endpoints return 401 without `X-API-Key`, 422 on invalid payload, 429 after rate limit breach, every request logged to `audit_log` table, `SECURITY_ARCHITECTURE.md` exists |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GBDT F1 below 90% on held-out test | Medium | High | Tune threshold, document actual metric, clarify "90% accuracy" was overall accuracy not F1 |
| GNN/LSTM scoring too slow under load | Medium | Medium | Cache embeddings, reduce test load target, document GPU requirement |
| Synthetic data not representative | High | Medium | Add explicit disclaimer to governance doc, recommend Captivus pilot |
| Flask-Limiter incompatible with current Flask version | Low | Low | Pin versions, test in dev before deploy |
| Load test exhausts memory running all models | Medium | Low | Run `/score/transaction` (GBDT only) first for initial TPS figure |
