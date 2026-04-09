# AML Topology Detection System — Complete Architecture

## 1. Project Overview

The **AML Topology Detection System** is a production-grade Anti-Money Laundering (AML)
framework for identifying suspicious financial patterns and money-laundering typologies in
transaction networks. It combines rule-based graph analysis, temporal forecasting, deep-learning
classifiers, and reinforcement-learning-based adversarial stress testing into a unified detection
pipeline with a REST API and interactive dashboard.

---

## 2. High-Level System Map

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          AML TOPOLOGY DETECTION SYSTEM                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────┐
 │   DATA INGESTION     │  simulator.py / transaction_simulator.py / transactions.csv
 └──────────┬───────────┘
            │ Pandas DataFrame: [source, target, amount, timestamp, ...]
            ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                          DETECTION PIPELINE                                      │
 │                                                                                  │
 │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐   │
 │  │  PHASE 2        │  │  PHASE 3         │  │  PHASE 4                     │   │
 │  │  SPATIAL        │  │  TEMPORAL        │  │  ML / DEEP LEARNING          │   │
 │  │  (Detective)    │  │  (Predictive)    │  │                              │   │
 │  ├─────────────────┤  ├──────────────────┤  ├──────────────────────────────┤   │
 │  │ graph_analyzer  │  │temporal_predictor│  │ lstm_link_predictor          │   │
 │  │ csr_cycle_det.  │  │sequence_detector │  │ gnn_trainer                  │   │
 │  │ behavioral_det. │  │                  │  │ gbdt_detector                │   │
 │  │ pattern_library │  │                  │  │ adversarial_agent (PPO)      │   │
 │  └────────┬────────┘  └────────┬─────────┘  └──────────────┬───────────────┘   │
 │           │                    │                             │                   │
 │           └────────────────────┼─────────────────────────── ┘                   │
 │                                ▼                                                 │
 │                    ┌───────────────────────┐                                     │
 │                    │  PHASE 5              │                                     │
 │                    │  RISK CONSOLIDATION   │  risk_consolidator.py               │
 │                    └──────────┬────────────┘                                     │
 │                               │                                                  │
 │                    ┌──────────▼────────────┐                                     │
 │                    │  PHASE 6              │                                     │
 │                    │  OUTPUT & SERVING     │                                     │
 │                    │  REST API / Dashboard │  inference_api.py / dashboard.py    │
 │                    └───────────────────────┘                                     │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
aml-topology/
│
├── main.py                          # Orchestrates full 5-phase detection pipeline
├── train.py                         # Trains all ML models (LSTM, GNN, GBDT, PPO)
├── inference.py                     # Batch inference pipeline (loads persisted models)
├── inference_api.py                 # Flask REST API server (root-level entry point)
├── dashboard.py                     # Streamlit interactive dashboard
├── pipeline_simulation.py           # End-to-end simulation runner
├── adversarial_retrain.py           # Adversarial retraining loop
├── transaction_simulator.py         # Alternative simulator entry point
├── requirements.txt                 # Python dependencies
│
├── src/                             # Core modules (21 files, ~6,800 lines)
│   ├── simulator.py                 # Synthetic transaction data generation
│   ├── graph_analyzer.py            # Fan-in, fan-out, centrality detection
│   ├── csr_cycle_detector.py        # CSR-based efficient cycle detection
│   ├── behavioral_detector.py       # Cyber-behavioral anomaly detection
│   ├── pattern_library.py           # Pattern definitions and rule thresholds
│   ├── temporal_predictor.py        # Temporal forecasting & baseline analysis
│   ├── sequence_detector.py         # Transaction sequence pattern detection
│   ├── lstm_link_predictor.py       # LSTM for emerging link prediction
│   ├── gnn_trainer.py               # Graph Neural Network trainer & classifier
│   ├── gbdt_detector.py             # LightGBM/XGBoost GBDT transaction scorer
│   ├── adversarial_agent.py         # PPO adversarial agent (evasion patterns)
│   ├── adversarial_env.py           # Gymnasium RL environment for adversarial training
│   ├── risk_consolidator.py         # Multi-phase weighted risk aggregation
│   ├── embedding_builder.py         # Time-series node embedding construction
│   ├── gnn_embedding_cache.py       # File + Redis embedding cache
│   ├── inference_api.py             # Flask REST API (module-level)
│   ├── inference_client.py          # HTTP client for the REST API
│   ├── metrics_logger.py            # Structured logging & metrics (SQLite)
│   ├── visualizer.py                # NetworkX graph visualization
│   └── run_lstm_demo.py             # LSTM standalone demo runner
│
├── models/                          # Persisted model artefacts
│   ├── lstm_link_predictor.pt       # LSTM weights (PyTorch checkpoint)
│   ├── lstm_metadata.json           # LSTM feature/config metadata
│   ├── gnn_adversarial.pt           # GNN model weights
│   ├── gnn_metadata.json            # GNN training metadata
│   ├── adversarial_agent.pt         # PPO agent weights
│   ├── adversarial_metadata.json    # PPO config metadata
│   ├── lgb_model.txt                # LightGBM GBDT model (text format)
│   ├── gbdt_metadata.json           # GBDT feature list & thresholds
│   └── consolidation_config.json    # Risk consolidation weight config
│
├── configs/
│   └── rules.yml                    # Neuro-symbolic rule toggles & weights
│
├── evaluate_models.py               # Model evaluation harness (Phase 4 governance)
│
├── scripts/
│   ├── demo_live.py                 # Live demo runner
│   ├── generate_eval_dataset.py     # Generates fixed held-out test set (seed=2025)
│   ├── generate_gnn_embeddings.py   # Pre-computes GNN embeddings
│   └── test_gnn_integration.py      # Integration test for GNN pipeline
│
├── data/
│   ├── eval_dataset.csv             # Held-out test set (10,001 rows, seed=2025)
│   └── eval_dataset_metadata.json   # Checksum, fraud rate, train/test split metadata
│
├── outputs/                         # Generated evaluation outputs
│   ├── evaluation_results.json      # Per-model metrics from evaluate_models.py
│   ├── roc_curves.png               # ROC curves — all models
│   ├── pr_curves.png                # Precision-Recall curves — all models
│   ├── metrics_comparison.png       # F1/Precision/Recall bar chart
│   └── confusion_matrix_<model>.png # Per-model confusion matrices
│
├── MODEL_GOVERNANCE.md              # Auditor-grade governance doc (v1.1)
├── MODEL_EVALUATION_REPORT.md       # Auto-generated evaluation report
├── metrics.db                       # SQLite metrics store
├── simulation_pipeline_results.csv  # Pipeline run results
├── consolidated_risk_scores.csv     # Final risk rankings output
├── transactions.csv                 # Sample/exported transaction data
└── aml_network_graph.png            # Example graph visualization output
```

---

## 4. Module Descriptions

### 4.1 Data Ingestion

| File | Class / Function | Responsibility |
|---|---|---|
| `src/simulator.py` | `TransactionSimulator` | Generates synthetic transaction DataFrames with injected fraud patterns (fan-in, fan-out, cycles) |
| `transaction_simulator.py` | — | Root-level alias / extended simulator entry point |
| `transactions.csv` | — | Static sample or exported transaction dataset |

**Output schema:** `source, target, amount, timestamp, [mcc, payment_type, channel, ...]`

---

### 4.2 Phase 2 — Spatial / Detective Analysis

Detects **current** suspicious graph structures.

#### `src/graph_analyzer.py` — `AMLGraphAnalyzer`

| Method | Pattern | AML Typology |
|---|---|---|
| `detect_fan_in()` | N→1 flows | Structuring / Placement |
| `detect_fan_out()` | 1→N flows | Integration / Dissipation |
| `calculate_centrality()` | Bridge nodes | Layering intermediaries |
| `get_node_stats()` | Per-node in/out flows | General profiling |

Uses **NetworkX** directed graph. Graph is rebuilt per run from the transaction DataFrame.

#### `src/csr_cycle_detector.py` — CSR Cycle Detection

- Builds a **Compressed Sparse Row (CSR)** adjacency representation for memory efficiency.
- Runs bounded DFS cycle detection (`max_k=6`, `max_cycles=500`) to avoid exponential blowup.
- Canonicalises cycles to deduplicate rotations.
- Detects **layering / round-tripping** patterns (A→B→C→A).

#### `src/behavioral_detector.py` — `BehavioralDetector`

Cyber-behavioral detection (10+ sub-detectors):

| Detector | Signal |
|---|---|
| `detect_credential_stuffing()` | High failed-login ratio |
| `detect_low_and_slow()` | Extended brute-force campaigns |
| `detect_bruteforce_and_new_device()` | Attack followed by new device login |
| `detect_post_compromise()` | Rapid post-login transfers |
| `detect_bust_out()` | Sudden credit drain |
| `detect_channel_hopping()` | Multi-channel activity in short window |
| `detect_impossible_travel()` | Geographically impossible login sequence |
| `detect_app_fraud()` | Payee mismatch, unusual country/hour |
| `detect_synthetic_identity()` | Thin-file + sudden utilisation spike |
| `detect_money_mules()` | Multi-homing, dormancy-then-spike |
| `compare_fingerprints()` | Device fingerprint deviation |
| `analyze_biometrics()` | Behavioural biometric anomalies |

#### `src/pattern_library.py` — `PatternLibrary` (838 lines)

Central registry of pattern definitions, rule thresholds, and regulatory reference mappings.  
Consumed by graph_analyzer, behavioral_detector, and risk_consolidator.

---

### 4.3 Phase 3 — Temporal / Predictive Analysis

Detects **emerging** trends and forecasts future risk.

#### `src/temporal_predictor.py` — `TemporalPredictor` + `SequenceAnalyzer`

**`TemporalPredictor`** establishes per-account baselines and runs 5 detectors:

| Method | What it detects | Horizon |
|---|---|---|
| `establish_baselines()` | Per-account normal metrics (avg amount, frequency, counterparties) | Lookback |
| `detect_volume_acceleration()` | Rapid growth vs. baseline (sigma-threshold) | Now |
| `detect_behavioral_shift()` | Multi-dimension deviation from baseline | Now |
| `forecast_risk_escalation()` | Bayesian multi-signal probability aggregation | +7 days |
| `detect_temporal_concentration()` | Transaction bursts within time windows | Now |
| `predict_cycle_emergence()` | Bidirectional relationship formation → cycle precursor | +14 days |
| `forecast_account_summary()` | Full temporal forecast report per account | Combined |

**Key algorithm — Bayesian signal combination:**
```
P(risk) = 1 − ∏(1 − pᵢ)  for independent signals p₁…pₙ
```

**`SequenceAnalyzer`**

| Method | Pattern |
|---|---|
| `detect_structuring_sequence()` | Repeated just-below-threshold transactions (CTR evasion) |

---

### 4.4 Phase 4 — Machine Learning & Deep Learning

#### `src/lstm_link_predictor.py` — `LSTMLinkPredictor`

| Property | Detail |
|---|---|
| Architecture | 2-layer LSTM + sigmoid output head |
| Input | Time-series node embeddings `(seq_len × feature_dim)` |
| Output | Link probability score `[0, 1]` |
| Loss | `BCEWithLogitsLoss` with class-weight balancing |
| Training | Early stopping on validation AUC |
| Purpose | Predict **emerging financial relationships** before they form spatial patterns |
| Persistence | `models/lstm_link_predictor.pt` + `lstm_metadata.json` |

#### `src/gnn_trainer.py` — Graph Neural Network (1,009 lines)

- Generates synthetic transaction graphs with planted fraud topologies (mule hubs, synthetic identities, APP fraud, smurfing rings).
- Trains a GNN for **node classification** — identifying money-mule accounts.
- Produces per-node risk embeddings reusable by downstream modules.
- Persistence: `models/gnn_adversarial.pt` + `gnn_metadata.json`.

#### `src/gbdt_detector.py` — `GBDTDetector`

| Property | Detail |
|---|---|
| Frameworks | LightGBM (default) → XGBoost → CatBoost → PyTorch MLP (fallback) |
| Feature engineering | 21 features across 5 groups (see table below) |
| Training | Stratified split + 5-fold CV; class-weight balancing via `scale_pos_weight` |
| Threshold | F1-optimal per fold (`_find_best_threshold`); no fixed 0.5 cutoff |
| Output | Per-transaction fraud probability score |
| Persistence | `models/lgb_model.txt` + `models/gbdt_metadata.json` + `models/gbdt_cv_results.json` |

**Feature groups (21 total):**

| Group | Features | Notes |
|---|---|---|
| Amount | `amt_log`, `amt_gt_1k`, `amt_gt_3k` | Log-transform + binary thresholds at $1k/$3k |
| Categorical | `mcc_enc`, `payment_type_enc`, `is_crypto` | Ordinal encoding + explicit crypto flag |
| Device / IP | `device_change`, `ip_risk`, `ip_risk_high`, `ip_risk_mid` | Continuous + bucketised risk bands |
| Velocity | `count_1h`, `sum_24h`, `uniq_payees_24h`, `high_velocity`, `avg_tx_24h`, `velocity_score` | Transaction rate and volume over sliding windows |
| Geography | `is_international`, `is_high_risk_country` | High-risk jurisdictions: RU, NG, PK, CN |
| Interactions | `amt_x_ip_risk`, `high_amt_device_change`, `crypto_high_risk_country` | Cross-feature fraud signals |

**Key LightGBM hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `num_boost_round` | 500 | More rounds with lower LR for smoother convergence |
| `num_leaves` | 63 | Deep enough to capture interaction features |
| `learning_rate` | 0.03 | Conservative; pairs with 500 rounds |
| `metric` | `auc` | Optimise AUROC directly under class imbalance |
| `scale_pos_weight` | neg/pos ratio | Balances ~10% fraud minority class |
| `feature_fraction` | 0.8 | Column subsampling for regularisation |

**Exported functions:**

| Function | Purpose |
|---|---|
| `generate_synthetic_transactions(n, seed)` | Generates transactions with feature-correlated fraud labels via `_compute_fraud_risk()` |
| `featurize(df)` | Builds feature matrix + encoding maps from a dataframe |
| `apply_featurize(df, maps)` | Applies pre-computed encoding maps to a new dataframe (use for test/inference sets) |
| `train_gbdt(X, y, cv=5)` | Trains model with optional k-fold CV; saves weights and CV results |
| `score_transaction(tx, maps, model)` | Scores a single transaction dict; returns probability |
| `_find_best_threshold(y, scores)` | Sweeps thresholds to find F1-maximising cutoff |

#### `src/adversarial_agent.py` + `src/adversarial_env.py` — PPO Adversarial Agent

| Property | Detail |
|---|---|
| Algorithm | Proximal Policy Optimisation (PPO) |
| Action space | Hybrid: discrete pattern type (5) × temporal mode (3) × chain length (16) + continuous peel_pct / wash_intensity / cycle_probability via Beta distributions |
| Pattern types | peel_chain, forked_peel, smurfing, funnel, mule |
| Purpose | Stress-test the detector by generating realistic evasion patterns; used to harden the model via adversarial retraining |
| Persistence | `models/adversarial_agent.pt` + `adversarial_metadata.json` |

#### `src/sequence_detector.py` — `SequenceDetector` (323 lines)

- Encodes transaction events to integer IDs.
- Uses temporal windowing to match multi-step suspicious sequences.
- Deep sequence model (LSTM/Transformer) for learned pattern matching.

---

### 4.5 Phase 5 — Risk Consolidation

#### `src/risk_consolidator.py` — `RiskConsolidator`

Aggregates signals from all upstream phases into a single ranked risk score.

**Default phase weights** (configurable via `models/consolidation_config.json`):

| Phase | Weight | Signals Included |
|---|---|---|
| Spatial | 20 % | Cycles, fan-in/out, centrality |
| Behavioral | 10 % | Cyber/device alerts |
| Temporal | 35 % | Volume acceleration, shifts, concentration, cycle emergence |
| LSTM | 25 % | Emerging link predictions |
| Cyber | 10 % | Biometric anomalies |

**Aggregation steps:**
1. Normalise each phase score to `[0, 1]`.
2. Apply weights and compute weighted sum.
3. Apply `15 %` boost when ≥ 2 phases flag the same account.
4. Rank all accounts by final score.
5. Emit priority tiers: `CRITICAL (≥75)`, `HIGH (60-75)`, `MEDIUM (40-60)`, `LOW (20-40)`.

---

### 4.6 Phase 6 — Output & Serving

#### `src/inference_api.py` / `inference_api.py` — Flask REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/score/transaction` | POST | GBDT per-transaction fraud score |
| `/analyze/sequence` | POST | Sequence pattern analysis |
| `/predict/links` | POST | LSTM emerging-link probabilities |
| `/graph/analyze` | POST | GNN-based node classification |
| `/consolidate/risks` | POST | Multi-phase risk aggregation |
| `/health` | GET | Service health check |

- Models are loaded once at startup and cached in memory.
- Responses are standardised JSON with `risk_score`, `signals`, and `priority_tier`.

#### `dashboard.py` — Streamlit Dashboard

- Interactive network graph with risk-coloured nodes.
- Side panels for account-level drill-down.
- Phase-by-phase signal attribution.
- Launched via `streamlit run dashboard.py`.

#### `src/visualizer.py` — `AMLVisualizer`

- NetworkX + Matplotlib graph rendering.
- Suspicious nodes highlighted in red; bridge nodes in orange.
- Exports to PNG/SVG.

---

## 5. Supporting Modules

| Module | Class | Purpose |
|---|---|---|
| `src/embedding_builder.py` | `EmbeddingBuilder` | Builds time-series node embeddings (frequency, volume, counterparty diversity, temporal stats) for LSTM input |
| `src/gnn_embedding_cache.py` | `GNNEmbeddingCache` | File-based + optional Redis cache for GNN embeddings; tracks hit/miss statistics |
| `src/metrics_logger.py` | `MetricsLogger` | Structured logging to SQLite (`metrics.db`); records detection events, model metrics, latency |
| `src/inference_client.py` | `InferenceClient` | Python HTTP client for the REST API; used by scripts and tests |

---

## 6. Data Flow (End-to-End)

```
Raw Transactions (CSV / DataFrame)
          │
          ▼
  ┌───────────────┐
  │  Validation   │  Normalise timestamps, validate required columns
  └───────┬───────┘
          │
   ┌──────┴──────────────────────────────────────────────┐
   │                                                      │
   ▼                                                      ▼
┌────────────────────┐                      ┌─────────────────────────┐
│    SPATIAL PATH    │                      │     TEMPORAL PATH        │
│                    │                      │                          │
│ 1. Build graph     │                      │ 1. Establish baselines   │
│ 2. detect_fan_in   │                      │ 2. Volume acceleration   │
│ 3. detect_fan_out  │                      │ 3. Behavioral shift      │
│ 4. detect_cycles   │                      │ 4. Risk escalation       │
│ 5. centrality      │                      │ 5. Temporal concentration│
│ 6. behavioral det. │                      │ 6. Cycle emergence pred. │
│                    │                      │ 7. Structuring sequences │
│ Output: alert list │                      │ Output: risk forecasts   │
└──────────┬─────────┘                      └───────────┬─────────────┘
           │                                            │
           └───────────────────┬────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │     ML PATH         │
                    │                     │
                    │ 1. Build embeddings │  embedding_builder.py
                    │ 2. LSTM predict     │  lstm_link_predictor.py
                    │ 3. GBDT score       │  gbdt_detector.py
                    │ 4. GNN classify     │  gnn_trainer.py
                    │ 5. Adversarial test │  adversarial_agent.py
                    │                     │
                    │ Output: prob. scores│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  RISK CONSOLIDATION │  risk_consolidator.py
                    │                     │
                    │ 1. Normalise phases │
                    │ 2. Apply weights    │
                    │ 3. Multi-flag boost │
                    │ 4. Rank accounts   │
                    │ 5. Assign tier     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
        REST API          Dashboard         CSV Export
    (inference_api)     (dashboard.py)  (consolidated_risk
                                          _scores.csv)
```

---

## 7. Machine Learning Models

| Model | File | Framework | Task | Held-out F1 | Held-out AUROC |
|---|---|---|---|---|---|
| LSTM Link Predictor | `models/lstm_link_predictor.pt` | PyTorch | Predict emerging financial links | 0.9895 | 0.9990 |
| GNN Node Classifier | `models/gnn_adversarial.pt` | PyTorch | Classify mule/fraud nodes | 0.9121 | 0.6526 |
| GBDT Transaction Scorer | `models/lgb_model.txt` | LightGBM | Per-transaction fraud probability | 0.4730 | 0.8799 |
| Sequence Detector | `models/sequence_detector_model.pt` | PyTorch | Event-sequence anomaly detection | 1.0000 | 1.0000 |
| PPO Adversarial Agent | `models/adversarial_agent.pt` | PyTorch | Generate evasion patterns for hardening | — | — |

Metrics are from `outputs/evaluation_results.json` (seed=2025, 10,001-row held-out test set). Sequence Detector and LSTM metrics are on independent synthetic test sets — see `MODEL_EVALUATION_REPORT.md` for caveats.

All models are accompanied by `*_metadata.json` files capturing feature lists, hyperparameters, training-time metrics, and version tags. GBDT additionally produces `models/gbdt_cv_results.json` with per-fold cross-validation results.

---

## 8. Detection Typologies

### Placement (Structuring)
- Fan-in: Multiple sources aggregating into a single account
- Volume acceleration: Rapid growth in transaction volumes
- Structuring sequences: Repeated just-below-CTR-threshold transactions ($9,000–$9,900)
- Amount clustering around the $10,000 FinCEN reporting threshold

### Layering (Obfuscation)
- Cycles: Circular flows A→B→C→A detected via CSR DFS
- Cycle emergence prediction: Bidirectional relationship formation → future cycle
- Temporal clustering: Coordinated bursts across accounts
- Bridge nodes: High betweenness centrality actors connecting subgraphs

### Integration (Dispersion)
- Fan-out: Single source disbursing to many targets
- Temporal concentration: High activity in narrow time windows
- Bust-out: Rapid credit drain after account establishment

### Account Compromise / Behavioral
- Credential stuffing, brute-force + new device
- Impossible travel, device fingerprint deviation
- Channel hopping across payment methods
- Behavioral biometric anomalies

### Advanced / Synthetic
- Synthetic identity: Thin file + sudden utilisation spike
- APP fraud: Authorised push payment with payee mismatch
- Money mules: Multi-homing, dormancy-then-spike
- PPO-generated evasion: Peel chains, forked peels, nested cycles

---

## 9. Risk Consolidation Logic

```python
# Per-phase scores normalised to [0, 1]
weights = {
    'spatial':    0.20,
    'behavioral': 0.10,
    'temporal':   0.35,
    'lstm':       0.25,
    'cyber':      0.10,
}

raw_score = sum(weights[phase] * scores[phase] for phase in weights)

# 15% boost if >= 2 phases flag the same account
if phases_flagging >= 2:
    final_score = min(raw_score * 1.15, 1.0)
else:
    final_score = raw_score

# Priority tiers
tier = (
    'CRITICAL' if final_score >= 0.75 else
    'HIGH'     if final_score >= 0.60 else
    'MEDIUM'   if final_score >= 0.40 else
    'LOW'
)
```

---

## 10. Temporal Forecasting Detail

The temporal predictor establishes **per-account baselines** over a configurable lookback window, then measures deviations in sigma units.

**Configuration profiles:**

| Profile | `lookback_days` | `threshold_sigma` | `deviation_threshold` | `early_warning_threshold` |
|---|---|---|---|---|
| Conservative | 60 | 3.0 | 3.0 | 0.8 |
| Balanced (default) | 30 | 2.5 | 2.0 | 0.6 |
| Aggressive | 14 | 2.0 | 1.5 | 0.4 |

**Early warning window:** Temporal signals typically emerge **7–21 days** before spatial patterns become detectable.

```
Day 1–7:   Temporal: "Volume accelerating, behavioral shift detected"
Day 8–15:  Spatial:  No patterns yet
Day 16–21: Temporal: "Risk escalation forecast 78%"
Day 22–28: Spatial:  "Fan-in pattern now visible"
```

---

## 11. Adversarial Training Loop

```
┌──────────────────────────────────────────────────────┐
│  ADVERSARIAL HARDENING CYCLE                         │
│                                                      │
│  1. PPO Agent generates evasion transaction graph    │
│  2. Detection pipeline evaluates the graph           │
│  3. Agent reward = −(detection score)                │
│     (rewarded for evading detection)                 │
│  4. Agent updates policy via PPO clipping            │
│  5. Detected evasion patterns added to training set  │
│  6. GBDT + GNN retrained on augmented dataset        │
│  7. Repeat                                           │
│                                                      │
│  Entry: adversarial_retrain.py                       │
└──────────────────────────────────────────────────────┘
```

**PPO Action Space:**

| Dimension | Type | Options |
|---|---|---|
| Pattern type | Discrete | peel_chain, forked_peel, smurfing, funnel, mule |
| Temporal mode | Discrete | rapid_fire, evasion_pause, mixed |
| Chain length | Discrete | 5–20 hops (16 options) |
| Peel percentage | Continuous (Beta) | 2–15 % |
| Wash intensity | Continuous (Beta) | 0–1 |
| Cycle probability | Continuous (Beta) | 0–1 |

Beta distributions are used for bounded continuous actions to prevent gradient saturation.

---

## 12. Embedding Pipeline

```
Transaction DataFrame
        │
        ▼
EmbeddingBuilder.build()
  - Per-node time-series aggregation
  - Features: tx_frequency, avg_amount, std_amount,
              unique_counterparties, temporal_entropy,
              hour_distribution, day_distribution
        │
        ▼
GNNEmbeddingCache (file / Redis)
  - SHA-256 cache key on node_id + window
  - Hit: return cached tensor
  - Miss: compute → store → return
        │
        ▼
LSTMLinkPredictor.predict_proba()  /  GNNTrainer.classify()
```

---

## 13. Entry Points

| Script | Purpose | Usage |
|---|---|---|
| `main.py` | Full pipeline (simulation → detection → viz) | `python main.py` |
| `train.py` | Train / retrain all ML models | `python train.py` |
| `inference.py` | Batch inference on new transactions | `python inference.py` |
| `inference_api.py` | Start Flask REST API server | `python inference_api.py` |
| `dashboard.py` | Launch Streamlit dashboard | `streamlit run dashboard.py` |
| `adversarial_retrain.py` | Run adversarial hardening loop | `python adversarial_retrain.py` |
| `pipeline_simulation.py` | Simulate full pipeline run | `python pipeline_simulation.py` |
| `scripts/generate_gnn_embeddings.py` | Pre-compute GNN embeddings | `python scripts/generate_gnn_embeddings.py` |

---

## 14. Key Dependencies

```
Core
  pandas, numpy              Data manipulation
  networkx                   Graph construction and algorithms

Deep Learning
  torch, torch.nn            LSTM, GNN, PPO models
  torch_geometric (opt.)     Graph neural network layers

Gradient Boosting
  lightgbm                   Primary GBDT backend
  xgboost                    Secondary GBDT backend
  catboost                   Tertiary GBDT backend

Reinforcement Learning
  gymnasium                  PPO environment interface

Web Services
  flask                      REST API server
  streamlit                  Interactive dashboard
  requests                   HTTP client

Caching & Scheduling
  redis                      Distributed embedding cache (optional)
  apscheduler                Job scheduling

Visualisation
  matplotlib, plotly         Graph and chart rendering

Utilities
  six                        Python 2/3 compatibility shim
  pyyaml                     rules.yml config parsing
  sqlite3 (stdlib)           metrics.db logging store
```

---

## 15. Complexity & Performance

| Operation | Complexity | ~Time (100k txs) |
|---|---|---|
| Graph construction | O(n) | < 100 ms |
| Fan-in / fan-out | O(n log n) | 200–400 ms |
| CSR cycle detection | O(n + e), bounded | 1–2 s |
| Temporal baselines | O(n) | 200 ms |
| GBDT batch scoring | O(depth × features) | 10–20 ms / batch |
| LSTM inference | O(seq_len × hidden) | 50–100 ms / batch |
| GNN node classification | O(nodes × layers) | 500 ms – 2 s |
| Full pipeline | O(n × k), k ≈ 8 | 2–5 s total |

**Memory footprint:**
- Baselines: ~1 KB / account
- Loaded models: ~1.3 GB (all four models)
- Working graph: ~20 MB / 100k transactions

---

## 16. Configuration Reference

### `configs/rules.yml`
```yaml
fan_in_smurf:     { enabled: true,  weight: 1.0 }
mule_hub:         { enabled: true,  weight: 1.0 }
impossible_travel:{ enabled: false, weight: 1.0 }
# Add / toggle any pattern from pattern_library.py
```

### `models/consolidation_config.json`
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

## 17. Regulatory Alignment

| Standard | Relevance |
|---|---|
| FinCEN CTR ($10,000 threshold) | Structuring detection, just-below-threshold sequences |
| FATF Risk-Based Approach | Parameterised thresholds per risk profile |
| AML/CFT Best Practices | Multi-layer detection with early warning |
| GDPR / Data Minimisation | Embeddings do not store raw PII |

---

## 18. Known Limitations

| Limitation | Detail |
|---|---|
| Cycle detection bound | `max_k=6`, `max_cycles=500` — long chains or dense graphs may be truncated |
| LSTM cold start | Requires ≥ 14 days of history per account for reliable embeddings |
| GNN memory scaling | Full graph must fit in memory; large networks (> 1M nodes) require batched GNN |
| Baseline recomputation | Baselines are recomputed per run; no persistent database (future work) |
| Streaming mode | Pipeline is batch-oriented; real-time streaming requires additional wrapper |
| GBDT synthetic labels | Fraud labels are derived from a heuristic risk-scoring function, not real SAR filings — `_compute_fraud_risk()` in `gbdt_detector.py`; real-data retraining required for production |
| GBDT threshold portability | F1-optimal threshold is tuned on synthetic data; must be re-derived on real labelled data before deployment |
| Evaluation datasets | All held-out metrics are on synthetic data; Sequence Detector and LSTM use independent test sets that do not share the same transaction universe as the GBDT test set |

---

*Last updated: 2026-04-05*
