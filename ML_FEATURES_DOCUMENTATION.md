# ML Features Documentation

**Complete Feature Inventory for AML Detection System**

This document catalogs all machine learning features used across the 4 trained models in the system.

---

## Table of Contents
1. [GBDT Detector Features](#1-gbdt-detector-features)
2. [GNN (GraphSage) Features](#2-gnn-graphsage-features)
3. [LSTM Link Predictor Features](#3-lstm-link-predictor-features)
4. [Sequence Detector Features](#4-sequence-detector-features)
5. [Risk Consolidator Inputs](#5-risk-consolidator-inputs)

---

## 1. GBDT Detector Features

**Model File:** `src/gbdt_detector.py`  
**Model Type:** LightGBM/XGBoost/CatBoost Gradient Boosted Decision Trees  
**Purpose:** Transaction-level fraud scoring

### Input Features (11 total)

| Feature Name | Type | Description | Range/Values |
|-------------|------|-------------|--------------|
| **amt_log** | Numerical | Log-transformed transaction amount: `log1p(amount)` | [0, inf) |
| **mcc_enc** | Categorical | Encoded Merchant Category Code | 0-7 (mapped indices) |
| **payment_type_enc** | Categorical | Encoded payment method | 0-3 (card, wire, ach, crypto) |
| **device_change** | Binary | Whether device ID changed from baseline | 0/1 |
| **ip_risk** | Numerical | IP address risk score | [0.0, 1.0] |
| **count_1h** | Numerical | Transaction count in past 1 hour | [0, inf) |
| **sum_24h** | Numerical | Sum of transactions in past 24 hours | [0, inf) |
| **uniq_payees_24h** | Numerical | Unique payees in past 24 hours | [0, inf) |
| **is_international** | Binary | International transaction flag | 0/1 |
| **avg_tx_24h** | Derived | `sum_24h / (uniq_payees_24h + 1)` | [0, inf) |
| **velocity_score** | Derived | `count_1h * (amt_log / 10.0)` | [0, inf) |

### Feature Engineering Details

**Raw Input Fields:**
- `amount`, `mcc`, `payment_type`, `device_change`, `ip_risk`
- `count_1h`, `sum_24h`, `uniq_payees_24h`, `country`

**Transformations:**
1. **Log Transform**: `amt_log = log1p(amount)` - handles wide amount ranges
2. **Categorical Encoding**: MCC and payment_type mapped to integer indices
3. **Ratio Feature**: `avg_tx_24h` - average transaction per payee
4. **Velocity Feature**: Combines frequency and amount magnitude
5. **Binary Flag**: `is_international = (country != 'US')`

**Feature Maps Persisted:**
```json
{
  "mcc_map": {"5311": 0, "5411": 1, "6011": 2, ...},
  "pay_map": {"card": 0, "wire": 1, "ach": 2, "crypto": 3}
}
```

---

## 2. GNN (GraphSage) Features

**Model File:** `src/gnn_trainer.py`  
**Model Type:** Graph Neural Network (GraphSAGE with message passing)  
**Purpose:** Node-level fraud detection using graph topology

### Node Features (12 total)

| Feature Name | Type | Description | Normalization |
|-------------|------|-------------|---------------|
| **in_degree** | Graph | Number of incoming edges | Z-score normalized |
| **out_degree** | Graph | Number of outgoing edges | Z-score normalized |
| **pagerank** | Graph | PageRank centrality score | Z-score normalized |
| **betweenness** | Graph | Betweenness centrality | Z-score normalized |
| **in_cycle** | Graph | Binary flag if node is in a cycle | 0/1 |
| **avg_tx_amount** | Transaction | Average transaction amount (outgoing) | Z-score normalized |
| **recv_fraction** | Transaction | Fraction of incoming vs total volume | Z-score normalized |
| **unique_devices** | Behavioral | Count of unique device IDs used | Z-score normalized |
| **unique_ips** | Behavioral | Count of unique IP addresses | Z-score normalized |
| **avg_interarrival** | Temporal | Average time between transactions (seconds) | Z-score normalized |
| **age** | Profile | Account age (days or user age) | Z-score normalized |
| **credit_history_years** | Profile | Years of credit history | Z-score normalized |

### Graph Construction

**Edge Features:**
- `amount`: Transaction amount
- `timestamp`: Unix timestamp
- `channel`: Transaction channel (bank_transfer, p2p, wire, etc.)
- `ip`, `device_id`, `lat`, `lon`: Metadata

**Feature Extraction Process:**

1. **Structural Features** (computed via NetworkX):
   ```python
   in_degree = G.in_degree(node)
   out_degree = G.out_degree(node)
   pagerank = nx.pagerank(G)[node]
   betweenness = nx.betweenness_centrality(G)[node]
   ```

2. **Cycle Detection** (CSR-based):
   ```python
   cycles = detect_cycles_csr(indptr, indices, node_list, max_k=6)
   in_cycle[node] = 1 if node in any cycle else 0
   ```

3. **Transaction Aggregation**:
   ```python
   avg_tx = df_tx.groupby('source')['amount'].mean()
   unique_devices = df_tx.groupby('source')['device_id'].nunique()
   ```

4. **Normalization** (Z-score):
   ```python
   features = (raw - mean) / (std + 1e-6)
   ```

### GNN Architecture Details

- **Input Dimension**: 12 features per node
- **Hidden Dimension**: 64
- **Output Dimension**: 32 embeddings
- **Aggregation**: Mean aggregation of neighbor features
- **Layers**: 2-layer GraphSAGE with concatenation

**Forward Pass:**
```python
h1 = concat(self_features, mean(neighbor_features))
h2 = ReLU(Linear(h1))
output = Sigmoid(Linear(h2))
```

---

## 3. LSTM Link Predictor Features

**Model File:** `src/lstm_link_predictor.py`, `src/embedding_builder.py`  
**Model Type:** Bidirectional LSTM  
**Purpose:** Predict emerging suspicious links between node pairs

### Sequence Features (per timestep: 25 total)

Each node pair has a time-series of embeddings with these features per timestamp:

#### Static Features (8 features)
| Feature | Description |
|---------|-------------|
| `in_degree` | Number of incoming edges |
| `out_degree` | Number of outgoing edges |
| `total_degree` | Sum of in + out degree |
| `pagerank` | PageRank score |
| `clustering_coeff` | Clustering coefficient (undirected) |
| `reciprocal_count` | Count of bidirectional connections |
| `unique_counterparties` | Total unique neighbors |
| `baseline_avg_out_amount` | Historical average outbound amount |

#### Dynamic Features (9 features per time bucket)
| Feature | Description |
|---------|-------------|
| `tx_count_out` | Transactions sent in time bucket |
| `tx_count_in` | Transactions received in time bucket |
| `total_out_amount` | Total amount sent |
| `total_in_amount` | Total amount received |
| `avg_out_amount` | Average outbound transaction |
| `median_out_amount` | Median outbound transaction |
| `std_out_amount` | Std deviation of outbound amounts |
| `unique_out_counterparties_in_bucket` | Unique recipients in bucket |
| `time_since_last_tx` | Seconds since previous transaction |

#### Meta Features (1 feature)
| Feature | Description |
|---------|-------------|
| `fraud_score` | Historical fraud probability [0.0, 1.0] |

### Input Shape

- **Sequences**: `(N, seq_len, feat_dim)`
  - `N`: Number of node pairs
  - `seq_len`: Time sequence length (typically 5-10 timesteps)
  - `feat_dim`: Feature dimension (25 features per node, 50 for pair)

**Pair Embedding Construction:**
```python
# For node pair (u, v):
pair_embedding = concat(node_u_embedding, node_v_embedding)
# Shape: (seq_len, 2 * 25) = (seq_len, 50)
```

### LSTM Model Architecture

```
Input: (batch, seq_len, 50)
  ↓
LSTM(input_size=50, hidden_size=128, num_layers=2, bidirectional=False)
  ↓
Hidden State: (128,)
  ↓
Linear(128 → 1)
  ↓
Sigmoid → Output [0, 1]
```

**Hyperparameters:**
- `input_size`: 64 (from metadata)
- `hidden_size`: 128
- `num_layers`: 2
- `dropout`: 0.2
- `bidirectional`: False

---

## 4. Sequence Detector Features

**Model File:** `src/sequence_detector.py`  
**Model Type:** LSTM or Transformer for sequence anomaly detection  
**Purpose:** Detect anomalous behavioral event sequences

### Event Sequence Features (3 features per event)

Each event in a sequence has:

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| **event_idx** | Categorical | Event type index | 0-8 |
| **amount** | Numerical | Transaction amount (log-scaled) | `log1p(amount) / 10.0` |
| **device_change** | Binary | Whether device changed | 0/1 |

### Event Types (9 total)

```python
EVENT_TYPES = [
    'login_success',      # 0
    'login_failed',       # 1
    'password_change',    # 2
    'add_payee',          # 3
    'navigate_help',      # 4
    'view_account',       # 5
    'transfer',           # 6
    'max_transfer',       # 7
    'logout'              # 8
]
```

### Sequence Structure

**Input Shape**: `(batch, max_len, 3)`
- `max_len`: 20 events (padded with zeros)
- Features: `[event_idx, log_amount, device_change]`

**Example Fraud Sequence:**
```
[login_failed, 0.0, 0],
[login_failed, 0.0, 0],
[login_success, 0.0, 1],    # Device change!
[password_change, 0.0, 1],
[add_payee, 0.0, 1],
[max_transfer, 9.9, 1],     # Large amount + device change
[logout, 0.0, 0]
```

**Example Benign Sequence:**
```
[login_success, 0.0, 0],
[view_account, 0.0, 0],
[transfer, 5.5, 0],         # Small amount
[view_account, 0.0, 0],
[logout, 0.0, 0]
```

### Model Architecture

**LSTM Detector:**
```
Input: (batch, 20, 3)
  ↓
Embedding(9 event types → 16 dim)
  ↓
LSTM(input_size=18, hidden_size=64, num_layers=2)
  ↓
Linear(64 → 1)
  ↓
Sigmoid → Anomaly Score [0, 1]
```

---

## 5. Risk Consolidator Inputs

**Model File:** `src/risk_consolidator.py`  
**Model Type:** Weighted ensemble aggregator  
**Purpose:** Combine all model outputs into final risk score

### Input Signals (5 components)

| Component | Source | Features | Weight |
|-----------|--------|----------|--------|
| **Spatial** | Graph Analysis | Fan-in, fan-out, cycles, centrality alerts | 0.20 |
| **Behavioral** | Cyber Detector | Credential stuffing, brute force, impossible travel | 0.10 |
| **Temporal** | Temporal Predictor | Volume acceleration, behavioral shift, risk escalation | 0.35 |
| **LSTM** | Link Predictor | Emerging link probabilities | 0.25 |
| **Cyber** | GBDT | Transaction-level fraud scores | 0.10 |

### Consolidation Formula

```python
final_score = Σ(weight_i × normalized_score_i) / Σ(weight_i)

# With adaptive weighting when GNN available:
if gnn_available:
    weights = {'gbdt': 0.20, 'gnn': 0.40, 'temporal': 0.20, 'lstm': 0.10, 'sequence': 0.10}
else:
    weights = {'gbdt': 0.30, 'temporal': 0.30, 'lstm': 0.20, 'sequence': 0.20}
```

### Risk Level Mapping

```python
if score >= 0.7: risk_level = 'HIGH'
elif score >= 0.4: risk_level = 'MEDIUM'
elif score > 0.0: risk_level = 'LOW'
else: risk_level = 'CLEAN'
```

---

## Feature Summary Table

| Model | Feature Count | Feature Type | Dimension |
|-------|---------------|--------------|-----------|
| **GBDT** | 11 | Tabular (transaction-level) | (N, 11) |
| **GNN** | 12 | Graph (node-level) | (N_nodes, 12) |
| **LSTM** | 25/node × 2 | Time-series (pair-level) | (N_pairs, seq_len, 50) |
| **Sequence** | 3/event | Sequential (event-level) | (N_seq, 20, 3) |
| **Consolidator** | 5 signals | Ensemble (account-level) | (N_accounts, 5) |

---

## Feature Engineering Best Practices

### 1. Normalization
- **GBDT**: No normalization (tree-based)
- **GNN**: Z-score normalization per feature
- **LSTM**: Z-score on static features, log-scale on amounts
- **Sequence**: Log-scaling for amounts, one-hot for events

### 2. Missing Value Handling
- **Numerical**: Fill with 0 or median
- **Categorical**: Add "UNKNOWN" category
- **Time-series**: Forward-fill or interpolate

### 3. Feature Interactions
- **GBDT**: Trees learn interactions automatically
- **GNN**: Message passing creates implicit interactions
- **LSTM**: Temporal interactions learned via recurrence
- **Sequence**: Event co-occurrence patterns learned

### 4. Feature Importance (GBDT Example)
```
Top 5 Features by Gain:
1. amt_log (32.5%)
2. velocity_score (18.2%)
3. ip_risk (15.1%)
4. sum_24h (12.8%)
5. device_change (8.4%)
```

---

## Model Input/Output Summary

| Model | Input | Output | Usage |
|-------|-------|--------|-------|
| **GBDT** | 11 transaction features | Fraud probability [0, 1] | Real-time transaction scoring |
| **GNN** | 12 node features + graph | Node risk score [0, 1] | Batch graph analysis |
| **LSTM** | 50-dim embeddings × seq_len | Link probability [0, 1] | Predict emerging links |
| **Sequence** | 3 features × 20 events | Anomaly score [0, 1] | Behavioral pattern detection |
| **Consolidator** | 5 component scores | Final risk [0, 1] + level | Unified risk assessment |

---

## Configuration Files

### GBDT Metadata
**File**: `models/gbdt_metadata.json`
```json
{
  "library": "lightgbm",
  "feature_count": 11,
  "feature_names": ["amt_log", "mcc_enc", ...],
  "maps": {
    "mcc_map": {...},
    "pay_map": {...}
  }
}
```

### LSTM Metadata
**File**: `models/lstm_metadata.json`
```json
{
  "input_size": 64,
  "hidden_size": 128,
  "num_layers": 2,
  "seq_len": 5
}
```

### GNN Metadata
**File**: `models/gnn_adversarial_metadata.json`
```json
{
  "num_rounds": 2,
  "gnn_epochs": 20,
  "in_features": 12
}
```

### Risk Consolidator Config
**File**: `models/consolidation_config.json`
```json
{
  "weights": {
    "spatial": 0.20,
    "behavioral": 0.10,
    "temporal": 0.35,
    "lstm": 0.25,
    "cyber": 0.10
  }
}
```

---

## Feature Evolution & Future Enhancements

### Planned Feature Additions

1. **GBDT**: 
   - Time-of-day features (hour, day-of-week)
   - Merchant reputation scores
   - Historical fraud rate per MCC

2. **GNN**: 
   - Community detection features
   - Temporal edge weights
   - Multi-hop neighborhood aggregation

3. **LSTM**: 
   - Attention mechanism weights
   - Multi-scale temporal features (hourly, daily, weekly)
   - External risk signals (sanctions lists, PEP)

4. **Sequence**: 
   - Session duration features
   - Geo-location consistency
   - User agent fingerprinting

---

## References

- GBDT Implementation: `src/gbdt_detector.py`
- GNN Implementation: `src/gnn_trainer.py`
- LSTM Implementation: `src/lstm_link_predictor.py`
- Sequence Implementation: `src/sequence_detector.py`
- Feature Engineering: `src/embedding_builder.py`
- Risk Consolidation: `src/risk_consolidator.py`

---

**Last Updated**: January 12, 2026  
**Version**: 1.0  
**Total Features**: 51 unique features across 4 models
