# Temporal Predictor API Reference Card

## Quick Reference for Developers

### Import Statement
```python
from src.temporal_predictor import TemporalPredictor, SequenceAnalyzer
```

---

## TemporalPredictor Class

### Initialization
```python
temporal = TemporalPredictor(
    lookback_days=30,      # Historical window for baseline (default: 30)
    forecast_days=7        # Prediction horizon (default: 7)
)
```

### Method 1: Establish Baselines
```python
baselines = temporal.establish_baselines(df)

# Parameters:
#   df: DataFrame with columns [source, target, amount, timestamp]

# Returns:
#   {account_id: {
#       'avg_out_amount': float,
#       'median_out_amount': float,
#       'std_out_amount': float,
#       'avg_out_frequency': float,
#       'avg_in_amount': float,
#       'unique_out_counterparties': int,
#       'unique_in_counterparties': int,
#       'total_out_volume': float,
#       'total_in_volume': float
#   }, ...}
```

### Method 2: Volume Acceleration
```python
alerts = temporal.detect_volume_acceleration(
    df,
    threshold_sigma=2.5    # Sensitivity (default: 2.5)
)

# Returns: List of {
#   'account': str,
#   'type': 'volume_acceleration',
#   'current_daily_volume': float,
#   'baseline_daily_volume': float,
#   'acceleration_rate': float,  # e.g., 0.452 for 45.2%
#   'score': float (0-100),
#   'reason': str
# }
```

### Method 3: Behavioral Shift
```python
alerts = temporal.detect_behavioral_shift(
    df,
    deviation_threshold=2.0  # Z-score threshold (default: 2.0)
)

# Returns: List of {
#   'account': str,
#   'type': 'behavioral_shift',
#   'metric_changes': {
#       'avg_amount_zscore': float,
#       'frequency_change_pct': float,
#       'counterparty_diversity_change_pct': float
#   },
#   'current_metrics': {
#       'avg_transaction': float,
#       'recent_tx_count': int,
#       'unique_targets': int
#   },
#   'baseline_metrics': {...},
#   'score': float (0-100),
#   'reason': str
# }
```

### Method 4: Risk Escalation (Multi-Signal)
```python
predictions = temporal.forecast_risk_escalation(
    df,
    early_warning_threshold=0.6  # Probability threshold (default: 0.6)
)

# Returns: List of {
#   'account': str,
#   'type': 'risk_escalation',
#   'predicted_risk_probability': float (0-1),  # e.g., 0.725
#   'risk_signals': [str],  # e.g., ['structuring_precursor', 'rapid_small_transfers']
#   'individual_signal_scores': {signal_name: float, ...},
#   'score': float (0-100),
#   'forecast_horizon_days': int,
#   'reason': str
# }
```

**Risk Signals:**
- `structuring_precursor`: Transactions $5k-$10k (just below CTR threshold)
- `rapid_small_transfers`: Pattern of small, quick consecutive txs
- `counterparty_network_expansion`: Rapidly increasing unique recipients
- `timing_clustering`: Synchronized transaction timing

### Method 5: Temporal Concentration
```python
alerts = temporal.detect_temporal_concentration(
    df,
    min_transactions=4,      # Minimum in window (default: 4)
    time_window_hours=24     # Window size (default: 24)
)

# Returns: List of {
#   'account': str,
#   'type': 'temporal_concentration',
#   'burst_transactions': int,
#   'window_hours': int,
#   'concentration_pct': float,  # e.g., 65.5 for 65.5%
#   'total_transactions': int,
#   'burst_window': {
#       'start': str (timestamp),
#       'end': str (timestamp)
#   },
#   'score': float (0-100),
#   'reason': str
# }
```

### Method 6: Cycle Emergence
```python
predictions = temporal.predict_cycle_emergence(
    df,
    min_chain_length=3  # Minimum for prediction (default: 3)
)

# Returns: List of {
#   'account': str,
#   'type': 'cycle_emergence',
#   'predicted_cycle_probability': float (0-1),
#   'bidirectional_counterparties': int,
#   'incoming_connections': int,
#   'outgoing_connections': int,
#   'cycle_risk_indicators': {
#       'reciprocal_relationship_count': int,
#       'transaction_balance': float  # Out - In
#   },
#   'score': float (0-100),
#   'reason': str
# }
```

### Method 7: Comprehensive Summary
```python
report = temporal.forecast_account_summary(
    df,
    include_all_detections=True  # Include full detection lists
)

# Returns: {
#   'forecast_type': str,  # 'temporal_predictive'
#   'lookback_days': int,
#   'forecast_days': int,
#   'baseline_statistics': {
#       'total_accounts_baselined': int,
#       'mean_baseline_transaction': float
#   },
#   'detection_summary': {
#       'volume_acceleration_alerts': int,
#       'behavioral_shift_alerts': int,
#       'risk_escalation_predictions': int,
#       'temporal_concentration_alerts': int,
#       'cycle_emergence_predictions': int,
#       'total_flagged_accounts': int
#   },
#   'highest_risk_accounts': [
#       {
#           'account': str,
#           'temporal_risk_score': float,
#           'signal_count': int,
#           'signals': [str]
#       }, ...
#   ],
#   'all_predictions': {
#       'volume_acceleration': [...],
#       'behavioral_shift': [...],
#       'risk_escalation': [...],
#       'temporal_concentration': [...],
#       'cycle_emergence': [...]
#   }
# }
```

---

## SequenceAnalyzer Class

### Initialization
```python
analyzer = SequenceAnalyzer()
```

### Method: Structuring Sequence
```python
alerts = analyzer.detect_structuring_sequence(
    df,
    threshold_amount=10000,        # Regulatory threshold (default: 10000)
    just_below_threshold=9000,     # Range floor (default: 9000)
    time_window_days=7             # Detection window (default: 7)
)

# Returns: List of {
#   'account': str,
#   'type': 'structuring_sequence',
#   'transaction_count': int,
#   'time_window_days': int,
#   'total_structured_amount': float,
#   'avg_transaction_amount': float,
#   'amount_range': str,  # e.g., "9000-10000"
#   'score': float (0-100),
#   'reason': str
# }
```

---

## Common Usage Patterns

### Full Temporal Analysis Pipeline
```python
# 1. Initialize
temporal_pred = TemporalPredictor(lookback_days=30, forecast_days=7)
seq_analyzer = SequenceAnalyzer()

# 2. Establish baselines
baselines = temporal_pred.establish_baselines(df)

# 3. Run all detections
volume = temporal_pred.detect_volume_acceleration(df)
behavior = temporal_pred.detect_behavioral_shift(df)
risks = temporal_pred.forecast_risk_escalation(df)
bursts = temporal_pred.detect_temporal_concentration(df)
cycles = temporal_pred.predict_cycle_emergence(df)
structuring = seq_analyzer.detect_structuring_sequence(df)

# 4. Generate comprehensive report
report = temporal_pred.forecast_account_summary(df)

# 5. Process results
for alert in volume + behavior + risks + bursts + cycles:
    if alert['score'] > 60:
        print(f"HIGH RISK: {alert['account']}")
```

### Sensitivity Adjustment for Different Risk Profiles
```python
# CONSERVATIVE (Lower FP rate, higher FN rate)
temporal_pred.detect_volume_acceleration(df, threshold_sigma=3.0)
temporal_pred.detect_behavioral_shift(df, deviation_threshold=3.0)
temporal_pred.forecast_risk_escalation(df, early_warning_threshold=0.8)

# BALANCED (Recommended)
temporal_pred.detect_volume_acceleration(df, threshold_sigma=2.5)
temporal_pred.detect_behavioral_shift(df, deviation_threshold=2.0)
temporal_pred.forecast_risk_escalation(df, early_warning_threshold=0.6)

# AGGRESSIVE (Higher FP rate, lower FN rate)
temporal_pred.detect_volume_acceleration(df, threshold_sigma=2.0)
temporal_pred.detect_behavioral_shift(df, deviation_threshold=1.5)
temporal_pred.forecast_risk_escalation(df, early_warning_threshold=0.4)
```

### Filtering High-Risk Accounts
```python
# Get top 5 highest risk accounts
report = temporal_pred.forecast_account_summary(df)
top_accounts = report['highest_risk_accounts'][:5]

# Get only critical alerts (score > 75)
critical_alerts = [
    alert for alert in volume + behavior + risks + bursts + cycles
    if alert['score'] > 75
]

# Get accounts with 3+ risk signals
multi_signal = [
    acc for acc in report['highest_risk_accounts']
    if acc['signal_count'] >= 3
]
```

---

## Data Format Requirements

### Input DataFrame Columns
```python
df = pd.DataFrame({
    'source': ['ACC_001', 'ACC_002', ...],          # Required
    'target': ['ACC_003', 'ACC_004', ...],          # Required
    'amount': [5000.0, 9500.0, ...],                # Required
    'timestamp': [1609459200, 1609545600, ...],     # Required (Unix seconds)
    'channel': ['bank_transfer', 'wire', ...],      # Optional
})
```

### Timestamp Format
- **Unix seconds**: `1609459200` (automatically converted to datetime)
- **Pandas datetime**: `pd.Timestamp('2021-01-01')`
- **ISO string**: `'2021-01-01T00:00:00'` (will be converted)

---

## Alert Score Interpretation

```
Score Range    | Severity    | Action
0-30          | Informational| Monitor, log
30-60         | Medium       | Review, verify customer
60-80         | High         | Escalate, investigate
80-100        | Critical     | Immediate action, consider freeze
```

---

## Performance Characteristics

| Operation | Complexity | ~100k txs | ~1M txs |
|---|---|---|---|
| `establish_baselines()` | O(n) | <100ms | <1s |
| `detect_volume_acceleration()` | O(n log n) | ~200ms | ~2s |
| `detect_behavioral_shift()` | O(n) | ~150ms | ~1.5s |
| `forecast_risk_escalation()` | O(n) | ~300ms | ~3s |
| `detect_temporal_concentration()` | O(n²) worst | ~500ms | ~5s |
| `predict_cycle_emergence()` | O(n + edges) | ~100ms | ~1s |
| `forecast_account_summary()` | All above | ~2s total | ~15s total |

---

## Error Handling

```python
# Handle missing or empty data
if df is None or len(df) == 0:
    print("No transaction data provided")
    return

# Handle timestamp conversion
try:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
except Exception as e:
    print(f"Timestamp conversion error: {e}")

# Handle accounts with insufficient data
if len(baselines) < 5:
    print("Warning: Few accounts with sufficient history for baselining")
```

---

## Tips & Tricks

1. **Run `establish_baselines()` first** - Ensures all detections use consistent baselines

2. **Adjust thresholds gradually** - Start with defaults, then tune based on your data

3. **Use `forecast_account_summary()`** - Easier than running individual methods

4. **Filter by score** - Focus investigation on accounts with score > 60

5. **Look for signal overlap** - Accounts with 3+ different signals are highest priority

6. **Compare with spatial alerts** - Accounts in both systems deserve immediate attention

7. **Track prediction accuracy** - Log which predictions led to confirmed AML cases

---

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| No alerts | Data too small | Ensure 3+ txs per account |
| Too many alerts | Low threshold | Increase threshold_sigma, etc. |
| Memory error | Large dataset | Process in batches |
| NaN in results | Missing columns | Check timestamp format |
| Inconsistent scores | Baseline drift | Re-run `establish_baselines()` |

---

**Version**: 1.0  
**Last Updated**: 2026-01-07
