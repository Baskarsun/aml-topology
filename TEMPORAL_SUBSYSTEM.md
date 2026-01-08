# Temporal & Predictive AML Subsystem

## Overview

The temporal prediction subsystem complements the existing spatial/detective AML system with **forward-looking, predictive capabilities**. While the original system detects active suspicious patterns (fan-in, fan-out, cycles), the temporal subsystem forecasts future anomalies and behavioral changes.

### System Architecture

```
AML TOPOLOGY SYSTEM
├── SPATIAL/DETECTIVE (Original)
│   ├── Fan-In Detection (Structuring)
│   ├── Fan-Out Detection (Dissipation)
│   ├── Cycle Detection (Round-tripping)
│   └── Centrality Analysis (Bridge Nodes)
│
└── TEMPORAL/PREDICTIVE (New)
    ├── Volume Acceleration Analysis
    ├── Behavioral Shift Detection
    ├── Risk Escalation Forecasting
    ├── Temporal Concentration Analysis
    ├── Cycle Emergence Prediction
    └── Structuring Sequence Analysis
```

## Core Components

### 1. TemporalPredictor Class

The main predictive engine that establishes baselines and forecasts account behavior changes.

#### Key Methods:

**`establish_baselines(df)`**
- Establishes statistical baselines for all accounts
- Computes: average transaction amounts, frequency, counterparty diversity
- Output: Dictionary of baseline metrics per account

**`detect_volume_acceleration(df, threshold_sigma=2.5)`**
- **Purpose**: Detect accelerating transaction volumes (precursor to structuring)
- **Logic**: Analyzes daily transaction volume trends; flags accounts with >30% daily acceleration
- **Use Case**: Early warning for accounts preparing to conduct structuring activities
- **Output**: List of alerts with acceleration rates and risk scores

**`detect_behavioral_shift(df, deviation_threshold=2.0)`**
- **Purpose**: Identify statistically significant deviations from normal behavior
- **Metrics Analyzed**:
  - Transaction amount deviation (Z-score)
  - Frequency changes (transaction count)
  - Counterparty network expansion
- **Use Case**: Accounts exhibiting unusual changes in transaction patterns
- **Output**: List of behavioral anomalies with metric changes

**`forecast_risk_escalation(df, early_warning_threshold=0.6)`**
- **Purpose**: Multi-signal risk prediction for future suspicious behavior
- **Risk Signals Detected**:
  1. **Structuring Precursor**: Just-below-threshold transactions ($5k-$10k)
  2. **Rapid Small Transfers**: Pattern of small, quick transactions
  3. **Counterparty Network Expansion**: Rapidly increasing unique targets
  4. **Timing Clustering**: Orchestrated transaction timing
- **Logic**: Bayesian aggregation of individual signal probabilities
- **Output**: Risk escalation predictions with component signals

**`detect_temporal_concentration(df, min_transactions=4, time_window_hours=24)`**
- **Purpose**: Identify suspicious time clustering of transactions
- **Logic**: Detects when >40% of account activity occurs in a single time window
- **Use Case**: Identifies coordinated or orchestrated activities
- **Output**: Burst alerts with concentration metrics

**`predict_cycle_emergence(df, min_chain_length=3)`**
- **Purpose**: Forecast accounts likely to form circular money flows
- **Logic**: Identifies bidirectional transaction relationships
- **Output**: Cycle emergence predictions with relationship counts

**`forecast_account_summary(df)`**
- **Purpose**: Generate comprehensive temporal risk profile for all accounts
- **Output**: Ranked list of highest-risk accounts and detection summary

### 2. SequenceAnalyzer Class

Analyzes multi-step transaction sequences to predict organized schemes.

**`detect_structuring_sequence(df, threshold_amount=10000, just_below_threshold=9000, time_window_days=7)`**
- **Purpose**: Identify structuring patterns (breaking large amounts into smaller ones)
- **Logic**: Finds clusters of transactions just below regulatory thresholds within short time windows
- **Regulatory Context**: $10,000 CTR threshold in US; €10,000 in EU
- **Output**: Structuring sequence alerts with transaction details

## Key Features

### Multi-Signal Risk Aggregation
The system combines multiple weak signals into strong predictions using Bayesian probability:
```
Aggregate Risk = 1 - ∏(1 - signal_score) for all signals
```

### Baseline-Relative Analysis
All detections are relative to established baselines, reducing false positives for naturally active accounts.

### Temporal Window Flexibility
Configurable lookback periods and forecast horizons adapt to your business needs.

## Integration with Spatial System

The temporal system enhances the spatial detection through:

1. **Risk Prioritization**: Accounts flagged by both systems get higher priority
2. **Early Warning**: Temporal alerts often precede spatial pattern detection
3. **Context**: Behavioral shifts explain why spatial patterns emerge
4. **Pattern Prediction**: Forecast which accounts will form cycles or fan patterns

## Usage Example

```python
from src.temporal_predictor import TemporalPredictor, SequenceAnalyzer

# Initialize predictors
temporal_pred = TemporalPredictor(lookback_days=30, forecast_days=7)
seq_analyzer = SequenceAnalyzer()

# Establish baselines from historical data
baselines = temporal_pred.establish_baselines(df)

# Run predictive detections
volume_alerts = temporal_pred.detect_volume_acceleration(df)
behavior_alerts = temporal_pred.detect_behavioral_shift(df)
risk_predictions = temporal_pred.forecast_risk_escalation(df)
cycle_predictions = temporal_pred.predict_cycle_emergence(df)
structuring_seqs = seq_analyzer.detect_structuring_sequence(df)

# Generate comprehensive forecast
forecast = temporal_pred.forecast_account_summary(df)
```

## Output Structure

All detection methods return lists of dictionaries with structure:
```python
{
    'account': str,           # Account identifier
    'type': str,              # Detection type
    'score': float (0-100),   # Risk score
    'reason': str,            # Human-readable explanation
    # Additional type-specific fields...
}
```

## Risk Scoring

Risk scores (0-100) are calibrated for action thresholds:
- **0-30**: Low risk, informational
- **30-60**: Medium risk, monitor closely
- **60-80**: High risk, escalate investigation
- **80-100**: Critical risk, immediate action

## Tuning Parameters

Key parameters for different risk profiles:

| Parameter | Low Risk | Medium Risk | High Risk |
|-----------|----------|-------------|-----------|
| `threshold_sigma` | 3.0 | 2.5 | 2.0 |
| `deviation_threshold` | 3.0 | 2.0 | 1.5 |
| `early_warning_threshold` | 0.8 | 0.6 | 0.4 |
| `concentration_pct` | 60% | 40% | 20% |

## Performance Considerations

- **Baseline Establishment**: O(n) for n transactions
- **Single Detection**: O(n log n) due to sorting/grouping
- **Full Forecast**: O(n * m) where m is number of detection types (typically ~6)

For 100k transactions: ~2-3 seconds total runtime on modern hardware

## Future Enhancements

1. **Machine Learning Integration**: Train RNN/LSTM models on temporal sequences
2. **Network Effects**: Consider correlated behavior of connected accounts
3. **Geographic/Channel Patterns**: Temporal analysis by location or payment method
4. **Ensemble Methods**: Combine temporal predictions with spatial detections via ensemble
5. **Feedback Loops**: Learn from confirmed cases to improve signal weights

## Integration Points with Existing System

- **Main.py**: Temporal module integrated as Phase 3 (between Phase 2 spatial analysis and Phase 4 visualization)
- **Visualization**: Suspicious accounts flagged by both systems highlighted with different markers
- **Reporting**: Temporal alerts can feed into rule engine for automated decisions

---

**Author**: AML Topology Enhancement  
**Version**: 1.0  
**Last Updated**: 2026-01-07
