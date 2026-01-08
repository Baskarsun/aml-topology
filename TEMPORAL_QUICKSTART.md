# Quick Start: Temporal Prediction Subsystem

## What's New

You now have a **predictive subsystem** that forecasts suspicious behavior before it manifests in observable patterns. While your original system detects *what is happening*, the temporal system predicts *what will happen*.

## Running the Enhanced System

Simply run your normal pipeline:
```bash
python main.py
```

The temporal analysis runs as Phase 3, after spatial detection (Phase 2) and before visualization (Phase 4).

## Understanding the Alerts

### New Alert Types in Output

**1. Volume Acceleration**
```
PREDICTIVE ALERT: Account ACC_0015 showing 45.2% volume acceleration
```
- Account transaction volume growing rapidly
- **Why it matters**: Precursor to structuring (breaking large amounts into smaller chunks)
- **Action**: Monitor for pattern of just-below-threshold transactions

**2. Behavioral Shift**
```
PREDICTIVE ALERT: Account ACC_0022 shows significant behavioral shift
```
- Account's transaction patterns deviating from normal
- **Why it matters**: Could indicate account compromise or suspicious use
- **Action**: Verify legitimacy; check for unauthorized access

**3. Risk Escalation**
```
PREDICTIVE ALERT: Account ACC_0010 has 72.5% predicted risk of escalation
Risk signals: structuring_precursor, rapid_small_transfers, counterparty_network_expansion
```
- Multiple risk signals combined suggesting escalation to organized scheme
- **Why it matters**: Multi-signal approach catches complex patterns early
- **Action**: Enhanced monitoring; consider customer contact

**4. Temporal Concentration**
```
PREDICTIVE ALERT: Account ACC_0025 shows 65% temporal concentration
```
- Transactions clustered in short time window (coordinated activity)
- **Why it matters**: Suggests orchestrated rather than organic behavior
- **Action**: Investigate timing coordination with other flagged accounts

**5. Cycle Emergence**
```
PREDICTIVE ALERT: Account ACC_0030 likely to form cycles (78% probability)
```
- Account positioned in bidirectional flows (classic cycle setup)
- **Why it matters**: Precursor to round-tripping/layering schemes
- **Action**: Monitor for completion of cyclical flows

**6. Structuring Sequence**
```
PREDICTIVE ALERT: Account ACC_0012 shows structuring sequence (7 txs)
```
- Multiple transactions just below regulatory thresholds in short window
- **Why it matters**: Clear structuring pattern
- **Action**: Consider CTR/SAR filing; freeze account if necessary

## Risk Score Interpretation

All alerts include a risk score (0-100):

| Score | Meaning | Recommended Action |
|-------|---------|-------------------|
| 30-40 | Low to Medium | Monitor, log |
| 40-60 | Medium | Enhanced monitoring, verify customer |
| 60-75 | High | Escalate, begin investigation |
| 75-100 | Critical | Immediate action, consider freeze |

## Account Risk Summary

At the end of Phase 3, you'll see:
```
Temporal Forecast Summary:
  Total flagged accounts: 8
  Risk escalation predictions: 3
  Temporal concentration alerts: 2

  Top 5 highest temporal risk accounts:
    - ACC_0010: Risk Score 78.5 (3 signals)
    - ACC_0025: Risk Score 65.2 (2 signals)
    - ...
```

## Combining with Spatial Detections

Accounts flagged by **both** spatial AND temporal systems are highest priority:

```
Example suspicious_set (end of execution):
- ACC_0010: Fan-in + Risk Escalation + Structuring Sequence
- ACC_0025: Fan-out + Temporal Concentration + Cycle Emergence
- ACC_0030: Cycle detected + Cycle Emergence Prediction
```

**These accounts warrant immediate investigation.**

## Customizing Detection Sensitivity

Edit the temporal detection calls in `main.py` to adjust sensitivity:

```python
# More sensitive (catch more, more false positives)
risk_predictions = temporal_pred.forecast_risk_escalation(df, early_warning_threshold=0.4)

# Less sensitive (catch fewer, fewer false positives)
risk_predictions = temporal_pred.forecast_risk_escalation(df, early_warning_threshold=0.8)

# Adjust other detections similarly:
volume_alerts = temporal_pred.detect_volume_acceleration(df, threshold_sigma=2.0)  # More sensitive
behavior_alerts = temporal_pred.detect_behavioral_shift(df, deviation_threshold=1.5)  # More sensitive
```

## Output Files

The system now generates:
- **transactions.csv**: Transaction data (unchanged)
- **aml_network_graph.png**: Network visualization with spatial + temporal flags
- Console output: Detailed alerts from both spatial and temporal systems

## Example Workflow

1. **Run the system**: `python main.py`
2. **Review spatial alerts**: Fan-in, fan-out, cycles (what's happening now)
3. **Review temporal alerts**: Volume acceleration, behavioral shifts, risk escalation (what might happen next)
4. **Identify overlap**: Accounts in multiple alert categories = highest priority
5. **Take action**: Investigate, verify, monitor, or freeze based on risk level

## Common Patterns to Watch For

### Structuring Indicators
- ✓ Volume acceleration (30%+)
- ✓ Structuring sequence (3+ txs just below threshold)
- ✓ Behavioral shift towards small transactions
- ✓ Risk escalation with "structuring_precursor" signal

### Cycling Indicators
- ✓ Cycle emergence prediction (bidirectional relationships)
- ✓ Temporal concentration of bidirectional transactions
- ✓ Cycle detection from spatial system

### Complex Schemes
- ✓ Multiple risk signals (3+)
- ✓ Network expansion (new counterparties)
- ✓ Combined with fan-in/fan-out patterns

## API Reference

```python
from src.temporal_predictor import TemporalPredictor, SequenceAnalyzer

# Main predictor
temporal = TemporalPredictor(lookback_days=30, forecast_days=7)

# Methods
temporal.establish_baselines(df)           # Setup
temporal.detect_volume_acceleration(df)    # Detection 1
temporal.detect_behavioral_shift(df)       # Detection 2
temporal.forecast_risk_escalation(df)      # Detection 3 (multi-signal)
temporal.detect_temporal_concentration(df) # Detection 4
temporal.predict_cycle_emergence(df)       # Detection 5
temporal.forecast_account_summary(df)      # Comprehensive report

# Sequence analyzer
seq = SequenceAnalyzer()
seq.detect_structuring_sequence(df)        # Structuring detection
```

## Troubleshooting

**Issue**: No temporal alerts generated
- **Check**: Does your transaction data have timestamps?
- **Check**: Are there 3+ transactions per account for baselines?
- **Solution**: Adjust thresholds lower (lower early_warning_threshold, etc.)

**Issue**: Too many false positives
- **Check**: Thresholds might be too sensitive
- **Solution**: Increase threshold_sigma, early_warning_threshold, deviation_threshold

**Issue**: "No module named temporal_predictor"
- **Check**: File is at `src/temporal_predictor.py`?
- **Solution**: Ensure file exists and imports are correct in main.py

## Next Steps

1. **Integrate with rules engine**: Feed temporal alerts into your existing rule system
2. **Add persistence**: Store temporal baselines and predictions in database
3. **Enable learning**: Track which predictions led to confirmed AML cases
4. **Combine with ML**: Use temporal features for gradient boosting (gbdt_detector.py)
5. **Multi-channel analysis**: Extend temporal analysis by channel or geography

---

For detailed documentation, see `TEMPORAL_SUBSYSTEM.md`
