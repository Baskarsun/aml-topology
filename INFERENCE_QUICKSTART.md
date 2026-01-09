# Inference API Quick Start

**Status**: ✅ Ready to Use  
**Setup Time**: < 5 minutes

## Step 1: Install Dependencies

```bash
pip install flask requests
```

## Step 2: Train Models (if not done)

```bash
python main.py
```

This creates all model files in `models/` directory.

## Step 3: Start API Server

```bash
python -m src.inference_api
```

You should see:
```
Starting AML Inference API Server...
Loading inference models...
✓ GBDT model loaded
✓ Sequence Detector model loaded
✓ LSTM Link Predictor model loaded
✓ Risk Consolidator loaded

API Server running on http://localhost:5000

Available endpoints:
  GET  /health                      - Health check
  POST /score/transaction           - Score single transaction
  POST /score/sequence              - Score event sequence
  POST /score/consolidate           - Consolidate all signals
  POST /batch/score                 - Batch scoring
  GET  /models/info                 - Model information

Press Ctrl+C to stop.
```

## Step 4: Test API

### Via curl

**Health Check**:
```bash
curl http://localhost:5000/health
```

**Score a Transaction**:
```bash
curl -X POST http://localhost:5000/score/transaction \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5000.0,
    "mcc": "5411",
    "payment_type": "card",
    "device_change": 0,
    "ip_risk": 0.1,
    "count_1h": 3,
    "sum_24h": 15000.0,
    "uniq_payees_24h": 2,
    "country": "US"
  }'
```

Expected response:
```json
{
  "transaction": {
    "amount": 5000.0,
    ...
  },
  "gbdt_score": 0.45,
  "gbdt_risk_level": "MEDIUM",
  "model": "gbdt",
  "timestamp": "2026-01-09T14:30:45.123456"
}
```

**Score an Event Sequence**:
```bash
curl -X POST http://localhost:5000/score/sequence \
  -H "Content-Type: application/json" \
  -d '{
    "events": ["login_success", "view_account", "transfer", "logout"]
  }'
```

**Consolidate All Signals**:
```bash
curl -X POST http://localhost:5000/score/consolidate \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "ACC_0025",
    "transaction": {
      "amount": 5000.0,
      "mcc": "5411",
      "payment_type": "card",
      "device_change": 0,
      "ip_risk": 0.1,
      "count_1h": 3,
      "sum_24h": 15000.0,
      "uniq_payees_24h": 2,
      "country": "US"
    },
    "events": ["login_success", "transfer", "logout"]
  }'
```

### Via Python Client

**Create `test_api.py`**:
```python
from src.inference_client import InferenceClient
import json

# Connect to API
client = InferenceClient('http://localhost:5000')

# Test 1: Health check
print("=== HEALTH CHECK ===")
health = client.health_check()
print(f"Status: {health['status']}")
print(f"Models loaded: {health['models_loaded']}\n")

# Test 2: Score transaction
print("=== SCORE TRANSACTION ===")
transaction = {
    'amount': 5000.0,
    'mcc': '5411',
    'payment_type': 'card',
    'device_change': 0,
    'ip_risk': 0.1,
    'count_1h': 3,
    'sum_24h': 15000.0,
    'uniq_payees_24h': 2,
    'country': 'US'
}
result = client.score_transaction(transaction)
print(f"GBDT Score: {result['gbdt_score']:.3f}")
print(f"Risk Level: {result['gbdt_risk_level']}\n")

# Test 3: Score sequence
print("=== SCORE EVENT SEQUENCE ===")
events = ['login_success', 'view_account', 'transfer', 'logout']
result = client.score_sequence(events)
print(f"Sequence Score: {result['sequence_score']:.3f}")
print(f"Anomaly Risk: {result['anomaly_risk_level']}\n")

# Test 4: Consolidate
print("=== CONSOLIDATE RISKS ===")
result = client.consolidate_risks(
    account_id='ACC_0025',
    transaction=transaction,
    events=events
)
print(f"Consolidated Score: {result['consolidated_risk_score']:.3f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}\n")

# Test 5: Batch scoring
print("=== BATCH SCORING ===")
batch = [
    {
        'account_id': 'ACC_0001',
        'transaction': {
            'amount': 1000.0,
            'mcc': '5411',
            'payment_type': 'card',
            'device_change': 0,
            'ip_risk': 0.05,
            'count_1h': 1,
            'sum_24h': 5000.0,
            'uniq_payees_24h': 1,
            'country': 'US'
        },
        'events': ['login_success', 'logout']
    },
    {
        'account_id': 'ACC_0002',
        'transaction': {
            'amount': 50000.0,
            'mcc': '5411',
            'payment_type': 'wire',
            'device_change': 1,
            'ip_risk': 0.8,
            'count_1h': 10,
            'sum_24h': 150000.0,
            'uniq_payees_24h': 5,
            'country': 'NG'
        },
        'events': ['login_failed', 'login_failed', 'login_success', 'max_transfer']
    }
]
result = client.batch_score(batch)
print(f"Total Accounts: {result['total']}")
print(f"High Risk: {result['summary']['high_risk']}")
print(f"Medium Risk: {result['summary']['medium_risk']}")
print(f"Low Risk: {result['summary']['low_risk']}")
print(f"Clean: {result['summary']['clean']}\n")

for account_result in result['results']:
    print(f"Account: {account_result['account_id']}")
    print(f"  Risk Score: {account_result['consolidated_risk_score']:.3f}")
    print(f"  Risk Level: {account_result['risk_level']}")
    print(f"  Recommendation: {account_result['recommendation']}\n")

# Test 6: Get models info
print("=== MODEL INFORMATION ===")
info = client.get_models_info()
print(f"Available Models: {info['available_models']}")
print(f"Consolidator Weights: {info['consolidator_weights']}\n")

client.close()
print("All tests completed!")
```

**Run tests**:
```bash
python test_api.py
```

## Common Use Cases

### Use Case 1: Real-Time Transaction Scoring

```python
from src.inference_client import InferenceClient

client = InferenceClient('http://localhost:5000')

# Score incoming transaction
transaction = {
    'amount': 5000.0,
    'mcc': '5411',
    'payment_type': 'card',
    'device_change': 0,
    'ip_risk': 0.1,
    'count_1h': 3,
    'sum_24h': 15000.0,
    'uniq_payees_24h': 2,
    'country': 'US'
}

result = client.consolidate_risks(
    account_id='ACCT_12345',
    transaction=transaction
)

# Make decision
if result['risk_level'] == 'HIGH':
    print(f"BLOCK: {result['recommendation']}")
elif result['risk_level'] == 'MEDIUM':
    print(f"REVIEW: {result['recommendation']}")
else:
    print(f"ALLOW: {result['recommendation']}")
```

### Use Case 2: Batch Daily Monitoring

```python
from src.inference_client import InferenceClient
import pandas as pd

client = InferenceClient('http://localhost:5000')

# Load transactions from file
df = pd.read_csv('daily_transactions.csv')

# Prepare batch request
batch = []
for _, row in df.iterrows():
    batch.append({
        'account_id': row['account_id'],
        'transaction': {
            'amount': row['amount'],
            'mcc': row['mcc'],
            'payment_type': row['payment_type'],
            'device_change': row['device_change'],
            'ip_risk': row['ip_risk'],
            'count_1h': row['count_1h'],
            'sum_24h': row['sum_24h'],
            'uniq_payees_24h': row['uniq_payees_24h'],
            'country': row['country']
        }
    })

# Score all accounts
result = client.batch_score(batch)

# Generate report
print(f"\nDaily Monitoring Report")
print(f"Total Accounts: {result['total']}")
print(f"High Risk: {result['summary']['high_risk']}")
print(f"Medium Risk: {result['summary']['medium_risk']}")

# Alert on high risk
for account in result['results']:
    if account['risk_level'] == 'HIGH':
        print(f"⚠️  ALERT: {account['account_id']} - {account['recommendation']}")
```

### Use Case 3: Event Sequence Monitoring

```python
from src.inference_client import InferenceClient

client = InferenceClient('http://localhost:5000')

# Monitor user session
user_events = []

# As user performs actions, collect events
events = [
    'login_success',
    'view_account',
    'add_payee',
    'transfer',          # Suspicious: just added payee and immediate transfer
    'max_transfer',      # Very suspicious: max amount
]

# Score the sequence
result = client.score_sequence(events)

if result['anomaly_risk_level'] == 'HIGH':
    print("⚠️  SUSPICIOUS SEQUENCE DETECTED")
    print(f"Events: {' → '.join(events)}")
    print(f"Risk Score: {result['sequence_score']:.3f}")
```

## API Response Codes Reference

| Code | Description | Example |
|------|---|---|
| `HIGH` | Risk ≥ 0.7 | Fraud likely - block immediately |
| `MEDIUM` | Risk 0.4-0.7 | Suspicious - require verification |
| `LOW` | Risk 0.0-0.4 | Minor risk - monitor |
| `CLEAN` | Risk = 0.0 | No risk detected |

## Troubleshooting

### Problem: "Connection refused"
```
Error: Failed to connect to http://localhost:5000
```
**Solution**: Start API server with `python -m src.inference_api`

### Problem: "Models not loaded"
```
Error: GBDT model not loaded
```
**Solution**: Train models first with `python main.py`

### Problem: "Invalid JSON"
```
Error: No JSON data provided
```
**Solution**: Ensure request has `Content-Type: application/json` header

### Problem: "Missing fields"
```
Error: Missing required field
```
**Solution**: Check all required fields are provided in request

## Next Steps

1. **Integrate with your system**: Use Python client or REST API
2. **Monitor performance**: Track response times and accuracy
3. **Tune thresholds**: Adjust risk levels based on business needs
4. **Deploy to production**: Use Docker or cloud platform

---

**Ready to score transactions!** 🚀
