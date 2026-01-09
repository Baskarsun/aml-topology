# Inference API Documentation

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Last Updated**: January 9, 2026

## Overview

The AML Inference API provides REST endpoints to score transactions, event sequences, and accounts using trained ML models. All models are loaded at startup for optimal performance.

## Quick Start

### 1. Install Dependencies
```bash
pip install flask requests
```

### 2. Start API Server
```bash
python -m src.inference_api
```

Server starts at: `http://localhost:5000`

### 3. Test Endpoint
```bash
curl http://localhost:5000/health
```

## API Endpoints

### Health Check
Check if API is running and models are loaded.

**Request**:
```
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-09T14:30:45.123456",
  "models_loaded": {
    "gbdt": true,
    "sequence": true,
    "lstm": true,
    "gnn": false,
    "consolidator": true
  },
  "metadata": {
    "gbdt": {
      "library": "lightgbm",
      "timestamp": "2026-01-09T10:00:00"
    },
    ...
  }
}
```

---

### Score Single Transaction (GBDT)
Score a single transaction using LightGBM model.

**Request**:
```
POST /score/transaction
Content-Type: application/json

{
  "amount": 5000.0,
  "mcc": "5411",
  "payment_type": "card",
  "device_change": 0,
  "ip_risk": 0.1,
  "count_1h": 3,
  "sum_24h": 15000.0,
  "uniq_payees_24h": 2,
  "country": "US"
}
```

**Response**:
```json
{
  "transaction": {
    "amount": 5000.0,
    "mcc": "5411",
    ...
  },
  "gbdt_score": 0.45,
  "gbdt_risk_level": "MEDIUM",
  "model": "gbdt",
  "timestamp": "2026-01-09T14:30:45.123456"
}
```

**Risk Levels**:
- `HIGH`: score ≥ 0.7
- `MEDIUM`: 0.4 ≤ score < 0.7
- `LOW`: 0.0 < score < 0.4
- `CLEAN`: score = 0.0

---

### Score Event Sequence
Detect anomalies in event sequences.

**Request**:
```
POST /score/sequence
Content-Type: application/json

{
  "events": [
    "login_success",
    "view_account",
    "transfer",
    "logout"
  ]
}
```

**Valid Event Types**:
- `login_success` - Successful login
- `login_failed` - Failed login attempt
- `password_change` - Password changed
- `add_payee` - Added payee
- `navigate_help` - Navigated to help
- `view_account` - Viewed account
- `transfer` - Transfer initiated
- `max_transfer` - Max transfer amount
- `logout` - Logged out

**Response**:
```json
{
  "events": ["login_success", "view_account", "transfer", "logout"],
  "sequence_score": 0.25,
  "anomaly_risk_level": "LOW",
  "model": "sequence_detector",
  "timestamp": "2026-01-09T14:30:45.123456"
}
```

---

### Consolidate All Signals
Combine GBDT, Sequence Detector, and other signals into final risk score.

**Request**:
```
POST /score/consolidate
Content-Type: application/json

{
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
  "events": [
    "login_success",
    "view_account",
    "transfer",
    "logout"
  ]
}
```

**Response**:
```json
{
  "account_id": "ACC_0025",
  "timestamp": "2026-01-09T14:30:45.123456",
  "component_scores": {
    "gbdt": {
      "score": 0.45,
      "weight": 0.1,
      "status": "success"
    },
    "sequence": {
      "score": 0.25,
      "weight": 0.35,
      "status": "success"
    }
  },
  "consolidated_risk_score": 0.38,
  "risk_level": "LOW",
  "recommendation": "Allow - no suspicious activity detected",
  "timestamp": "2026-01-09T14:30:45.123456"
}
```

**Recommendations**:
- `HIGH` (≥0.7): "Block or require additional verification"
- `MEDIUM` (0.4-0.7): "Monitor closely, may require review"
- `LOW` (0.0-0.4): "Log for monitoring"
- `CLEAN` (0.0): "Allow - no suspicious activity detected"

---

### Batch Score Multiple Accounts
Score multiple accounts in a single request.

**Request**:
```
POST /batch/score
Content-Type: application/json

{
  "transactions": [
    {
      "account_id": "ACC_0001",
      "transaction": {
        "amount": 5000.0,
        "mcc": "5411",
        ...
      },
      "events": ["login_success", "transfer", "logout"]
    },
    {
      "account_id": "ACC_0002",
      "transaction": {
        "amount": 8000.0,
        ...
      },
      "events": ["login_failed", "login_failed", "login_failed"]
    }
  ]
}
```

**Response**:
```json
{
  "batch_id": "batch_20260109_143045",
  "total": 2,
  "results": [
    {
      "account_id": "ACC_0001",
      "consolidated_risk_score": 0.38,
      "risk_level": "LOW",
      "recommendation": "Allow - no suspicious activity detected",
      ...
    },
    {
      "account_id": "ACC_0002",
      "consolidated_risk_score": 0.82,
      "risk_level": "HIGH",
      "recommendation": "Block or require additional verification",
      ...
    }
  ],
  "summary": {
    "high_risk": 1,
    "medium_risk": 0,
    "low_risk": 1,
    "clean": 0
  },
  "timestamp": "2026-01-09T14:30:45.123456"
}
```

---

### Get Model Information
Retrieve information about loaded models.

**Request**:
```
GET /models/info
```

**Response**:
```json
{
  "available_models": ["gbdt", "sequence", "lstm", "gnn", "consolidator"],
  "metadata": {
    "gbdt": {
      "library": "lightgbm",
      "feature_count": 11,
      "feature_names": ["amt_log", "mcc_enc", ...],
      "metrics": {
        "accuracy": 0.894,
        "precision": 0.821,
        "recall": 0.756,
        "auc": 0.923
      },
      "timestamp": "2026-01-09T10:00:00"
    },
    "sequence": {
      "model_type": "lstm",
      "epochs_trained": 8,
      "metrics": {
        "accuracy": 0.856,
        "precision": 0.742,
        "recall": 0.821
      },
      "timestamp": "2026-01-09T10:00:00"
    }
  },
  "consolidator_weights": {
    "spatial": 0.2,
    "behavioral": 0.1,
    "temporal": 0.35,
    "lstm": 0.25,
    "cyber": 0.1
  },
  "timestamp": "2026-01-09T14:30:45.123456"
}
```

---

## Python Client Usage

### Basic Client
```python
from src.inference_client import InferenceClient

# Initialize
client = InferenceClient('http://localhost:5000')

# Health check
status = client.health_check()
print(f"API Status: {status['status']}")

# Score transaction
result = client.score_transaction({
    'amount': 5000.0,
    'mcc': '5411',
    'payment_type': 'card',
    'device_change': 0,
    'ip_risk': 0.1,
    'count_1h': 3,
    'sum_24h': 15000.0,
    'uniq_payees_24h': 2,
    'country': 'US'
})
print(f"Risk Level: {result['gbdt_risk_level']}")
print(f"Score: {result['gbdt_score']:.3f}")

# Score sequence
result = client.score_sequence(['login_success', 'transfer', 'logout'])
print(f"Anomaly Risk: {result['anomaly_risk_level']}")

# Consolidate
result = client.consolidate_risks(
    account_id='ACC_0025',
    transaction={...},
    events=['login_success', 'transfer']
)
print(f"Final Risk Score: {result['consolidated_risk_score']:.3f}")
print(f"Recommendation: {result['recommendation']}")

# Batch scoring
results = client.batch_score([
    {
        'account_id': 'ACC_0001',
        'transaction': {...},
        'events': [...]
    },
    ...
])
print(f"High Risk: {results['summary']['high_risk']}")
print(f"Medium Risk: {results['summary']['medium_risk']}")

client.close()
```

### Async Client
```python
import asyncio
from src.inference_client import InferenceClientAsync

async def main():
    async with InferenceClientAsync('http://localhost:5000') as client:
        # Score multiple accounts concurrently
        results = await asyncio.gather(
            client.score_transaction({...}),
            client.score_sequence([...]),
            client.consolidate_risks(...)
        )
        return results

# Run
asyncio.run(main())
```

---

## curl Examples

### Health Check
```bash
curl http://localhost:5000/health
```

### Score Transaction
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

### Score Sequence
```bash
curl -X POST http://localhost:5000/score/sequence \
  -H "Content-Type: application/json" \
  -d '{
    "events": ["login_success", "view_account", "transfer", "logout"]
  }'
```

### Consolidate Risks
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

### Get Models Info
```bash
curl http://localhost:5000/models/info | jq .
```

---

## Response Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid JSON or missing fields) |
| 500 | Server Error (model error or exception) |

### Error Response Format
```json
{
  "error": "Error description"
}
```

---

## Performance Considerations

### Response Times (Approximate)
- Health check: < 10ms
- Transaction scoring: 5-20ms
- Sequence scoring: 10-30ms
- Consolidation: 20-50ms
- Batch (100 items): 2-5 seconds

### Throughput
- Single endpoints: ~50-100 requests/second
- Batch endpoint: ~1000-2000 items/second (with batch_size=100)

### Resource Usage
- Memory: ~2-3 GB (all models loaded)
- CPU: Single-threaded, minimal usage at idle

### Optimization Tips
1. **Use batch endpoint** for multiple accounts (more efficient)
2. **Reuse client connection** (connection pooling)
3. **Cache model info** (retrieved once per startup)
4. **Use async client** for high concurrency

---

## Configuration

### Custom Port
```bash
python -c "
from src.inference_api import create_app, InferenceEngine
app = create_app()
app.run(host='0.0.0.0', port=8080)
"
```

### Custom Model Paths
Edit `InferenceEngine.load_models()` in inference_api.py to use custom paths:
```python
lstm_path = '/custom/path/to/lstm_link_predictor.pt'
```

---

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install flask torch pandas numpy

EXPOSE 5000

CMD ["python", "-m", "src.inference_api"]
```

### Docker Compose
```yaml
version: '3.9'
services:
  aml-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    environment:
      - FLASK_ENV=production
```

### Run in Production
```bash
# With gunicorn (production WSGI server)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 'src.inference_api:create_app()'

# With uWSGI
pip install uwsgi
uwsgi --http :5000 --wsgi-file src/inference_api.py --callable create_app
```

---

## Troubleshooting

### Models Not Loading
```
Error: GBDT model not loaded
```
**Solution**: Run `python main.py` to train and save models first

### Flask Not Installed
```
Error: Flask is required
```
**Solution**: `pip install flask`

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Use different port: `app.run(port=8080)`

### Model File Not Found
```
Error: FileNotFoundError: models/gbdt_metadata.json not found
```
**Solution**: Ensure models/ directory exists with model files from training

---

## Security Considerations

1. **Input Validation**: API validates JSON structure
2. **Error Handling**: Exceptions caught and logged
3. **HTTPS**: Deploy with SSL/TLS in production
4. **Authentication**: Add authentication layer in production:
   ```python
   from flask_httpauth import HTTPBasicAuth
   auth = HTTPBasicAuth()
   
   @app.before_request
   @auth.login_required
   def verify():
       pass
   ```
5. **Rate Limiting**: Add rate limiting for production:
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app)
   
   @app.route('/score/transaction')
   @limiter.limit("100 per minute")
   def score_transaction_endpoint():
       ...
   ```

---

## API Versioning

Current version: `1.0.0`

Future versions may be accessed at:
```
GET /v2/score/transaction
POST /v2/batch/score
```

---

## Support

For issues or questions:
1. Check MODEL_PERSISTENCE_IMPLEMENTATION.md
2. Review inference_api.py inline comments
3. Test endpoints with curl first
4. Check Flask error logs

---

**Status**: ✅ Production Ready  
**Last Updated**: January 9, 2026
