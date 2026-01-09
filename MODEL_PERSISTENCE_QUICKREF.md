# Model Persistence Quick Reference

## Summary: All 5 Models Now Support Save/Load

| Model | Type | Status | Save Function | Load Function | Output Files |
|-------|------|--------|---------------|---------------|--------------|
| **GNN** | GraphSAGE | ✅ NEW | `save_gnn_model()` | `load_gnn_model()` | gnn_model.pt, gnn_metadata.json |
| **Sequence** | LSTM/Transformer | ✅ NEW | `save_sequence_model()` | `load_sequence_model()` | sequence_detector_model.pt, metadata.json |
| **GBDT** | LightGBM/XGBoost/CatBoost | ✅ ENHANCED | `save_gbdt_model()` | `load_gbdt_model()` | lgb_model.txt (or xgb/catboost), gbdt_metadata.json |
| **LSTM Link** | LSTM Link Predictor | ✅ EXISTING | `save_model()` | `load_model()` | lstm_link_predictor.pt, lstm_metadata.json |
| **Risk Consolidator** | Parameter Store | ✅ EXISTING | (automatic in main) | (automatic in main) | consolidation_config.json |

## Usage Examples

### Train and Save (Automatic in main.py)
```python
# Run main orchestrator - trains and saves all models
python main.py
```

### Train Specific Model with Persistence
```python
# GNN
from src.gnn_trainer import train_demo
train_demo(num_nodes=800, epochs=50, save_model_flag=True)
# Output: models/gnn_model.pt + models/gnn_metadata.json

# Sequence Detector
from src.sequence_detector import demo_run
demo_run(num_sequences=2000, seq_len=20, epochs=8, save_model_flag=True)
# Output: models/sequence_detector_model.pt + metadata.json

# GBDT
from src.gbdt_detector import demo_run as gbdt_demo
gbdt_demo(n=20000, save_model_flag=True)
# Output: models/lgb_model.txt (or xgb_model.json) + gbdt_metadata.json
```

### Load and Inference
```python
# Load GNN
from src.gnn_trainer import load_gnn_model
model, metadata = load_gnn_model()
predictions = model(X, adjacency)

# Load Sequence Detector
from src.sequence_detector import load_sequence_model
model, metadata = load_sequence_model()
predictions = model(X)

# Load GBDT
from src.gbdt_detector import load_gbdt_model
model, metadata = load_gbdt_model()
predictions = model.predict(X)

# Load LSTM Link Predictor
from src.lstm_link_predictor import load_model as load_lstm
model, metadata = load_lstm('models/lstm_link_predictor.pt')
predictions = model(X)
```

### Disable Model Persistence (if needed)
```python
# Skip persistence - train model but don't save
from src.gnn_trainer import train_demo
train_demo(num_nodes=800, epochs=50, save_model_flag=False)
```

## File Locations

```
models/
├── gnn_model.pt                    # GraphSAGE model weights
├── gnn_metadata.json               # {model_type, num_nodes, epochs_trained, constraint_lambda, timestamp}
├── sequence_detector_model.pt      # LSTM/Transformer model weights
├── sequence_detector_metadata.json # {model_type, epochs_trained, metrics: {accuracy, precision, recall, ...}}
├── lgb_model.txt                   # LightGBM model (or xgb_model.json for XGBoost, catboost_model for CatBoost)
├── gbdt_metadata.json              # {library, feature_names, metrics: {accuracy, precision, recall, auc}}
├── lstm_link_predictor.pt          # LSTM link predictor weights
├── lstm_metadata.json              # {training_params, validation_stats}
└── consolidation_config.json       # {weights: {spatial, behavioral, temporal, lstm}, signal_thresholds, timestamp}
```

## Metadata Structure Examples

### GNN Metadata
```json
{
  "model_type": "graphsage",
  "num_nodes": 800,
  "epochs_trained": 50,
  "constraint_lambda": 1.0,
  "timestamp": 1704854321.45,
  "node_count": 800,
  "nodes_sample": ["A_00000", "A_00001", "A_00002", ...]
}
```

### Sequence Detector Metadata
```json
{
  "model_type": "lstm",
  "num_sequences": 2000,
  "epochs_trained": 8,
  "seq_len": 20,
  "timestamp": 1704854321.45,
  "num_event_types": 9,
  "event_types": ["login_success", "login_failed", ...],
  "metrics": {
    "accuracy": 0.856,
    "precision": 0.742,
    "recall": 0.821,
    "tp": 45,
    "fp": 8,
    "fn": 5,
    "tn": 142
  }
}
```

### GBDT Metadata
```json
{
  "library": "lightgbm",
  "timestamp": 1704854321.45,
  "feature_count": 11,
  "feature_names": ["amt_log", "mcc_enc", "payment_type_enc", ...],
  "metrics": {
    "accuracy": 0.894,
    "precision": 0.821,
    "recall": 0.756,
    "auc": 0.923
  }
}
```

## Key Features

✅ **5/5 Models Persisted**: Full coverage of all trained models
✅ **Load Utilities**: Easy-to-use load_*() functions for inference
✅ **Metadata**: Training parameters and performance metrics captured
✅ **Format Consistency**: All models use .pt (PyTorch) + .json (metadata)
✅ **Library Agnostic**: GBDT supports LightGBM, XGBoost, CatBoost
✅ **Backward Compatible**: Existing code still works unchanged
✅ **Orchestration**: main.py automatically trains and persists all models

## Integration Points

### main.py Phase [5.5]
Main orchestrator now includes optional model training with persistence:
- [5.5.1] GNN training
- [5.5.2] Sequence Detector training
- [5.5.3] GBDT training

Each phase can be independently disabled via `_HAS_GNN`, `_HAS_SEQUENCE`, `_HAS_GBDT` flags.

## Production Workflow

1. **Development**: Train models with main.py
2. **Persistence**: Models automatically saved to models/ directory
3. **Testing**: Load models and test inference
4. **Deployment**: Copy models/ directory to production
5. **Inference**: Use load_*() functions to score new transactions

## Troubleshooting

**Issue**: Model file not found when loading
```
FileNotFoundError: GNN model files not found in models/
```
**Solution**: Run main.py to train and save models first
```python
python main.py
```

**Issue**: Want to disable model persistence
```python
# In main.py, modify line ~326
train_gnn_demo(num_nodes=50, epochs=20, save_model_flag=False)  # Don't save
```

**Issue**: Want to use different GBDT library
```python
# train_gbdt() automatically detects available library (LightGBM > XGBoost > CatBoost)
# Or explicitly specify:
from src.gbdt_detector import train_gbdt
model, meta = train_gbdt(X, y, lib='xgboost', save_model_flag=True)
```

---

**Last Updated**: January 2026  
**Status**: ✅ All models persisted and loadable
