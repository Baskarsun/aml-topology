# Model Persistence Analysis - AML Topology

**Date:** January 9, 2026  
**Status:** Comprehensive workspace analysis of all trained models and persistence mechanisms

---

## Executive Summary

| Total Models | Persisted | Not Persisted | Persistence Rate |
|---|---|---|---|
| **5** | **2** | **3** | **40%** |

---

## Detailed Model Analysis

### 1. **LSTM Link Predictor** ✅ PERSISTED
**File:** `src/lstm_link_predictor.py`  
**Called From:** `main.py` (Phase 5)

| Aspect | Status | Details |
|--------|--------|---------|
| Model Save | ✅ Yes | `torch.save(model.state_dict(), path)` |
| Save Location | ✅ Yes | `models/lstm_link_predictor.pt` (107.8 KB) |
| Metadata | ✅ Yes | `models/lstm_metadata.json` (658 bytes) |
| Trainable? | ✅ Yes | Full PyTorch LSTM model with optimizer |
| Load Support | ✅ Yes | `load_model()` function available |
| Integration | ✅ Yes | Integrated in `main.py` Phase 5 |

**Metadata Saved:**
```json
{
  "input_size": 36,
  "hidden_size": 64,
  "num_layers": 1,
  "dropout": 0.1,
  "num_sequences": 1225,
  "num_epochs_trained": 6,
  "final_val_auc": 0.0,
  "final_train_loss": 0.995,
  "feature_names": [18 features]
}
```

---

### 2. **Risk Consolidator** ✅ PERSISTED
**File:** `src/risk_consolidator.py`  
**Called From:** `main.py` (Phase 6)

| Aspect | Status | Details |
|--------|--------|---------|
| Weights Save | ✅ Yes | Phase weights to JSON |
| Save Location | ✅ Yes | `models/consolidation_config.json` (467 bytes) |
| Metadata | ✅ Yes | Thresholds, config, timestamp |
| Trainable? | ⚠️  No | Weights are configuration-based, not learned |
| Load Support | ✅ Yes | Can reload config from JSON |
| Integration | ✅ Yes | Integrated in `main.py` Phase 6 |

**Config Saved:**
```json
{
  "weights": {
    "spatial": 0.20,
    "behavioral": 0.10,
    "temporal": 0.35,
    "lstm": 0.25,
    "cyber": 0.10
  },
  "signal_thresholds": {...},
  "normalize_output": true,
  "timestamp": "2026-01-09T06:15:25.765849"
}
```

---

### 3. **GNN/GraphSAGE Model** ❌ NOT PERSISTED
**File:** `src/gnn_trainer.py`  
**Functions:** `train_demo()`, `train_demo_hetero()`

| Aspect | Status | Details |
|--------|--------|---------|
| Model Training | ✅ Yes | PyTorch nn.Module GNN/GraphSAGE |
| Model Save | ❌ No | No `torch.save()` call found |
| Save Location | ❌ No | Not implemented |
| Metadata | ❌ No | Training config not saved |
| Trainable? | ✅ Yes | Full neural network model |
| Load Support | ❌ No | No load mechanism |
| Integration | ❌ No | Not called from `main.py` |
| Current Usage | Demo Only | `train_demo()` and `train_demo_hetero()` |

**Issue:** Model is trained in-memory but discarded after run. No checkpoint/persistence mechanism.

---

### 4. **Sequence Detector (LSTM/Transformer)** ❌ NOT PERSISTED
**File:** `src/sequence_detector.py`  
**Functions:** `train_sequence_model()`, `demo_run()`

| Aspect | Status | Details |
|--------|--------|---------|
| Model Training | ✅ Yes | LSTMDetector or TransformerDetector |
| Model Save | ❌ No | No save call in training function |
| Save Location | ❌ No | Not implemented |
| Metadata | ❌ No | No metrics saved beyond console output |
| Trainable? | ✅ Yes | Full sequence-to-label model |
| Load Support | ❌ No | No load mechanism |
| Integration | ❌ No | Not called from `main.py` |
| Current Usage | Demo Only | `demo_run()` function |

**Issue:** After training completes, model is only returned; not persisted to disk.

---

### 5. **GBDT (LightGBM/XGBoost/CatBoost)** ⚠️  PARTIAL
**File:** `src/gbdt_detector.py`  
**Functions:** `train_gbdt()`, `demo_run()`

| Aspect | Status | Details |
|--------|--------|---------|
| Model Training | ✅ Yes | LightGBM, XGBoost, or CatBoost classifier |
| Model Save | ⚠️  Conditional | Only if LightGBM library is available |
| Save Location | ⚠️  Conditional | `models/lgb_model.txt` (if LightGBM) |
| Metadata | ❌ No | No metadata file for model config |
| Trainable? | ✅ Yes | Boosted gradient tree ensemble |
| Load Support | ❌ No | No load function in codebase |
| Integration | ❌ No | Not called from `main.py` |
| Current Usage | Demo Only | `demo_run()` function |

**Issue:** 
- Only saves if using LightGBM (not XGBoost or CatBoost)
- No metadata/config saved
- No load mechanism to restore saved models
- Not integrated into main orchestration pipeline

---

## Summary Table: Persistence Status

| Model | Type | Status | Save Location | Integration | Production Ready? |
|---|---|---|---|---|---|
| LSTM Link Predictor | PyTorch | ✅ Persisted | `models/*.pt` + metadata | ✅ `main.py` | ✅ Yes |
| Risk Consolidator | Config-based | ✅ Persisted | `models/*.json` | ✅ `main.py` | ✅ Yes |
| GNN/GraphSAGE | PyTorch | ❌ Not Persisted | — | ❌ Demo only | ❌ No |
| Sequence Detector | PyTorch | ❌ Not Persisted | — | ❌ Demo only | ❌ No |
| GBDT (LightGBM) | Sklearn-like | ⚠️ Partial | `models/lgb_model.txt` | ❌ Demo only | ⚠️ Incomplete |

---

## Recommendations

### Priority 1: Add Persistence to GNN Model
```python
# In src/gnn_trainer.py, after training completes:
def train_demo(num_nodes=1000, epochs=60, ...):
    # ... training code ...
    
    # Save model weights
    os.makedirs('models', exist_ok=True)
    gnn_model_path = 'models/gnn_model.pt'
    torch.save(model.state_dict(), gnn_model_path)
    
    # Save metadata
    gnn_metadata = {
        'num_nodes': num_nodes,
        'epochs_trained': epochs,
        'constraint_lambda': constraint_lambda,
        'model_type': 'GraphSAGE'
    }
    with open('models/gnn_metadata.json', 'w') as f:
        json.dump(gnn_metadata, f, indent=2)
    
    return model, nodes_sorted, out.numpy(), y.numpy()
```

### Priority 2: Add Persistence to Sequence Detector
```python
# In src/sequence_detector.py:
def train_sequence_model(...):
    # ... training code ...
    
    # Save after training
    model_path = 'models/sequence_detector_model.pt'
    torch.save(model.state_dict(), model_path)
    return model
```

### Priority 3: Complete GBDT Persistence
```python
# In src/gbdt_detector.py:
def train_gbdt(...):
    # ... training code ...
    
    # Add for all GBDT libraries (not just LightGBM)
    if GBDT_LIB == 'lightgbm':
        model.save_model('models/lgb_model.txt')
    elif GBDT_LIB == 'xgboost':
        model.save_model('models/xgb_model.json')  # or .pkl
    elif GBDT_LIB == 'catboost':
        model.save_model('models/catboost_model')
    
    # Always save metadata
    meta = {'library': GBDT_LIB, 'training_params': ...}
    with open('models/gbdt_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
```

### Priority 4: Integrate All Models into Main Pipeline
- Call `train_gnn_demo()` and persist results in Phase X
- Call `sequence_detector.demo_run()` and persist results in Phase Y
- Call `gbdt_detector.train_gbdt()` with full persistence
- Add model loading utilities for inference-only runs

---

## Files Affected

| File | Change Type | Effort |
|---|---|---|
| `src/gnn_trainer.py` | Add save logic | 🟡 Medium |
| `src/sequence_detector.py` | Add save logic | 🟢 Low |
| `src/gbdt_detector.py` | Enhance existing save | 🟡 Medium |
| `main.py` | Integrate models into phases | 🟠 High |
| `models/` | New metadata files | 🟢 Low |

---

## Next Steps

1. ✅ **Verify LSTM & Consolidator** - Already done
2. 🔲 **Implement GNN persistence** - Save weights + metadata
3. 🔲 **Implement Sequence Detector persistence** - Save LSTM/Transformer weights
4. 🔲 **Complete GBDT persistence** - Handle all libraries + metadata
5. 🔲 **Update main.py** - Integrate all models into main pipeline
6. 🔲 **Add load utilities** - Enable inference-only mode (load pre-trained models)
7. 🔲 **Testing** - Verify save/load round-trip for all models

---

**Generated:** 2026-01-09  
**Total Models Analyzed:** 5  
**Persistence Coverage:** 40% (2/5 fully persisted)
