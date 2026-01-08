LSTM Link Predictor - Quick Guide

This document describes `src/lstm_link_predictor.py` usage.

Overview
- Model: `LSTMLinkPredictor` (PyTorch LSTM + linear head)
- Input: sequences shaped (N, seq_len, feat_dim)
- Task: Binary link prediction (probability a link forms in forecast horizon)

Basic training example
```python
import numpy as np
from src.lstm_link_predictor import LSTMLinkPredictor, train_model, predict_proba

# sequences: np.ndarray (N, seq_len, feat_dim)
# labels: np.ndarray (N,) binary

input_size = sequences.shape[2]
model = LSTMLinkPredictor(input_size=input_size, hidden_size=128)
model, history = train_model(model, sequences, labels, epochs=30, batch_size=128)
probs = predict_proba(model, sequences)
```

Building sequences
- Provide `emb_map` mapping node -> list of (timestamp, embedding)
- Use `build_pair_sequences_from_embeddings(emb_map, pair_list, seq_len)`
  to build sequences for node pairs; embeddings for a pair are concatenation of the two node embeddings at matching timestamps.

Notes
- `emb_map` entries must be sorted by timestamp ascending.
- If you already have pair-wise embeddings per timestep, you can skip building and directly create `sequences`.
- The module uses `sklearn` for AUC if available; otherwise AUC returns 0.0.

Recommendations
- Normalize embeddings (zero-mean, unit-variance) before training.
- Use balanced batches or class weighting for imbalanced datasets.
- Persist trained model with `torch.save(model.state_dict(), path)` and reload via `load_model`.

If you want, I can integrate a training/inference call into `main.py` and add an end-to-end example using your simulator data.
