"""LSTM-based link prediction for transaction graphs.

This module provides a simple, well-documented PyTorch LSTM classifier
that predicts the probability that a link will appear between a node pair
in the near future based on sequences of embeddings.

Expectations for inputs:
- `sequences`: numpy array or torch tensor shaped (N, seq_len, feat_dim)
- `labels`: binary labels shaped (N,) where 1 means link formed in horizon

Utility helpers are provided to build sequences when you have time-ordered
embeddings per node.
"""

from typing import Optional, Tuple, List
import os
import math

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMLinkPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, (hn, cn) = self.lstm(x)
        # use last hidden state from last layer
        if self.bidirectional:
            # concat forward/backward
            last = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last = hn[-1]
        logits = self.fc(last)
        return logits.squeeze(1)  # (B,)


# ======== Training / Utility functions ========

def train_model(model: nn.Module,
                sequences: np.ndarray,
                labels: np.ndarray,
                epochs: int = 20,
                batch_size: int = 256,
                lr: float = 1e-3,
                weight_decay: float = 1e-5,
                device: Optional[str] = None,
                val_split: float = 0.1,
                patience: int = 5,
                use_class_weight: bool = True) -> Tuple[nn.Module, dict]:
    """Train the LSTM link predictor.

    Args:
        use_class_weight: if True, compute pos_weight from class imbalance in labels
                         to handle imbalanced datasets (many negative, few positive links).

    Returns trained model and training history dict.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    X = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    n = len(X)
    idx = int(n * (1 - val_split))
    perm = torch.randperm(n)
    train_idx = perm[:idx]
    val_idx = perm[idx:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Compute pos_weight if class imbalance handling is enabled
    pos_weight = None
    if use_class_weight:
        num_neg = (y[train_idx] == 0).sum().item()
        num_pos = (y[train_idx] == 1).sum().item()
        if num_pos > 0:
            pos_weight = num_neg / num_pos
            print(f"[Train] Class imbalance detected: {num_neg} negatives, {num_pos} positives. "
                  f"pos_weight={pos_weight:.2f}")
        else:
            print("[Train] Warning: no positive samples in training set, using uniform loss weights")

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -math.inf
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train_loss = total_loss / len(train_ds)

        # validation
        model.eval()
        val_loss = 0.0
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
                trues.append(yb.cpu().numpy())
        avg_val_loss = val_loss / max(1, len(val_ds))
        preds = np.concatenate(preds) if preds else np.array([])
        trues = np.concatenate(trues) if trues else np.array([])

        # simple AUC if possible
        val_auc = _safe_auc(trues, preds)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)

        # early stopping by AUC
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # load best state
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, history


def predict_proba(model: nn.Module, sequences: np.ndarray, batch_size: int = 512,
                  device: Optional[str] = None) -> np.ndarray:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    X = torch.tensor(sequences, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
    return np.concatenate(preds) if preds else np.array([])


def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str, device: Optional[str] = None) -> nn.Module:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ======== Helpers to build sequences from embeddings ========

def sequences_from_node_pair_embeddings(embeddings: List[Tuple[float, np.ndarray]],
                                        seq_len: int) -> Optional[np.ndarray]:
    """Given a time-ordered list of (timestamp, embedding) tuples for a node pair,
    return a single sequence of length seq_len (most recent). If there are fewer
    than seq_len entries, returns None.

    embeddings must be sorted by timestamp ascending.
    """
    if len(embeddings) < seq_len:
        return None
    # take last seq_len embeddings
    arr = np.stack([e for _, e in embeddings[-seq_len:]], axis=0)
    return arr.astype(np.float32)


def build_pair_sequences_from_embeddings(emb_map: dict,
                                          pair_list: List[Tuple[str, str]],
                                          seq_len: int) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Build sequences for a list of node pairs.

    emb_map: dict mapping (node) -> list of (timestamp, embedding) sorted ascending.
    pair_list: list of (u, v) pairs to construct sequences for.

    Returns (sequences_array, valid_pairs) where sequences_array shape is (N, seq_len, feat_dim).
    """
    sequences = []
    valid_pairs = []
    for u, v in pair_list:
        # create combined embedding timeline for the pair (simple concat of node embeddings)
        u_list = emb_map.get(u, [])
        v_list = emb_map.get(v, [])
        # align by timestamps: find timestamps where both have embeddings
        i = 0
        j = 0
        merged = []
        while i < len(u_list) and j < len(v_list):
            t_u, e_u = u_list[i]
            t_v, e_v = v_list[j]
            if t_u == t_v:
                merged.append((t_u, np.concatenate([e_u, e_v], axis=0)))
                i += 1
                j += 1
            elif t_u < t_v:
                i += 1
            else:
                j += 1
        seq = sequences_from_node_pair_embeddings(merged, seq_len)
        if seq is not None:
            sequences.append(seq)
            valid_pairs.append((u, v))
    if len(sequences) == 0:
        return np.zeros((0, seq_len, 0), dtype=np.float32), []
    return np.stack(sequences, axis=0), valid_pairs


# small safe AUC util
def _safe_auc(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return 0.0
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0
