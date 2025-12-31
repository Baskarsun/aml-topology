import os
import sys
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# simple deterministic seed for reproducibility in demo
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


EVENT_TYPES = [
    'login_success', 'login_failed', 'password_change', 'add_payee',
    'navigate_help', 'view_account', 'transfer', 'max_transfer', 'logout'
]
EVENT2IDX = {e: i for i, e in enumerate(EVENT_TYPES)}


def generate_synthetic_event_sequences(num_sequences: int = 2000,
                                       max_len: int = 20,
                                       fraud_rate: float = 0.12) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sequences of events and binary labels.

    Each sequence is a list of events with auxiliary numeric fields (amount, dt, device_change).
    Returns:
      X: numpy array shape (N, T, D) where D = 3 (event_idx, amount, device_change)
      y: numpy array shape (N,) labels 0/1
    """
    N = num_sequences
    T = max_len
    D = 3
    X = np.zeros((N, T, D), dtype=float)
    y = np.zeros(N, dtype=int)

    for i in range(N):
        is_fraud = random.random() < fraud_rate
        y[i] = 1 if is_fraud else 0

        seq = []
        # start with login attempts
        if is_fraud:
            # pattern: failed logins -> password change -> add_payee -> max_transfer
            failures = random.randint(1, 4)
            for _ in range(failures):
                seq.append(('login_failed', 0.0, 0))
            seq.append(('login_success', 0.0, 1))
            # short pause
            seq.append(('password_change', 0.0, 1))
            # possibly navigation to payee pages
            seq.append(('add_payee', 0.0, 1))
            # immediate big transfer
            amt = float(np.random.uniform(2000, 50000))
            seq.append(('max_transfer', amt, 1))
            # fill rest with benign navigation
            while len(seq) < T:
                seq.append((random.choice(['view_account','navigate_help','logout']), 0.0, 0))
        else:
            # benign user: successful login, some browsing, occasional small transfers
            seq.append(('login_success', 0.0, 0))
            steps = random.randint(3, T-3)
            for _ in range(steps):
                ev = random.choices(['view_account','navigate_help','transfer','add_payee','logout'], [0.4,0.2,0.15,0.05,0.2])[0]
                if ev == 'transfer':
                    amt = float(np.random.uniform(1, 1000))
                    seq.append((ev, amt, random.choice([0,0,1])))
                else:
                    seq.append((ev, 0.0, 0))
            while len(seq) < T:
                seq.append(('logout', 0.0, 0))

        # encode into X row
        for t in range(T):
            ev, amt, devchg = seq[t]
            X[i, t, 0] = EVENT2IDX.get(ev, 0)
            # normalize amount by a log-scale to keep magnitude stable
            X[i, t, 1] = np.log1p(amt) / 10.0
            X[i, t, 2] = float(devchg)

    return X, y


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMDetector(nn.Module):
    def __init__(self, n_event_types: int, emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.event_emb = nn.Embedding(n_event_types, emb_dim)
        # scalar features: amount, device_change -> 2 dims
        self.scalar_proj = nn.Linear(2, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim * 2, hidden_size=hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        # x: (B, T, D) where D=3 (event_idx, amount, devchg)
        ev_idx = x[:, :, 0].long()
        scalar = x[:, :, 1:]
        e = self.event_emb(ev_idx)
        s = torch.relu(self.scalar_proj(scalar))
        inp = torch.cat([e, s], dim=2)
        out, _ = self.lstm(inp)
        # use last hidden
        h = out[:, -1, :]
        return self.fc(h).squeeze(1)


class TransformerDetector(nn.Module):
    def __init__(self, n_event_types: int, emb_dim: int = 32, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.event_emb = nn.Embedding(n_event_types, emb_dim)
        self.scalar_proj = nn.Linear(2, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim * 2, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Sequential(nn.Linear(emb_dim * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        # x: (B, T, D)
        ev_idx = x[:, :, 0].long()
        scalar = x[:, :, 1:]
        e = self.event_emb(ev_idx)
        s = torch.relu(self.scalar_proj(scalar))
        inp = torch.cat([e, s], dim=2)  # (B, T, C)
        # transformer expects (T, B, C)
        src = inp.permute(1, 0, 2)
        out = self.transformer(src)
        out = out.permute(1, 2, 0)  # (B, C, T)
        pooled = self.pool(out).squeeze(2)
        return self.out(pooled).squeeze(1)


def train_sequence_model(model: nn.Module, train_dl: DataLoader, test_dl: DataLoader, epochs: int = 10, lr: float = 1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb = xb
            yb = yb
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_dl.dataset)

        # eval
        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for xb, yb in test_dl:
                out = model(xb)
                all_pred.append(out.cpu().numpy())
                all_true.append(yb.cpu().numpy())
        preds = np.concatenate(all_pred)
        trues = np.concatenate(all_true)
        pred_bin = (preds > 0.5).astype(int)
        tp = int(((pred_bin == 1) & (trues == 1)).sum())
        fp = int(((pred_bin == 1) & (trues == 0)).sum())
        fn = int(((pred_bin == 0) & (trues == 1)).sum())
        tn = int(((pred_bin == 0) & (trues == 0)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        acc = (tp + tn) / len(trues)
        print(f"Epoch {ep}/{epochs} loss={avg_loss:.4f} test_acc={acc:.4f} prec={prec:.4f} rec={rec:.4f}")

    return model


def demo_run(num_sequences: int = 2000, seq_len: int = 20, epochs: int = 8, model_type: str = 'lstm'):
    X, y = generate_synthetic_event_sequences(num_sequences, seq_len, fraud_rate=0.12)
    # split
    N = len(X)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx = idx[:split]
    test_idx = idx[split:]

    ds_train = SequenceDataset(X[train_idx], y[train_idx])
    ds_test = SequenceDataset(X[test_idx], y[test_idx])
    dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=128)

    if model_type == 'lstm':
        model = LSTMDetector(len(EVENT_TYPES))
    else:
        model = TransformerDetector(len(EVENT_TYPES))

    device = 'cpu'
    model.to(device)

    model = train_sequence_model(model, dl_train, dl_test, epochs=epochs)

    # final metrics on test
    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for xb, yb in dl_test:
            out = model(xb)
            all_pred.append(out.cpu().numpy())
            all_true.append(yb.cpu().numpy())
    preds = np.concatenate(all_pred)
    trues = np.concatenate(all_true)
    pred_bin = (preds > 0.5).astype(int)
    tp = int(((pred_bin == 1) & (trues == 1)).sum())
    fp = int(((pred_bin == 1) & (trues == 0)).sum())
    fn = int(((pred_bin == 0) & (trues == 1)).sum())
    tn = int(((pred_bin == 0) & (trues == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    acc = (tp + tn) / len(trues)
    print(f"Final Test — acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} tp={tp} fp={fp} fn={fn} tn={tn}")

    # print some high-risk examples
    idx_sorted = np.argsort(-preds)
    print("Top risky sequences (index within test set, score, true_label):")
    for i in idx_sorted[:10]:
        print(i, f"{preds[i]:.3f}", int(trues[i]))


if __name__ == '__main__':
    print("Running sequence detector demo (LSTM) — detecting APP-like event patterns...")
    demo_run(num_sequences=2000, seq_len=20, epochs=8, model_type='lstm')
