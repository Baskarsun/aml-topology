"""End-to-end small demo: embeddings -> pair sequences -> train LSTM -> print AUC

This script is intentionally small and uses the workspace simulator to
create synthetic transactions, then builds embeddings and trains a
lightweight LSTM model for link prediction.

If required packages (torch, sklearn) are missing the script will print
instructions to install them.
"""

import random
import numpy as np
import pandas as pd
import argparse

from src.simulator import TransactionSimulator
from src.embedding_builder import build_time_series_node_embeddings, build_pair_sequences_for_pairs
from src.lstm_link_predictor import LSTMLinkPredictor, train_model, predict_proba


def simple_label_for_pair(df: pd.DataFrame, u: str, v: str, cutoff_ts=None):
    """Label 1 if there exists at least one transaction from u->v after cutoff_ts (if provided),
    otherwise label 1 if any transaction exists in df from u->v.
    """
    if cutoff_ts is not None:
        df2 = df[pd.to_datetime(df['timestamp'], unit='s') > pd.to_datetime(cutoff_ts)]
    else:
        df2 = df
    return int(((df2['source'] == u) & (df2['target'] == v)).any())


def run_demo(random_seed: int = 42, freq: str = '1D', seq_len: int = 3):
    random.seed(random_seed)
    np.random.seed(random_seed)

    print("[Demo] Generating synthetic transaction data...")
    sim = TransactionSimulator(num_accounts=80)
    sim.generate_organic_traffic(num_transactions=800)
    # inject some typologies for signal diversity
    sim.inject_fan_in('ACC_0010', num_spokes=6, avg_amount=8000)
    sim.inject_fan_out('ACC_0025', num_beneficiaries=8, total_amount=60000)
    sim.inject_cycle(length=4, amount=15000)

    df = sim.get_dataframe()
    print(f"[Demo] Generated {len(df)} transactions and {len(df['source'].unique())} accounts.")

    print(f"[Demo] Building node embeddings (freq={freq})...")
    # optional: create some synthetic fraud scores for a few accounts
    fraud_scores = {f'ACC_{i:04d}': (0.9 if i % 17 == 0 else 0.0) for i in range(1, 200)}
    emb_map, feature_names = build_time_series_node_embeddings(df, freq=freq, fraud_scores=fraud_scores)
    print(f"[Demo] Built embeddings for {len(emb_map)} nodes; embedding dim={len(feature_names)}")

    # prepare pair list: sample random pairs from nodes
    nodes = list(emb_map.keys())
    if len(nodes) < 2:
        print("Not enough nodes with embeddings to build pairs. Exiting.")
        return

    pair_candidates = []
    # sample up to 2000 candidate pairs
    for _ in range(min(2000, max(200, len(nodes) * 5))):
        u, v = random.sample(nodes, 2)
        pair_candidates.append((u, v))

    print(f"[Demo] Building sequences for {len(pair_candidates)} candidate pairs (seq_len={seq_len})...")
    sequences, valid_pairs = build_pair_sequences_for_pairs(emb_map, pair_candidates, seq_len=seq_len, allow_padding=True)
    print(f"[Demo] {len(valid_pairs)} valid sequences built (with padding enabled).")

    if len(valid_pairs) == 0:
        print("No valid sequences (not enough aligned timestamps). Exiting demo.")
        return

    # create labels using simple heuristic: label=1 if any transaction u->v exists in df
    labels = []
    for (u, v) in valid_pairs:
        lbl = simple_label_for_pair(df, u, v)
        labels.append(lbl)
    labels = np.array(labels, dtype=np.float32)

    # if labels are all one class, randomize a few to ensure training works
    if len(np.unique(labels)) < 2:
        # flip some labels randomly
        n = len(labels)
        k = max(1, int(0.1 * n))
        inds = np.random.choice(n, k, replace=False)
        labels[inds] = 1 - labels[inds]

    # split into train/val
    n = len(sequences)
    idx = int(n * 0.8)
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[:idx], perm[idx:]
    X_train, y_train = sequences[train_idx], labels[train_idx]
    X_val, y_val = sequences[val_idx], labels[val_idx]

    input_size = sequences.shape[2]
    print(f"[Demo] Training LSTM (input_size={input_size}) on {len(X_train)} samples...")

    try:
        model = LSTMLinkPredictor(input_size=input_size, hidden_size=64, num_layers=1, dropout=0.1)
    except Exception as e:
        print("[Demo] Error constructing model (is torch installed?):", e)
        return

    try:
        model, history = train_model(model, X_train, y_train, epochs=10, batch_size=64, lr=1e-3, use_class_weight=True)
    except Exception as e:
        print("[Demo] Error during training (torch available and configured?):", e)
        return

    # evaluate on validation
    try:
        preds = predict_proba(model, X_val)
        # compute AUC if sklearn available
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_val, preds)
            print(f"[Demo] Validation AUC: {auc:.4f}")
        except Exception:
            # fallback: print balanced accuracy
            preds_bin = (preds >= 0.5).astype(float)
            acc = (preds_bin == y_val).mean()
            print(f"[Demo] sklearn not available — validation accuracy (fallback): {acc:.3f}")
    except Exception as e:
        print("[Demo] Error during prediction/evaluation:", e)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LSTM link-prediction demo')
    parser.add_argument('--freq', type=str, default='1D', help='bucket frequency for embeddings (pandas offset alias)')
    parser.add_argument('--seq-len', type=int, default=3, help='sequence length for pair sequences')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    run_demo(random_seed=args.seed, freq=args.freq, seq_len=args.seq_len)
