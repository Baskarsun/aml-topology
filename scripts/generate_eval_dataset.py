#!/usr/bin/env python3
"""
Phase 2, Task 2.1 — Generate Labelled Evaluation Dataset

Generates 50,000 synthetic transactions with a fixed seed for reproducibility.
Splits 80/20 (stratified), saves the 10,000-row held-out test set plus metadata.

Usage:
    python scripts/generate_eval_dataset.py
"""

import os
import sys
import json
import hashlib
import datetime

import numpy as np
import pandas as pd

# Allow imports from src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.gbdt_detector import generate_synthetic_transactions, featurize, train_test_split_manual

SEED = 2025
N_TOTAL = 50_000
TEST_FRAC = 0.2
DATA_DIR = os.path.join(ROOT, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'eval_dataset.csv')
META_PATH = os.path.join(DATA_DIR, 'eval_dataset_metadata.json')


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Generating {N_TOTAL:,} synthetic transactions (seed={SEED})...")
    df = generate_synthetic_transactions(n=N_TOTAL, seed=SEED)

    print(f"Total rows: {len(df):,}  |  Fraud rate: {df['label'].mean():.3%}")

    # Stratified 80/20 split
    y = df['label']
    df_train, df_test, _, _ = train_test_split_manual(
        df, y, test_size=TEST_FRAC, random_state=SEED, stratify=y.values
    )

    # Derive feature encoding maps from the training portion only
    _, maps = featurize(df_train)

    # Save held-out test set
    df_test.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df_test):,} test rows → {CSV_PATH}")

    # Checksum of the CSV
    sha256 = hashlib.sha256(open(CSV_PATH, 'rb').read()).hexdigest()

    metadata = {
        'seed': SEED,
        'n_total': N_TOTAL,
        'n_train': len(df_train),
        'n_test': len(df_test),
        'fraud_rate_test': float(df_test['label'].mean()),
        'fraud_rate_train': float(df_train['label'].mean()),
        'generation_date': datetime.datetime.utcnow().isoformat() + 'Z',
        'csv_sha256': sha256,
        'maps': {
            'mcc_map': maps['mcc_map'],
            'pay_map': maps['pay_map'],
        },
    }

    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Test fraud rate : {metadata['fraud_rate_test']:.3%}")
    print(f"SHA-256         : {sha256}")
    print(f"Metadata saved  → {META_PATH}")


if __name__ == '__main__':
    main()
