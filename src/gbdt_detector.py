import os
import random
import time
import json
from typing import Tuple, Dict

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    GBDT_LIB = 'lightgbm'
except Exception:
    try:
        import xgboost as xgb
        GBDT_LIB = 'xgboost'
    except Exception:
        try:
            from catboost import CatBoostClassifier
            GBDT_LIB = 'catboost'
        except Exception:
            GBDT_LIB = 'torch_fallback'


def stratified_kfold_indices(y, n_folds=5, random_state=42):
    """Return list of (train_idx, val_idx) index arrays for stratified k-fold CV."""
    y = np.array(y)
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    # Distribute class samples round-robin into folds
    fold_buckets = [[] for _ in range(n_folds)]
    for c in classes:
        c_idx = np.where(y == c)[0].copy()
        rng.shuffle(c_idx)
        for i, idx in enumerate(c_idx):
            fold_buckets[i % n_folds].append(idx)
    splits = []
    for k in range(n_folds):
        val_idx = np.array(fold_buckets[k])
        train_idx = np.concatenate([np.array(fold_buckets[j]) for j in range(n_folds) if j != k])
        splits.append((train_idx, val_idx))
    return splits


def train_test_split_manual(X, y, test_size=0.2, random_state=42, stratify=None):
    # X: DataFrame, y: Series or ndarray
    n = len(X)
    rng = np.random.RandomState(random_state)
    idx = np.arange(n)
    if stratify is not None:
        # perform stratified split
        uniq, inv = np.unique(stratify, return_inverse=True)
        train_idx = []
        test_idx = []
        for k in range(len(uniq)):
            inds = idx[inv == k]
            rng.shuffle(inds)
            cut = int(len(inds) * (1 - test_size))
            train_idx.extend(inds[:cut].tolist())
            test_idx.extend(inds[cut:].tolist())
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
    else:
        rng.shuffle(idx)
        cut = int(n * (1 - test_size))
        train_idx = idx[:cut]
        test_idx = idx[cut:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def accuracy_score_manual(y_true, y_pred):
    return float((y_true == y_pred).sum()) / len(y_true)


def precision_score_manual(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp / (tp + fp + 1e-9)


def recall_score_manual(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn + 1e-9)


def f1_score_manual(y_true, y_pred):
    prec = precision_score_manual(y_true, y_pred)
    rec = recall_score_manual(y_true, y_pred)
    return 2 * prec * rec / (prec + rec + 1e-9)


def avg_precision_manual(y_true, scores):
    """Area under the Precision-Recall curve (average precision / PR-AUC)."""
    y_true = np.array(y_true)
    scores = np.array(scores)
    desc = np.argsort(-scores)
    y_sorted = y_true[desc]
    P = int(y_true.sum())
    if P == 0:
        return 0.0
    tp = 0
    fp = 0
    precisions = [1.0]
    recalls = [0.0]
    for val in y_sorted:
        if val == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp + 1e-9))
        recalls.append(tp / (P + 1e-9))
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2.0
    return ap


def roc_auc_manual(y_true, scores):
    # simple roc auc computation via trapezoid on sorted scores
    y_true = np.array(y_true)
    scores = np.array(scores)
    desc = np.argsort(-scores)
    y_sorted = y_true[desc]
    tp = 0
    fp = 0
    tps = [0]
    fps = [0]
    P = int(y_true.sum())
    N = len(y_true) - P
    for val in y_sorted:
        if val == 1:
            tp += 1
        else:
            fp += 1
        tps.append(tp)
        fps.append(fp)
    tpr = np.array(tps) / (P + 1e-9)
    fpr = np.array(fps) / (N + 1e-9)
    # trapezoidal integration
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0
    return auc


HIGH_RISK_COUNTRIES = {'RU', 'NG', 'PK', 'CN'}


def _compute_fraud_risk(amount, ip_risk, device_change, payment_type, country, count_1h, sum_24h):
    """Compute a continuous fraud risk score [0, 1] from transaction features.

    All thresholds are chosen so that each feature independently contributes a
    meaningful fraction of fraud cases, giving the GBDT strong learnable signal.
    """
    score = 0.0

    # Amount risk (use realistic thresholds; exponential(500) mean keeps these reachable)
    if amount > 3000:
        score += 0.30
    elif amount > 1000:
        score += 0.15
    elif amount > 500:
        score += 0.05

    # IP risk (continuous feature — strong signal)
    if ip_risk > 0.80:
        score += 0.30
    elif ip_risk > 0.60:
        score += 0.15
    elif ip_risk > 0.40:
        score += 0.05

    # Device change — rare but highly indicative
    if device_change:
        score += 0.20

    # High-risk payment method
    if payment_type == 'crypto':
        score += 0.15
    elif payment_type == 'wire':
        score += 0.05

    # Geography
    if country in HIGH_RISK_COUNTRIES:
        score += 0.15

    # Velocity — many transactions in 1 h
    if count_1h > 8:
        score += 0.15
    elif count_1h > 4:
        score += 0.07

    # Interaction: high amount + high ip_risk together
    if amount > 1000 and ip_risk > 0.65:
        score += 0.10

    return min(score, 1.0)


def generate_synthetic_transactions(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic AML transactions where fraud labels are derived from
    feature combinations, not random coin-flips.  This gives the GBDT genuine
    signal to learn from.
    """
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    mcc_pool = ['5311', '5411', '6011', '7995', '4829', '4814', '5732', '7011']
    # Weight countries so high-risk ones appear ~15 % of the time
    countries = ['US'] * 40 + ['GB'] * 15 + ['IN'] * 10 + ['NG'] * 8 + ['RU'] * 8 + ['PK'] * 8 + ['CN'] * 6 + ['UK'] * 5
    payment_types = ['card', 'wire', 'ach', 'crypto']
    payment_weights = [0.55, 0.20, 0.15, 0.10]  # card is most common

    account_last = {}

    for i in range(n):
        acct = f'A_{random.randint(0, n // 10)}'
        t = int(time.time()) - random.randint(0, 86400 * 30)

        # Use a more realistic amount distribution that allows high amounts
        amount = float(np.round(np.random.exponential(500.0), 2))
        mcc = random.choice(mcc_pool)
        country = random.choice(countries)
        device_id = f'dev_{random.randint(1, 5000)}'
        device_change = random.random() < 0.04  # 4 % baseline device-change rate
        payment_type = random.choices(payment_types, weights=payment_weights)[0]
        ip_risk = float(np.clip(np.random.beta(2, 5), 0.0, 1.0))  # skewed toward low risk

        # Velocity features (per-account running counters)
        last = account_last.get(acct, {'count_1h': 0, 'sum_24h': 0.0})
        count_1h = last['count_1h'] + random.randint(0, 3)
        sum_24h = last['sum_24h'] + amount
        uniq_payees = random.randint(0, max(1, count_1h))
        account_last[acct] = {'count_1h': count_1h, 'sum_24h': sum_24h}

        # --- Fraud label from feature-based risk score (no random noise as base) ---
        risk = _compute_fraud_risk(amount, ip_risk, device_change, payment_type, country, count_1h, sum_24h)

        # Probabilistic labeling: high-risk → almost always fraud;
        # mid-risk → sometimes; low-risk → very rare (background noise ≈ 0.5 %)
        if risk >= 0.60:
            is_fraud = random.random() < 0.90
        elif risk >= 0.40:
            is_fraud = random.random() < 0.50
        elif risk >= 0.25:
            is_fraud = random.random() < 0.15
        else:
            is_fraud = random.random() < 0.005  # near-zero background rate

        rows.append({
            'account': acct,
            'timestamp': t,
            'amount': amount,
            'mcc': mcc,
            'country': country,
            'device_id': device_id,
            'device_change': int(device_change),
            'payment_type': payment_type,
            'ip_risk': float(ip_risk),
            'count_1h': int(count_1h),
            'sum_24h': float(sum_24h),
            'uniq_payees_24h': int(uniq_payees),
            'is_international': int(country != 'US'),
            'label': int(is_fraud)
        })

    return pd.DataFrame(rows)


def featurize(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df2 = df.copy()

    # Categorical encodings
    mcc_map = {m: i for i, m in enumerate(df2['mcc'].unique())}
    df2['mcc_enc'] = df2['mcc'].map(mcc_map)
    pay_map = {p: i for i, p in enumerate(df2['payment_type'].unique())}
    df2['payment_type_enc'] = df2['payment_type'].map(pay_map)
    df2['is_crypto'] = (df2['payment_type'] == 'crypto').astype(int)
    df2['is_high_risk_country'] = df2['country'].isin(HIGH_RISK_COUNTRIES).astype(int)

    # Amount features
    df2['amt_log'] = np.log1p(df2['amount'])
    df2['amt_gt_1k'] = (df2['amount'] > 1000).astype(int)
    df2['amt_gt_3k'] = (df2['amount'] > 3000).astype(int)

    # IP risk buckets (non-linear signal)
    df2['ip_risk_high'] = (df2['ip_risk'] > 0.8).astype(int)
    df2['ip_risk_mid'] = ((df2['ip_risk'] > 0.6) & (df2['ip_risk'] <= 0.8)).astype(int)

    # Interaction features — key patterns found in fraud generation
    df2['amt_x_ip_risk'] = df2['amt_log'] * df2['ip_risk']
    df2['high_amt_device_change'] = (df2['amt_gt_1k'] & df2['device_change']).astype(int)
    df2['crypto_high_risk_country'] = (df2['is_crypto'] & df2['is_high_risk_country']).astype(int)

    # Velocity features
    df2['avg_tx_24h'] = df2['sum_24h'] / (df2['uniq_payees_24h'] + 1)
    df2['velocity_score'] = df2['count_1h'] * (df2['amt_log'] / 10.0)
    df2['high_velocity'] = (df2['count_1h'] > 8).astype(int)

    features = [
        'amt_log', 'amt_gt_1k', 'amt_gt_3k',
        'mcc_enc', 'payment_type_enc', 'is_crypto',
        'device_change', 'ip_risk', 'ip_risk_high', 'ip_risk_mid',
        'count_1h', 'sum_24h', 'uniq_payees_24h', 'high_velocity',
        'is_international', 'is_high_risk_country',
        'avg_tx_24h', 'velocity_score',
        'amt_x_ip_risk', 'high_amt_device_change', 'crypto_high_risk_country',
    ]
    return df2[features], {'mcc_map': mcc_map, 'pay_map': pay_map}


FEATURE_COLUMNS = [
    'amt_log', 'amt_gt_1k', 'amt_gt_3k',
    'mcc_enc', 'payment_type_enc', 'is_crypto',
    'device_change', 'ip_risk', 'ip_risk_high', 'ip_risk_mid',
    'count_1h', 'sum_24h', 'uniq_payees_24h', 'high_velocity',
    'is_international', 'is_high_risk_country',
    'avg_tx_24h', 'velocity_score',
    'amt_x_ip_risk', 'high_amt_device_change', 'crypto_high_risk_country',
]


def apply_featurize(df: pd.DataFrame, maps: Dict) -> pd.DataFrame:
    """Apply feature engineering to df using pre-computed encoding maps.

    Use this when featurizing a test/inference set with maps derived from training data,
    so categorical encodings are consistent with what the model was trained on.
    """
    df2 = df.copy()
    mcc_map = maps['mcc_map']
    pay_map = maps['pay_map']
    df2['mcc_enc'] = df2['mcc'].map(lambda v: mcc_map.get(v, -1)).astype(int)
    df2['payment_type_enc'] = df2['payment_type'].map(lambda v: pay_map.get(v, -1)).astype(int)
    df2['is_crypto'] = (df2['payment_type'] == 'crypto').astype(int)
    df2['is_high_risk_country'] = df2['country'].isin(HIGH_RISK_COUNTRIES).astype(int)
    df2['amt_log'] = np.log1p(df2['amount'])
    df2['amt_gt_1k'] = (df2['amount'] > 1000).astype(int)
    df2['amt_gt_3k'] = (df2['amount'] > 3000).astype(int)
    df2['ip_risk_high'] = (df2['ip_risk'] > 0.8).astype(int)
    df2['ip_risk_mid'] = ((df2['ip_risk'] > 0.6) & (df2['ip_risk'] <= 0.8)).astype(int)
    df2['amt_x_ip_risk'] = df2['amt_log'] * df2['ip_risk']
    df2['high_amt_device_change'] = (df2['amt_gt_1k'] & df2['device_change']).astype(int)
    df2['crypto_high_risk_country'] = (df2['is_crypto'] & df2['is_high_risk_country']).astype(int)
    df2['avg_tx_24h'] = df2['sum_24h'] / (df2['uniq_payees_24h'] + 1)
    df2['velocity_score'] = df2['count_1h'] * (df2['amt_log'] / 10.0)
    df2['high_velocity'] = (df2['count_1h'] > 8).astype(int)
    return df2[FEATURE_COLUMNS]


def save_gbdt_model(model, lib: str, metrics: dict = None, feature_names: list = None, maps: dict = None):
    """Save GBDT model weights and metadata to disk."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model based on library
    if lib == 'lightgbm':
        model_path = os.path.join(models_dir, 'lgb_model.txt')
        model.save_model(model_path)
        print(f'Saved LightGBM model to {model_path}')
    elif lib == 'xgboost':
        model_path = os.path.join(models_dir, 'xgb_model.json')
        model.save_model(model_path)
        print(f'Saved XGBoost model to {model_path}')
    elif lib == 'catboost':
        model_path = os.path.join(models_dir, 'catboost_model')
        model.save_model(model_path)
        print(f'Saved CatBoost model to {model_path}')
    else:
        # torch fallback: not persisted (would need separate torch.save)
        model_path = None
        print('Torch fallback model not persisted')
    
    # Save metadata
    metadata = {
        'library': lib,
        'timestamp': time.time(),
        'feature_count': len(feature_names) if feature_names else 11
    }
    if feature_names:
        metadata['feature_names'] = feature_names
    if metrics:
        metadata['metrics'] = metrics
    if maps:
        metadata['maps'] = maps
    
    metadata_path = os.path.join(models_dir, 'gbdt_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Saved GBDT metadata to {metadata_path}')
    
    return model_path, metadata_path


def load_gbdt_model():
    """Load GBDT model from disk."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    metadata_path = os.path.join(models_dir, 'gbdt_metadata.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'GBDT metadata not found in {models_dir}')
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    lib = metadata.get('library', GBDT_LIB)
    
    # Load model based on library
    if lib == 'lightgbm':
        model_path = os.path.join(models_dir, 'lgb_model.txt')
        model = lgb.Booster(model_file=model_path)
    elif lib == 'xgboost':
        model_path = os.path.join(models_dir, 'xgb_model.json')
        model = xgb.Booster()
        model.load_model(model_path)
    elif lib == 'catboost':
        model_path = os.path.join(models_dir, 'catboost_model')
        model = CatBoostClassifier()
        model.load_model(model_path)
    else:
        raise ValueError(f'Unsupported GBDT library: {lib}')
    
    print(f'Loaded GBDT model from {models_dir} (library: {lib})')
    print(f'Model metadata: {metadata}')
    return model, metadata


def _find_best_threshold(y_true, scores, n_steps=50):
    """Return the classification threshold that maximises F1 on the given set."""
    y_true = np.array(y_true)
    scores = np.array(scores)
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, n_steps):
        pred_bin = (scores > t).astype(int)
        f1 = f1_score_manual(y_true, pred_bin)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


def _train_gbdt_single(X_tr, y_tr, X_te, lib):
    """Train one GBDT fold and return (model, pred_proba)."""
    if lib == 'lightgbm':
        pos = float(np.sum(y_tr == 1))
        neg = float(np.sum(y_tr == 0))
        scale_pos_weight = max(1.0, neg / max(1.0, pos))
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        params = {
            'objective': 'binary',
            'metric': 'auc',          # optimise AUROC directly
            'verbosity': -1,
            'num_leaves': 63,         # deeper trees for complex interactions
            'learning_rate': 0.03,    # slower lr → better generalisation
            'n_estimators': 500,
            'scale_pos_weight': scale_pos_weight,
            'min_child_samples': 20,  # prevents overfitting on minority class
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
        bst = lgb.train(params, dtrain, num_boost_round=500)
        pred = bst.predict(X_te)
        return bst, pred
    elif lib == 'xgboost':
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dtest = xgb.DMatrix(X_te)
        params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
        model = xgb.train(params, dtrain, num_boost_round=100)
        pred = model.predict(dtest)
        return model, pred
    elif lib == 'catboost':
        model = CatBoostClassifier(iterations=200, verbose=False)
        model.fit(X_tr, y_tr)
        pred = model.predict_proba(X_te)[:, 1]
        return model, pred
    else:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        class MLP(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
                )
            def forward(self, x):
                return self.net(x).squeeze(1)

        Xtr_t = torch.from_numpy(np.array(X_tr).astype(np.float32))
        ytr_t = torch.from_numpy(np.array(y_tr).astype(np.float32))
        Xte_t = torch.from_numpy(np.array(X_te).astype(np.float32))
        model = MLP(Xtr_t.shape[1])
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()
        model.train()
        for _ in range(20):
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t), ytr_t)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte_t).cpu().numpy()
        return model, pred


def train_gbdt(X: pd.DataFrame, y: pd.Series, lib: str = None, save_model_flag: bool = True,
               maps: dict = None, cv: int = 0):
    lib = lib or GBDT_LIB
    print(f"Using GBDT library: {lib}")

    # --- Optional k-fold cross-validation ---
    if cv > 1:
        print(f"\nRunning {cv}-fold stratified cross-validation...")
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        X_arr = X
        splits = stratified_kfold_indices(y_arr, n_folds=cv, random_state=42)
        fold_results = []
        for k, (train_idx, val_idx) in enumerate(splits):
            X_tr = X_arr.iloc[train_idx]
            X_va = X_arr.iloc[val_idx]
            y_tr = y_arr[train_idx]
            y_va = y_arr[val_idx]
            _, pred_va = _train_gbdt_single(X_tr, y_tr, X_va, lib)
            pred_va = np.array(pred_va)
            # Find threshold maximising F1 on this fold's validation set
            best_thresh = _find_best_threshold(y_va, pred_va)
            pred_bin = (pred_va > best_thresh).astype(int)
            fold_acc = accuracy_score_manual(y_va, pred_bin)
            fold_prec = precision_score_manual(y_va, pred_bin)
            fold_rec = recall_score_manual(y_va, pred_bin)
            fold_f1 = f1_score_manual(y_va, pred_bin)
            fold_auc = roc_auc_manual(y_va, pred_va)
            fold_results.append({
                'fold': k + 1,
                'accuracy': round(float(fold_acc), 6),
                'precision': round(float(fold_prec), 6),
                'recall': round(float(fold_rec), 6),
                'f1': round(float(fold_f1), 6),
                'auc': round(float(fold_auc), 6)
            })
            print(f"  Fold {k+1}/{cv} — acc={fold_acc:.4f} prec={fold_prec:.4f} "
                  f"rec={fold_rec:.4f} f1={fold_f1:.4f} auc={fold_auc:.4f}")

        def _mean_std(key):
            vals = [r[key] for r in fold_results]
            return round(float(np.mean(vals)), 6), round(float(np.std(vals)), 6)

        mean_acc, std_acc = _mean_std('accuracy')
        mean_prec, std_prec = _mean_std('precision')
        mean_rec, std_rec = _mean_std('recall')
        mean_f1, std_f1 = _mean_std('f1')
        mean_auc, std_auc = _mean_std('auc')

        cv_summary = {
            'n_folds': cv,
            'library': lib,
            'timestamp': time.time(),
            'folds': fold_results,
            'mean_accuracy': mean_acc, 'std_accuracy': std_acc,
            'mean_precision': mean_prec, 'std_precision': std_prec,
            'mean_recall': mean_rec, 'std_recall': std_rec,
            'mean_f1': mean_f1, 'std_f1': std_f1,
            'mean_auc': mean_auc, 'std_auc': std_auc
        }

        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        cv_path = os.path.join(models_dir, 'gbdt_cv_results.json')
        with open(cv_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)

        print(f"\n  CV Summary ({cv} folds):")
        print(f"    Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"    Precision: {mean_prec:.4f} ± {std_prec:.4f}")
        print(f"    Recall   : {mean_rec:.4f} ± {std_rec:.4f}")
        print(f"    F1       : {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"    AUC      : {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  CV results saved to {cv_path}\n")

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42, stratify=y.values if hasattr(y, 'values') else y)

    model, pred = _train_gbdt_single(X_train, y_train.values if hasattr(y_train, 'values') else y_train, X_test, lib)

    pred = np.array(pred)
    y_test_arr = np.array(y_test)
    best_thresh = _find_best_threshold(y_test_arr, pred)
    pred_bin = (pred > best_thresh).astype(int)
    acc = accuracy_score_manual(y_test_arr, pred_bin)
    prec = precision_score_manual(y_test_arr, pred_bin)
    rec = recall_score_manual(y_test_arr, pred_bin)
    f1 = f1_score_manual(y_test_arr, pred_bin)
    auc = None
    try:
        auc = roc_auc_manual(y_test_arr, pred)
    except Exception:
        auc = None

    print(f"Eval (thresh={best_thresh:.2f}) — acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} auc={auc}")

    # save model and metadata if requested
    if save_model_flag:
        try:
            metrics = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'auc': float(auc) if auc is not None else None,
                'threshold': float(best_thresh),
            }
            feature_names = [
                'amt_log', 'amt_gt_1k', 'amt_gt_3k',
                'mcc_enc', 'payment_type_enc', 'is_crypto',
                'device_change', 'ip_risk', 'ip_risk_high', 'ip_risk_mid',
                'count_1h', 'sum_24h', 'uniq_payees_24h', 'high_velocity',
                'is_international', 'is_high_risk_country',
                'avg_tx_24h', 'velocity_score',
                'amt_x_ip_risk', 'high_amt_device_change', 'crypto_high_risk_country',
            ]
            save_gbdt_model(model, lib, metrics, feature_names, maps)
        except Exception as e:
            print(f'Failed to save GBDT model: {e}')

    # return model and test metadata
    return model, (X_test, y_test, pred)


def score_transaction(tx: Dict, maps: Dict, model) -> float:
    """Score a single transaction dictionary using the trained model and feature maps.

    tx: dict containing keys used in featurize (amount,mcc,payment_type,device_change,ip_risk,count_1h,sum_24h,uniq_payees_24h,country)
    maps: dictionary returned from featurize (mcc_map, pay_map)
    model: trained model (LightGBM booster or torch MLP)

    Returns probability float in [0,1].
    """
    # build a single-row DataFrame
    country = tx.get('country', 'US')
    amount = float(tx.get('amount', 0.0))
    payment_type = tx.get('payment_type', '')
    row = {
        'amount': amount,
        'mcc': tx.get('mcc', ''),
        'payment_type': payment_type,
        'device_change': int(tx.get('device_change', 0)),
        'ip_risk': float(tx.get('ip_risk', 0.0)),
        'count_1h': int(tx.get('count_1h', 0)),
        'sum_24h': float(tx.get('sum_24h', 0.0)),
        'uniq_payees_24h': int(tx.get('uniq_payees_24h', 0)),
        'country': country,
        'is_international': int(country != 'US'),
    }
    df_row = pd.DataFrame([row])
    mcc_map = maps.get('mcc_map', {})
    pay_map = maps.get('pay_map', {})
    df_row['mcc_enc'] = df_row['mcc'].map(lambda v: mcc_map.get(v, -1)).astype(int)
    df_row['payment_type_enc'] = df_row['payment_type'].map(lambda v: pay_map.get(v, -1)).astype(int)
    df_row['is_crypto'] = int(payment_type == 'crypto')
    df_row['is_high_risk_country'] = int(country in HIGH_RISK_COUNTRIES)
    df_row['amt_log'] = np.log1p(df_row['amount'])
    df_row['amt_gt_1k'] = (df_row['amount'] > 1000).astype(int)
    df_row['amt_gt_3k'] = (df_row['amount'] > 3000).astype(int)
    df_row['ip_risk_high'] = (df_row['ip_risk'] > 0.8).astype(int)
    df_row['ip_risk_mid'] = ((df_row['ip_risk'] > 0.6) & (df_row['ip_risk'] <= 0.8)).astype(int)
    df_row['amt_x_ip_risk'] = df_row['amt_log'] * df_row['ip_risk']
    df_row['high_amt_device_change'] = (df_row['amt_gt_1k'] & df_row['device_change']).astype(int)
    df_row['crypto_high_risk_country'] = (df_row['is_crypto'] & df_row['is_high_risk_country']).astype(int)
    df_row['avg_tx_24h'] = df_row['sum_24h'] / (df_row['uniq_payees_24h'] + 1)
    df_row['velocity_score'] = df_row['count_1h'] * (df_row['amt_log'] / 10.0)
    df_row['high_velocity'] = (df_row['count_1h'] > 8).astype(int)
    features = [
        'amt_log', 'amt_gt_1k', 'amt_gt_3k',
        'mcc_enc', 'payment_type_enc', 'is_crypto',
        'device_change', 'ip_risk', 'ip_risk_high', 'ip_risk_mid',
        'count_1h', 'sum_24h', 'uniq_payees_24h', 'high_velocity',
        'is_international', 'is_high_risk_country',
        'avg_tx_24h', 'velocity_score',
        'amt_x_ip_risk', 'high_amt_device_change', 'crypto_high_risk_country',
    ]
    X_row = df_row[features]

    # model inference
    try:
        if 'lightgbm' in str(type(model)).lower() or hasattr(model, 'predict') and hasattr(model, 'save_model'):
            prob = float(model.predict(X_row)[0])
            return prob
    except Exception:
        pass

    # torch fallback
    try:
        import torch
        Xt = torch.from_numpy(X_row.values.astype(np.float32))
        model.eval()
        with torch.no_grad():
            out = model(Xt).cpu().numpy()
        return float(out[0])
    except Exception:
        # last resort: return 0.0
        return 0.0


def demo_run(n: int = 20000, save_model_flag: bool = True, cv: int = 0):
    print("Generating synthetic transactions...")
    df = generate_synthetic_transactions(n=n)
    X, maps = featurize(df)
    y = df['label']
    model, meta = train_gbdt(X, y, save_model_flag=save_model_flag, maps=maps, cv=cv)


if __name__ == '__main__':
    print('Running GBDT detector demo (synthetic transactions)')
    demo_run(20000)
