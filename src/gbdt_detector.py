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


def generate_synthetic_transactions(n: int = 20000, fraud_rate: float = 0.03, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    mcc_pool = ['5311', '5411', '6011', '7995', '4829', '4814', '5732', '7011']  # Added simulator MCCs
    countries = ['US', 'GB', 'NG', 'IN', 'CN', 'UK', 'RU', 'PK'] # Added simulator countries
    payment_types = ['card', 'wire', 'ach', 'crypto']

    # simulate per-account recent activity counters for velocity
    account_last = {}

    for i in range(n):
        acct = f'A_{random.randint(0, n//10)}'
        t = int(time.time()) - random.randint(0, 86400 * 30)
        amount = float(np.round(np.random.exponential(80.0), 2))
        mcc = random.choice(mcc_pool)
        country = random.choice(countries)
        device_id = f'dev_{random.randint(1,5000)}'
        device_change = random.random() < 0.02
        payment_type = random.choice(payment_types)
        ip_risk = random.random()  # 0-1

        # base fraud label low
        is_fraud = random.random() < fraud_rate

        # inject fraud patterns
        if random.random() < 0.002:
            # occasional high-risk event pattern: high amount, new device, high IP risk
            amount = random.uniform(2000, 50000)
            device_id = f'dev_{random.randint(5001,10000)}'
            device_change = True
            ip_risk = random.uniform(0.7, 1.0)
            is_fraud = True
            # Simulate high risk attributes from simulator
            if random.random() < 0.5:
                payment_type = 'crypto'
                country = random.choice(['RU', 'NG', 'PK'])

        # velocity features
        last = account_last.get(acct, {'count_1h': 0, 'sum_24h': 0.0, 'unique_payees_24h': set()})
        count_1h = last['count_1h'] + random.randint(0, 2)
        sum_24h = last['sum_24h'] + amount
        uniq_payees = len(last['unique_payees_24h']) + random.randint(0, 1)

        account_last[acct] = {'count_1h': count_1h, 'sum_24h': sum_24h, 'unique_payees_24h': last['unique_payees_24h']}

        # target label additional heuristics
        if (amount > 10000 and (device_change or ip_risk > 0.6)) or (payment_type == 'crypto' and amount > 5000):
            is_fraud = True

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

    df = pd.DataFrame(rows)
    # introduce some more realistic correlations
    # accounts with device_change more often have higher fraud
    mask_idx = df['device_change'] == 1
    bump = (np.random.rand(mask_idx.sum()) < 0.05).astype(int)
    df.loc[mask_idx, 'label'] = (df.loc[mask_idx, 'label'].astype(int).values | bump).astype(int)

    return df


def featurize(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df2 = df.copy()
    # simple categorical encodings
    mcc_map = {m: i for i, m in enumerate(df2['mcc'].unique())}
    df2['mcc_enc'] = df2['mcc'].map(mcc_map)
    pay_map = {p: i for i, p in enumerate(df2['payment_type'].unique())}
    df2['payment_type_enc'] = df2['payment_type'].map(pay_map)

    # ratios and normalized features
    df2['amt_log'] = np.log1p(df2['amount'])
    df2['avg_tx_24h'] = df2['sum_24h'] / (df2['uniq_payees_24h'] + 1)
    df2['velocity_score'] = df2['count_1h'] * (df2['amt_log'] / 10.0)

    features = ['amt_log', 'mcc_enc', 'payment_type_enc', 'device_change', 'ip_risk', 'count_1h', 'sum_24h', 'uniq_payees_24h', 'is_international', 'avg_tx_24h', 'velocity_score']
    return df2[features], {'mcc_map': mcc_map, 'pay_map': pay_map}


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


def train_gbdt(X: pd.DataFrame, y: pd.Series, lib: str = None, save_model_flag: bool = True, maps: dict = None):
    lib = lib or GBDT_LIB
    print(f"Using GBDT library: {lib}")
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42, stratify=y.values if hasattr(y, 'values') else y)

    if lib == 'lightgbm':
        # compute class weight (scale_pos_weight)
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        scale_pos_weight = max(1.0, neg / max(1.0, pos))
        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_test, label=y_test)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'scale_pos_weight': scale_pos_weight
        }
        # train with fixed rounds; early stopping via callbacks could be added if desired
        bst = lgb.train(params, dtrain, num_boost_round=200, valid_sets=[dtest])
        pred = bst.predict(X_test)
        model = bst
    elif lib == 'xgboost':
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
        model = xgb.train(params, dtrain, num_boost_round=100)
        pred = model.predict(dtest)
    elif lib == 'catboost':
        model = CatBoostClassifier(iterations=200, verbose=False)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]
    else:
        # torch fallback: train a small MLP when no GBDT libs are available
        import torch
        import torch.nn as nn
        import torch.optim as optim

        class MLP(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

            def forward(self, x):
                return self.net(x).squeeze(1)

        Xtr = torch.from_numpy(X_train.values.astype(np.float32))
        ytr = torch.from_numpy(y_train.values.astype(np.float32))
        Xte = torch.from_numpy(X_test.values.astype(np.float32))

        model = MLP(Xtr.shape[1])
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()
        model.train()
        for epoch in range(20):
            opt.zero_grad()
            out = model(Xtr)
            loss = loss_fn(out, ytr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte).cpu().numpy()

    pred = np.array(pred)
    y_test_arr = np.array(y_test)
    pred_bin = (pred > 0.5).astype(int)
    acc = accuracy_score_manual(y_test_arr, pred_bin)
    prec = precision_score_manual(y_test_arr, pred_bin)
    rec = recall_score_manual(y_test_arr, pred_bin)
    auc = None
    try:
        auc = roc_auc_manual(y_test_arr, pred)
    except Exception:
        auc = None

    print(f"Eval — acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} auc={auc}")

    # save model and metadata if requested
    if save_model_flag:
        try:
            metrics = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'auc': float(auc) if auc is not None else None
            }
            feature_names = ['amt_log', 'mcc_enc', 'payment_type_enc', 'device_change', 'ip_risk', 'count_1h', 'sum_24h', 'uniq_payees_24h', 'is_international', 'avg_tx_24h', 'velocity_score']
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
    row = {
        'amount': tx.get('amount', 0.0),
        'mcc': tx.get('mcc', ''),
        'payment_type': tx.get('payment_type', ''),
        'device_change': int(tx.get('device_change', 0)),
        'ip_risk': float(tx.get('ip_risk', 0.0)),
        'count_1h': int(tx.get('count_1h', 0)),
        'sum_24h': float(tx.get('sum_24h', 0.0)),
        'uniq_payees_24h': int(tx.get('uniq_payees_24h', 0)),
        'country': tx.get('country', 'US'),
        'is_international': int(tx.get('country', 'US') != 'US')
    }
    df_row = pd.DataFrame([row])
    # encode
    mcc_map = maps.get('mcc_map', {})
    pay_map = maps.get('pay_map', {})
    df_row['mcc_enc'] = df_row['mcc'].map(lambda v: mcc_map.get(v, -1)).astype(int)
    df_row['payment_type_enc'] = df_row['payment_type'].map(lambda v: pay_map.get(v, -1)).astype(int)
    df_row['amt_log'] = np.log1p(df_row['amount'])
    df_row['avg_tx_24h'] = df_row['sum_24h'] / (df_row['uniq_payees_24h'] + 1)
    df_row['velocity_score'] = df_row['count_1h'] * (df_row['amt_log'] / 10.0)
    features = ['amt_log', 'mcc_enc', 'payment_type_enc', 'device_change', 'ip_risk', 'count_1h', 'sum_24h', 'uniq_payees_24h', 'is_international', 'avg_tx_24h', 'velocity_score']
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


def demo_run(n: int = 20000, save_model_flag: bool = True):
    print("Generating synthetic transactions...")
    df = generate_synthetic_transactions(n=n)
    X, maps = featurize(df)
    y = df['label']
    model, meta = train_gbdt(X, y, save_model_flag=save_model_flag, maps=maps)


if __name__ == '__main__':
    print('Running GBDT detector demo (synthetic transactions)')
    demo_run(20000)
