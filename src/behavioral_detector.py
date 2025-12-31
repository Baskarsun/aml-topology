import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class BehavioralDetector:
    """Implements a set of heuristic detectors that fuse cyber behavioral signals with transaction data.

    Contract (input shapes):
      - logins_df: pandas.DataFrame with columns: user_id, timestamp (int/float seconds), success (bool), ip, subnet, asn (optional), user_agent, time_to_login (float seconds), device_id, lat (float), lon (float), new_device (bool)
      - events_df: pandas.DataFrame with columns: user_id, timestamp, event_type, page (optional), amount (optional), channel (optional), target_payee (optional)
      - tx_df: pandas.DataFrame of transactions with columns: source, target, amount, timestamp, channel (optional)
      - fingerprints: simple dicts describing device attributes

    Outputs: lists of flag dicts: {'user_id':..., 'type':..., 'score':..., 'reason':...}
    """

    def __init__(self):
        pass

    # ---- Pre-compromise detectors ----
    def detect_credential_stuffing(self, logins_df: pd.DataFrame, window_seconds: int = 3600,
                                   fail_ratio_threshold: float = 0.8, min_attempts: int = 30) -> List[Dict[str, Any]]:
        """Detect high ratio of failed logins per subnet/ASN inside rolling windows.

        Returns flags keyed by subnet/asn.
        """
        if logins_df is None or len(logins_df) == 0:
            return []

        df = logins_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Prefer ASN if present; otherwise use subnet if present; fallback to ip
        group_col = 'asn' if 'asn' in df.columns else ('subnet' if 'subnet' in df.columns else 'ip')

        flags = []
        for key, g in df.groupby(group_col):
            # sliding window by timestamp: compute counts per window using rolling on integer seconds
            times = g['timestamp'].astype('int64') // 10**9
            times = np.sort(times.values)
            n = len(times)
            if n < min_attempts:
                continue

            # two-pointer sliding window
            i = 0
            while i < n:
                j = i
                while j < n and times[j] - times[i] <= window_seconds:
                    j += 1
                window_size = j - i
                window_slice = g.iloc[i:j]
                fails = (~window_slice['success']).sum() if 'success' in window_slice else 0
                ratio = float(fails) / max(1, window_size)
                if window_size >= min_attempts and ratio >= fail_ratio_threshold:
                    flags.append({'group': key, 'type': 'credential_stuffing', 'count': int(window_size), 'fail_ratio': ratio,
                                  'reason': f"{window_size} attempts with {ratio:.2f} fail ratio in {window_seconds}s"})
                    break
                i += 1

        return flags

    def detect_low_and_slow(self, logins_df: pd.DataFrame, user_agent_consistency_threshold: float = 0.9,
                             fast_login_seconds: float = 1.0) -> List[Dict[str, Any]]:
        """Detect 'low-and-slow' distributed attacks where many attempts share UA or have extremely fast logins.
        Heuristic: if a user or account sees many fast time_to_login entries or highly similar user agents, flag it.
        """
        if logins_df is None or len(logins_df) == 0:
            return []

        df = logins_df.copy()
        flags = []
        for user, g in df.groupby('user_id'):
            if 'time_to_login' in g.columns:
                fast_count = (g['time_to_login'] <= fast_login_seconds).sum()
                if fast_count >= max(3, 0.25 * len(g)):
                    flags.append({'user_id': user, 'type': 'fast_login_cluster', 'count': int(fast_count),
                                  'reason': f"{fast_count}/{len(g)} logins under {fast_login_seconds}s"})

            if 'user_agent' in g.columns:
                ua_counts = g['user_agent'].value_counts(normalize=True)
                top_share = ua_counts.iloc[0] if len(ua_counts) > 0 else 0
                if top_share >= user_agent_consistency_threshold and len(g) >= 5:
                    flags.append({'user_id': user, 'type': 'ua_consistency', 'share': float(top_share),
                                  'reason': f"Top UA accounts for {top_share:.2f} of {len(g)} attempts"})

        return flags

    def detect_bruteforce_and_new_device(self, logins_df: pd.DataFrame, sequence_window: int = 300) -> List[Dict[str, Any]]:
        """Detect patterns of repeated failures followed by success and new device registration.
        sequence_window in seconds.
        """
        if logins_df is None or len(logins_df) == 0:
            return []

        df = logins_df.copy()
        df = df.sort_values(['user_id', 'timestamp'])
        flags = []
        for user, g in df.groupby('user_id'):
            g = g.reset_index(drop=True)
            for i in range(len(g) - 1):
                if ('success' in g.columns and not g.loc[i, 'success']) and ('success' in g.columns and g.loc[i+1, 'success']):
                    t0 = g.loc[i, 'timestamp']
                    t1 = g.loc[i+1, 'timestamp']
                    if abs(t1 - t0) <= sequence_window:
                        new_dev = g.loc[i+1].get('new_device', False)
                        flags.append({'user_id': user, 'type': 'bruteforce_followed_by_success', 'time_span': int(t1 - t0),
                                      'new_device': bool(new_dev), 'reason': 'failure(s) then success within window'})

        return flags

    # ---- Post-compromise detectors (warming) ----
    def detect_post_compromise(self, events_df: pd.DataFrame, sensitive_pages: List[str] = None,
                                tester_amount_threshold: float = 10.0) -> List[Dict[str, Any]]:
        """Detect reconnaissance, profile modifications, payee additions and tester transactions."""
        if events_df is None or len(events_df) == 0:
            return []

        sensitive_pages = sensitive_pages or ['settings', 'profile', 'statements', 'limits']
        flags = []
        for user, g in events_df.groupby('user_id'):
            pages = g[g['event_type'] == 'page_view']['page'].dropna().astype(str).tolist()
            sensitive_hits = [p for p in pages if any(s in p.lower() for s in sensitive_pages)]
            if len(sensitive_hits) >= 2:
                flags.append({'user_id': user, 'type': 'passive_recon', 'count': len(sensitive_hits),
                              'reason': f"Visited sensitive pages: {sensitive_hits[:5]}"})

            # profile modifications
            changes = g[g['event_type'] == 'change_contact']
            if len(changes) > 0:
                flags.append({'user_id': user, 'type': 'profile_modification', 'count': len(changes),
                              'reason': 'Contact info changed'})

            # payee addition + tester transaction
            payees = g[g['event_type'] == 'add_payee']['target_payee'].dropna().unique().tolist()
            if payees:
                # look for small transactions to those payees
                txs = g[g['event_type'] == 'transaction']
                for payee in payees:
                    small = txs[(txs['target_payee'] == payee) & (txs['amount'] <= tester_amount_threshold)]
                    if len(small) > 0:
                        flags.append({'user_id': user, 'type': 'tester_transaction', 'payee': payee, 'count': len(small),
                                      'reason': f"Small transactions to new payee {payee}"})

        return flags

    # ---- Bust-out detectors ----
    def detect_bust_out(self, tx_df: pd.DataFrame, window_seconds: int = 3600, amount_threshold: float = 10000.0,
                        tx_count_threshold: int = 3) -> List[Dict[str, Any]]:
        """Detect rapid high-value drains or many max transfers in short time.
        Returns flags per source account.
        """
        if tx_df is None or len(tx_df) == 0:
            return []

        df = tx_df.copy()
        flags = []
        df = df.sort_values('timestamp')
        for src, g in df.groupby('source'):
            times = np.sort(g['timestamp'].values)
            amounts = g['amount'].values
            n = len(times)
            if n == 0:
                continue
            i = 0
            while i < n:
                j = i
                sum_amount = 0.0
                cnt = 0
                while j < n and times[j] - times[i] <= window_seconds:
                    sum_amount += amounts[j]
                    cnt += 1
                    j += 1
                if cnt >= tx_count_threshold and sum_amount >= amount_threshold:
                    flags.append({'source': src, 'type': 'bust_out', 'window_seconds': window_seconds, 'count': int(cnt),
                                  'sum_amount': float(sum_amount), 'reason': f"{cnt} txs totalling {sum_amount:.2f} in {window_seconds}s"})
                    break
                i += 1

        return flags

    def detect_channel_hopping(self, tx_df: pd.DataFrame, window_seconds: int = 300) -> List[Dict[str, Any]]:
        """Detect channel hopping: multiple channels used in short time for same account.
        Requires a 'channel' column on transactions (e.g., 'web', 'mobile', 'api', 'wire').
        """
        if tx_df is None or 'channel' not in tx_df.columns:
            return []

        df = tx_df.copy()
        flags = []
        for src, g in df.groupby('source'):
            g = g.sort_values('timestamp')
            for i in range(len(g) - 1):
                t0 = g.iloc[i]['timestamp']
                t1 = g.iloc[i+1]['timestamp']
                if abs(t1 - t0) <= window_seconds and g.iloc[i]['channel'] != g.iloc[i+1]['channel']:
                    flags.append({'source': src, 'type': 'channel_hopping', 'channels': (g.iloc[i]['channel'], g.iloc[i+1]['channel']),
                                  'reason': f"Channels {g.iloc[i]['channel']} and {g.iloc[i+1]['channel']} within {window_seconds}s"})
                    break

        return flags

    # ---- Device fingerprint checks ----
    def compare_fingerprints(self, old_fp: Dict[str, Any], new_fp: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two fingerprint dicts and return a score and list of mismatches.
        Detects emulator-like signals such as constant battery=100, missing sensors, zero motion variance.
        """
        if not old_fp or not new_fp:
            return {'score': 0.0, 'mismatches': ['missing_fingerprint']}

        mismatches = []
        keys = set(old_fp.keys()).union(new_fp.keys())
        mismatch_score = 0
        for k in keys:
            v1 = old_fp.get(k)
            v2 = new_fp.get(k)
            if k == 'battery':
                # battery delta should be plausible
                try:
                    if abs(float(v1) - float(v2)) > 40:
                        mismatches.append(f'battery_jump({v1}->{v2})')
                        mismatch_score += 1
                except Exception:
                    pass
            elif k == 'motion_variance':
                if v2 is not None and float(v2) < 1e-3:
                    mismatches.append('no_motion')
                    mismatch_score += 1
            elif k == 'headless' or k == 'webdriver':
                if v2:
                    mismatches.append(k)
                    mismatch_score += 1
            else:
                if v1 != v2:
                    mismatches.append(f'{k}_diff')
                    mismatch_score += 0.2

        # emulator heuristic: battery==100 and no motion
        if new_fp.get('battery') == 100 and new_fp.get('motion_variance', 1.0) < 1e-3:
            mismatches.append('emulator_like')
            mismatch_score += 2

        # normalize score to 0..1
        score = min(1.0, mismatch_score / 5.0)
        return {'score': float(score), 'mismatches': mismatches}

    # ---- Impossible travel ----
    def detect_impossible_travel(self, logins_df: pd.DataFrame, velocity_kmph_threshold: float = 1000.0) -> List[Dict[str, Any]]:
        """Flag login pairs for which computed velocity exceeds threshold. Assumes lat/lon provided per login.
        Returns list of {'user_id','t0','t1','distance_km','velocity_kmph'}.
        """
        if logins_df is None or len(logins_df) == 0:
            return []

        flags = []
        df = logins_df.copy()
        if not {'lat', 'lon', 'timestamp'}.issubset(df.columns):
            return []

        for user, g in df.groupby('user_id'):
            g = g.sort_values('timestamp')
            prev = None
            for _, row in g.iterrows():
                if prev is None:
                    prev = row
                    continue
                dist = haversine_km(prev['lat'], prev['lon'], row['lat'], row['lon'])
                dt_hours = max(1e-6, (row['timestamp'] - prev['timestamp']) / 3600.0)
                vel = dist / dt_hours
                if vel > velocity_kmph_threshold:
                    flags.append({'user_id': user, 't0': int(prev['timestamp']), 't1': int(row['timestamp']),
                                  'distance_km': float(dist), 'velocity_kmph': float(vel),
                                  'reason': f"Impossible travel {dist:.1f}km in {(row['timestamp']-prev['timestamp']):.0f}s ({vel:.0f} km/h)"})
                prev = row

        return flags

    # ---- Behavioral biometrics heuristics ----
    def analyze_biometrics(self, bio: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single session's biometric metrics (mouse, keystroke, copy-paste) and return a risk score and reasons.
        Expected keys: mouse_straightness (0..1, higher is straighter), avg_mouse_speed, flight_time_std, dwell_time_std, copy_paste_count
        """
        score = 0.0
        reasons = []
        if bio.get('mouse_straightness', 0) > 0.85:
            score += 0.4
            reasons.append('mouse_straight')
        if bio.get('avg_mouse_speed', 9999) < 0.01:
            # virtually instant moves
            score += 0.3
            reasons.append('instant_moves')
        if bio.get('flight_time_std', 0) < 1e-3:
            score += 0.2
            reasons.append('keystroke_uniform')
        if bio.get('copy_paste_count', 0) >= 3:
            score += 0.3
            reasons.append('copy_paste')

        score = min(1.0, score)
        return {'score': float(score), 'reasons': reasons}

    # ---- APP fraud detectors ----
    def detect_app_fraud(self, tx_df: pd.DataFrame, events_df: pd.DataFrame = None,
                         user_profiles: Dict[str, Dict[str, Any]] = None,
                         unusual_hour_threshold: int = 2) -> List[Dict[str, Any]]:
        """Detect Authorized Push Payment (APP) indicators.

        Heuristics implemented:
          - payee_mismatch: payment to a beneficiary with no history and to a country where the user has no operations
          - bypass_workflow: events_df contains 'bypass_approval' or 'ignore_warning' entries
          - unusual_time: transaction occurs outside the user's typical active hours (uses user_profiles[user_id]['typical_hours'] if present)
          - active_call: events_df has 'device_call_active' true at transaction time

        tx_df expected columns: source, target, amount, timestamp, target_country (optional), channel (optional)
        events_df expected columns: user_id, timestamp, event_type
        user_profiles: optional map of user_id -> {typical_countries: set(...), typical_hours: (start_hour, end_hour)}
        """
        if tx_df is None or len(tx_df) == 0:
            return []

        flags = []
        user_profiles = user_profiles or {}

        # Build quick historical payee set per source
        history = {}
        for src, g in tx_df.groupby('source'):
            history[src] = set(g['target'].astype(str).unique())

        # index events by user for quick lookup
        events_by_user = {}
        if events_df is not None and len(events_df) > 0:
            for uid, ge in events_df.groupby('user_id'):
                events_by_user[uid] = ge.sort_values('timestamp')

        for _, row in tx_df.iterrows():
            src = row.get('source')
            tgt = row.get('target')
            t = row.get('timestamp')
            tgt_country = row.get('target_country')

            # Payee mismatch: no prior history of paying this target
            if src not in history or str(tgt) not in history.get(src, set()):
                # country mismatch: if profile lists typical countries and target_country is outside
                profile = user_profiles.get(src, {})
                typical_countries = set(profile.get('typical_countries', []))
                if tgt_country and typical_countries and tgt_country not in typical_countries:
                    flags.append({'source': src, 'type': 'app_payee_mismatch', 'target': tgt, 'target_country': tgt_country,
                                  'reason': 'New beneficiary in atypical country'})
                else:
                    flags.append({'source': src, 'type': 'app_new_payee', 'target': tgt,
                                  'reason': 'No prior payment history to beneficiary'})

            # Unusual transaction time
            if src in user_profiles and 'typical_hours' in user_profiles[src]:
                start_h, end_h = user_profiles[src]['typical_hours']
                try:
                    hour = int((int(t) % 86400) // 3600)
                except Exception:
                    hour = None
                if hour is not None:
                    # If typical_hours defines a window, check if tx is outside by more than threshold hours
                    if not (start_h <= hour <= end_h):
                        flags.append({'source': src, 'type': 'app_unusual_hour', 'hour': hour,
                                      'reason': f'Transaction at hour {hour} outside typical {start_h}-{end_h}'})

            # Bypass workflow or active call
            if src in events_by_user:
                ev = events_by_user[src]
                # find any events near timestamp t
                close = ev[(ev['timestamp'] >= t - 300) & (ev['timestamp'] <= t + 300)]
                if len(close) > 0:
                    if any(close['event_type'].astype(str).str.contains('bypass|ignore', case=False)):
                        flags.append({'source': src, 'type': 'app_bypass_workflow', 'reason': 'Approval/workflow bypass around tx time'})
                    if any(close['event_type'] == 'device_call_active'):
                        flags.append({'source': src, 'type': 'app_active_call', 'reason': 'Device was on an active call during transaction'})

        return flags

    # ---- Synthetic identity detectors ----
    def detect_synthetic_identity(self, identity_profiles: Dict[str, Dict[str, Any]],
                                  credit_activity_df: pd.DataFrame = None,
                                  thin_age_threshold: int = 30,
                                  thin_history_years: int = 1,
                                  utilization_spike_pct: float = 0.8) -> List[Dict[str, Any]]:
        """Detect synthetic identity patterns.

        identity_profiles: map user_id -> {age, credit_history_years, addresses:[], phone_numbers:[], authorized_users: []}
        credit_activity_df: optional DataFrame with columns user_id, timestamp, credit_limit, balance

        Heuristics:
          - thin_file: older user with very short credit history
          - multiple_addresses_or_phones: more than 1 address/phone (possible clustering)
          - credit_utilization_spike: utilization jumps above utilization_spike_pct within short window
        """
        flags = []
        for user, prof in (identity_profiles or {}).items():
            age = prof.get('age')
            hist_years = prof.get('credit_history_years', 0)
            if age is not None and age >= thin_age_threshold and hist_years <= thin_history_years:
                flags.append({'user_id': user, 'type': 'synthetic_thin_file', 'age': age, 'history_years': hist_years,
                              'reason': 'Older user with very short credit history'})

            addrs = prof.get('addresses', []) or []
            phones = prof.get('phone_numbers', []) or []
            if len(addrs) > 1 or len(phones) > 1:
                flags.append({'user_id': user, 'type': 'synthetic_multi_addr_phone', 'addresses': len(addrs), 'phones': len(phones),
                              'reason': 'Multiple addresses/phones associated with profile'})

            # authorized users / piggybacking
            auths = prof.get('authorized_users', []) or []
            if len(auths) > 0:
                flags.append({'user_id': user, 'type': 'synthetic_piggyback', 'count_authorized': len(auths),
                              'reason': 'Authorized users present (possible piggybacking)'})

        # Credit utilization spike detection
        if credit_activity_df is not None and len(credit_activity_df) > 0:
            cadf = credit_activity_df.copy()
            cadf = cadf.sort_values(['user_id', 'timestamp'])
            for uid, g in cadf.groupby('user_id'):
                g = g.reset_index(drop=True)
                # compute utilization
                util = (g['balance'] / g['credit_limit']).replace([np.inf, -np.inf], np.nan).fillna(0)
                # detect rapid jump: difference between rolling min and current
                if len(util) < 2:
                    continue
                for i in range(1, len(util)):
                    if util.iloc[i] >= utilization_spike_pct and util.iloc[i-1] < utilization_spike_pct:
                        flags.append({'user_id': uid, 'type': 'synthetic_util_spike', 'timestamp': int(g.loc[i,'timestamp']),
                                      'utilization': float(util.iloc[i]), 'reason': 'Rapid credit utilization spike'})
                        break

        return flags

    # ---- Money mule detectors ----
    def detect_money_mules(self, tx_df: pd.DataFrame, logins_df: pd.DataFrame = None,
                           dormancy_days_threshold: int = 180, tester_amount_threshold: float = 50.0,
                           spike_window_days: int = 7, spike_amount_threshold: float = 5000.0) -> List[Dict[str, Any]]:
        """Detect mule-like behavior: recruited mule, complicit mule, exploited mule.

        tx_df: columns source, target, amount, timestamp
        logins_df: optional for multi-homing detection; columns user_id, device_id, ip
        """
        if tx_df is None or len(tx_df) == 0:
            return []

        flags = []
        now_ts = int(pd.Timestamp.now().timestamp())

        # Build per-account history
        for src, g in tx_df.groupby('source'):
            g = g.sort_values('timestamp')
            first_ts = int(g.iloc[0]['timestamp'])
            last_ts = int(g.iloc[-1]['timestamp'])
            days_since_active = (now_ts - last_ts) / 86400.0

            # Dormancy then tester phase
            if days_since_active > dormancy_days_threshold:
                # look for small tester txs after dormancy (we can't detect future events here easily)
                # instead, if earliest tx is small and account was dormant before, flag
                if g.iloc[0]['amount'] <= tester_amount_threshold:
                    flags.append({'source': src, 'type': 'mule_dormant_tester', 'reason': 'Dormant account with small tester tx'})

            # Spike then dump (quick high incoming then outgoing with low retention)
            # We'll approximate: check incoming funds to account vs outgoing within spike_window_days
            start_window = now_ts - spike_window_days * 86400
            in_txs = tx_df[(tx_df['target'] == src) & (tx_df['timestamp'] >= start_window)]
            out_txs = tx_df[(tx_df['source'] == src) & (tx_df['timestamp'] >= start_window)]
            in_sum = float(in_txs['amount'].sum()) if len(in_txs) > 0 else 0.0
            out_sum = float(out_txs['amount'].sum()) if len(out_txs) > 0 else 0.0
            if in_sum >= spike_amount_threshold and out_sum >= 0.9 * in_sum and in_sum > 0:
                flags.append({'source': src, 'type': 'mule_spike_dump', 'in_sum': in_sum, 'out_sum': out_sum,
                              'reason': 'Large incoming funds followed by near-total outgoing within short window'})

        # Multi-homing / mule herding: same device or ip used across many accounts
        if logins_df is not None and len(logins_df) > 0:
            # device to accounts map
            dev_map = {}
            for _, r in logins_df.iterrows():
                dev = r.get('device_id') or r.get('ip')
                uid = r.get('user_id')
                if dev is None or uid is None:
                    continue
                dev_map.setdefault(dev, set()).add(uid)

            for dev, accounts in dev_map.items():
                if len(accounts) >= 3:
                    flags.append({'device': dev, 'type': 'mule_multi_homing', 'accounts': list(accounts),
                                  'reason': 'Single device/IP used to access multiple accounts'})

        return flags


__all__ = ['BehavioralDetector']
