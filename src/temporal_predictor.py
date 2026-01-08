"""Temporal Prediction Subsystem for AML Detection.

This module provides temporal and predictive capabilities to forecast:
1. Transaction volume anomalies
2. Behavioral shifts and account risk escalation
3. Future suspicious pattern emergence
4. Temporal clustering of suspicious activities

Complements the spatial/detective system with forward-looking analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class TemporalPredictor:
    """Predicts future anomalies and behavioral shifts based on temporal transaction patterns."""

    def __init__(self, lookback_days: int = 30, forecast_days: int = 7):
        """
        Args:
            lookback_days: Historical window for baseline establishment
            forecast_days: Forward prediction horizon
        """
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.baselines = {}  # Store baseline patterns per account
        self.volatility = {}  # Store volatility metrics

    def establish_baselines(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Establish baseline transaction patterns for each account.
        
        Args:
            df: DataFrame with columns: source, target, amount, timestamp, channel (optional)
            
        Returns:
            Dictionary of baseline metrics per account
        """
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        baselines = {}
        
        for account in pd.concat([df['source'], df['target']]).unique():
            # Outgoing transactions
            out_txs = df[df['source'] == account]
            # Incoming transactions
            in_txs = df[df['target'] == account]
            
            baselines[account] = {
                'avg_out_amount': float(out_txs['amount'].mean()) if len(out_txs) > 0 else 0,
                'median_out_amount': float(out_txs['amount'].median()) if len(out_txs) > 0 else 0,
                'std_out_amount': float(out_txs['amount'].std()) if len(out_txs) > 0 else 0,
                'avg_out_frequency': len(out_txs) / max(1, (out_txs['timestamp'].max() - out_txs['timestamp'].min()).days + 1) if len(out_txs) > 0 else 0,
                
                'avg_in_amount': float(in_txs['amount'].mean()) if len(in_txs) > 0 else 0,
                'median_in_amount': float(in_txs['amount'].median()) if len(in_txs) > 0 else 0,
                'std_in_amount': float(in_txs['amount'].std()) if len(in_txs) > 0 else 0,
                'avg_in_frequency': len(in_txs) / max(1, (in_txs['timestamp'].max() - in_txs['timestamp'].min()).days + 1) if len(in_txs) > 0 else 0,
                
                'unique_out_counterparties': len(out_txs['target'].unique()) if len(out_txs) > 0 else 0,
                'unique_in_counterparties': len(in_txs['source'].unique()) if len(in_txs) > 0 else 0,
                'total_out_volume': float(out_txs['amount'].sum()) if len(out_txs) > 0 else 0,
                'total_in_volume': float(in_txs['amount'].sum()) if len(in_txs) > 0 else 0,
            }
        
        self.baselines = baselines
        return baselines

    def detect_volume_acceleration(self, df: pd.DataFrame, 
                                   threshold_sigma: float = 2.5) -> List[Dict[str, Any]]:
        """Detect accounts with accelerating transaction volumes (precursor to structuring).
        
        Args:
            df: Transaction DataFrame
            threshold_sigma: Number of standard deviations to flag as anomaly
            
        Returns:
            List of flags for accounts showing volume acceleration
        """
        if not self.baselines:
            self.establish_baselines(df)
        
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        flags = []
        
        for account in pd.concat([df['source'], df['target']]).unique():
            # Get recent transactions
            recent_txs = df[df['source'] == account]
            if len(recent_txs) < 3:
                continue
            
            # Sort by timestamp
            recent_txs = recent_txs.sort_values('timestamp')
            
            # Compute rolling volumes
            daily_volumes = recent_txs.groupby(recent_txs['timestamp'].dt.date)['amount'].sum()
            
            if len(daily_volumes) < 3:
                continue
            
            volumes = daily_volumes.values
            
            # Check for acceleration trend
            daily_change = np.diff(volumes)
            avg_change = np.mean(daily_change)
            std_change = np.std(daily_change)
            
            if std_change > 0 and avg_change > 1.5 * std_change:
                accel_rate = avg_change / (np.mean(volumes) + 1)
                
                if accel_rate > 0.3:  # 30% acceleration
                    flags.append({
                        'account': account,
                        'type': 'volume_acceleration',
                        'current_daily_volume': float(volumes[-1]),
                        'baseline_daily_volume': float(self.baselines.get(account, {}).get('avg_out_frequency', 0)),
                        'acceleration_rate': float(accel_rate),
                        'score': min(100, accel_rate * 100),
                        'reason': f"Daily transaction volume accelerating at {accel_rate:.2%} rate. Recent: {volumes[-1]:.0f} vs baseline: {np.mean(volumes[:-1]):.0f}"
                    })
        
        return flags

    def detect_behavioral_shift(self, df: pd.DataFrame, 
                               deviation_threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect statistically significant shifts in account transaction behavior.
        
        Predicts accounts likely to engage in suspicious patterns by detecting
        behavioral deviations from established norms.
        
        Args:
            df: Transaction DataFrame
            deviation_threshold: Zscore threshold for anomaly
            
        Returns:
            List of behavioral shift detections
        """
        if not self.baselines:
            self.establish_baselines(df)
        
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        flags = []
        
        for account in self.baselines.keys():
            baseline = self.baselines[account]
            
            # Recent outgoing transactions
            recent_out = df[df['source'] == account]
            if len(recent_out) < 2:
                continue
            
            current_avg = float(recent_out['amount'].mean())
            current_freq = len(recent_out)
            current_unique = len(recent_out['target'].unique())
            
            baseline_avg = baseline['avg_out_amount']
            baseline_freq = baseline['avg_out_frequency']
            baseline_unique = baseline['unique_out_counterparties']
            baseline_std = baseline['std_out_amount'] + 1e-6
            
            # Amount deviation
            amount_zscore = abs(current_avg - baseline_avg) / max(baseline_std, 1)
            
            # Frequency deviation (accounts doing more transactions)
            freq_deviation = (current_freq - baseline_freq) / max(baseline_freq, 1)
            
            # Counterparty diversity change
            diversity_change = (current_unique - baseline_unique) / max(baseline_unique, 1)
            
            # Composite score
            shift_score = amount_zscore + abs(freq_deviation) + abs(diversity_change)
            
            if shift_score > deviation_threshold:
                flags.append({
                    'account': account,
                    'type': 'behavioral_shift',
                    'metric_changes': {
                        'avg_amount_zscore': float(amount_zscore),
                        'frequency_change_pct': float(freq_deviation * 100),
                        'counterparty_diversity_change_pct': float(diversity_change * 100)
                    },
                    'current_metrics': {
                        'avg_transaction': float(current_avg),
                        'recent_tx_count': int(current_freq),
                        'unique_targets': int(current_unique)
                    },
                    'baseline_metrics': {
                        'avg_transaction': float(baseline_avg),
                        'avg_daily_frequency': float(baseline_freq),
                        'avg_unique_targets': int(baseline_unique)
                    },
                    'score': min(100, shift_score * 15),
                    'reason': f"Behavior shifted: Amount zscore={amount_zscore:.2f}, Frequency +{freq_deviation:.0%}, Diversity +{diversity_change:.0%}"
                })
        
        return flags

    def forecast_risk_escalation(self, df: pd.DataFrame,
                                 early_warning_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Forecast which accounts are likely to escalate into higher-risk activities.
        
        Uses temporal patterns to predict future suspicious behavior (e.g., 
        structuring, cycling, complex patterns).
        
        Args:
            df: Transaction DataFrame
            early_warning_threshold: Probability threshold for risk prediction
            
        Returns:
            List of risk escalation predictions
        """
        if not self.baselines:
            self.establish_baselines(df)
        
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        predictions = []
        
        # Risk signals to aggregate
        for account in self.baselines.keys():
            baseline = self.baselines[account]
            recent_txs = df[df['source'] == account].sort_values('timestamp', ascending=False).head(20)
            
            if len(recent_txs) < 3:
                continue
            
            risk_signals = []
            
            # Signal 1: Just-Below-Threshold Pattern (structuring precursor)
            amounts = recent_txs['amount'].values
            if np.all(amounts < 10000) and np.all(amounts > 5000):
                structuring_risk = 0.7
                risk_signals.append(('structuring_precursor', structuring_risk))
            
            # Signal 2: Multiple Small Rapid Transfers (potential fan-in/cycling)
            if len(recent_txs) >= 5:
                is_small = amounts < baseline['median_out_amount'] * 1.2
                if np.sum(is_small) >= 3:
                    small_tx_risk = min(0.8, len(recent_txs) / 10)
                    risk_signals.append(('rapid_small_transfers', small_tx_risk))
            
            # Signal 3: Expanding Counterparty Network
            recent_targets = len(recent_txs['target'].unique())
            baseline_targets = baseline['unique_out_counterparties']
            if baseline_targets > 0 and recent_targets > baseline_targets * 1.5:
                network_expansion_risk = min(0.65, (recent_targets - baseline_targets) / max(baseline_targets, 1) * 0.2)
                risk_signals.append(('counterparty_network_expansion', network_expansion_risk))
            
            # Signal 4: Transaction Timing Clustering (suggests orchestration)
            if len(recent_txs) >= 4:
                time_deltas = pd.to_datetime(recent_txs['timestamp']).diff().dt.total_seconds().dropna()
                if len(time_deltas) > 0:
                    clustering_coefficient = 1 - (np.std(time_deltas) / (np.mean(time_deltas) + 1))
                    if clustering_coefficient > 0.5:  # High clustering
                        timing_risk = min(0.7, clustering_coefficient * 0.7)
                        risk_signals.append(('timing_clustering', timing_risk))
            
            # Aggregate risk signals
            if risk_signals:
                signal_names = [s[0] for s in risk_signals]
                risk_scores = np.array([s[1] for s in risk_signals])
                
                # Compute aggregate probability (Bayesian-style)
                aggregate_risk = 1 - np.prod(1 - risk_scores) if len(risk_scores) > 0 else 0
                
                if aggregate_risk >= early_warning_threshold:
                    predictions.append({
                        'account': account,
                        'type': 'risk_escalation',
                        'predicted_risk_probability': float(aggregate_risk),
                        'risk_signals': signal_names,
                        'individual_signal_scores': {name: float(score) for name, score in risk_signals},
                        'score': min(100, aggregate_risk * 100),
                        'forecast_horizon_days': self.forecast_days,
                        'reason': f"Multi-signal risk: {', '.join(signal_names)}. Aggregate probability: {aggregate_risk:.1%}"
                    })
        
        return predictions

    def detect_temporal_concentration(self, df: pd.DataFrame,
                                      min_transactions: int = 4,
                                      time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """Detect abnormal temporal concentration of transactions (burst patterns).
        
        Identifies accounts with suspicious time clustering that may indicate
        coordinated activities.
        
        Args:
            df: Transaction DataFrame
            min_transactions: Minimum transactions to trigger alert
            time_window_hours: Time window for concentration analysis
            
        Returns:
            List of temporal concentration detections
        """
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        flags = []
        
        for account in pd.concat([df['source'], df['target']]).unique():
            txs = df[(df['source'] == account) | (df['target'] == account)].sort_values('timestamp')
            
            if len(txs) < min_transactions:
                continue
            
            # Look for time windows with high concentration
            timestamps = pd.to_datetime(txs['timestamp']).values
            
            for i in range(len(timestamps) - min_transactions + 1):
                window_start = timestamps[i]
                window_end = window_start + np.timedelta64(time_window_hours, 'h')
                
                txs_in_window = np.sum((timestamps >= window_start) & (timestamps <= window_end))
                
                if txs_in_window >= min_transactions:
                    # Calculate concentration metric
                    total_txs = len(timestamps)
                    concentration_pct = txs_in_window / total_txs * 100
                    
                    if concentration_pct > 40:  # Significant burst
                        flags.append({
                            'account': account,
                            'type': 'temporal_concentration',
                            'burst_transactions': int(txs_in_window),
                            'window_hours': time_window_hours,
                            'concentration_pct': float(concentration_pct),
                            'total_transactions': int(total_txs),
                            'burst_window': {
                                'start': str(pd.Timestamp(window_start)),
                                'end': str(pd.Timestamp(window_end))
                            },
                            'score': min(100, concentration_pct * 1.5),
                            'reason': f"{txs_in_window} txs in {time_window_hours}h window ({concentration_pct:.0f}% of total activity)"
                        })
                        break  # Report first burst for account
        
        return flags

    def predict_cycle_emergence(self, df: pd.DataFrame,
                               min_chain_length: int = 3) -> List[Dict[str, Any]]:
        """Predict accounts likely to form cycles based on transaction patterns.
        
        Early warning for potential round-tripping or circular money flows.
        
        Args:
            df: Transaction DataFrame
            min_chain_length: Minimum chain for escalation prediction
            
        Returns:
            List of cycle emergence predictions
        """
        df = df.copy()
        
        predictions = []
        
        # Build transaction graph
        accounts = pd.concat([df['source'], df['target']]).unique()
        
        for account in accounts:
            # Outgoing connections
            outgoing = set(df[df['source'] == account]['target'].unique())
            # Incoming connections
            incoming = set(df[df['target'] == account]['source'].unique())
            
            # Cross-over accounts (those that appear in both directions)
            bidirectional = incoming & outgoing
            
            if len(bidirectional) >= min_chain_length - 1:
                cycle_risk_score = min(1.0, len(bidirectional) / 10)
                
                predictions.append({
                    'account': account,
                    'type': 'cycle_emergence',
                    'predicted_cycle_probability': float(cycle_risk_score),
                    'bidirectional_counterparties': int(len(bidirectional)),
                    'incoming_connections': int(len(incoming)),
                    'outgoing_connections': int(len(outgoing)),
                    'cycle_risk_indicators': {
                        'reciprocal_relationship_count': int(len(bidirectional)),
                        'transaction_balance': float(
                            df[df['source'] == account]['amount'].sum() - 
                            df[df['target'] == account]['amount'].sum()
                        )
                    },
                    'score': min(100, cycle_risk_score * 100),
                    'reason': f"Account has {len(bidirectional)} bidirectional relationships. Cycle emergence risk: {cycle_risk_score:.0%}"
                })
        
        return predictions

    def forecast_account_summary(self, df: pd.DataFrame,
                                include_all_detections: bool = True) -> Dict[str, Any]:
        """Generate comprehensive temporal forecast for all accounts.
        
        Args:
            df: Transaction DataFrame
            include_all_detections: Include all detection types or only highest risk
            
        Returns:
            Comprehensive forecast report
        """
        # Run all predictive detections
        volume_accel = self.detect_volume_acceleration(df)
        behavior_shifts = self.detect_behavioral_shift(df)
        risk_escalations = self.forecast_risk_escalation(df)
        temporal_bursts = self.detect_temporal_concentration(df)
        cycle_predictions = self.predict_cycle_emergence(df)
        
        all_predictions = {
            'volume_acceleration': volume_accel,
            'behavioral_shift': behavior_shifts,
            'risk_escalation': risk_escalations,
            'temporal_concentration': temporal_bursts,
            'cycle_emergence': cycle_predictions
        }
        
        # Aggregate by account
        account_risk_scores = defaultdict(float)
        account_signals = defaultdict(list)
        
        for detection_type, detections in all_predictions.items():
            for detection in detections:
                account = detection['account']
                score = detection['score']
                account_risk_scores[account] = max(account_risk_scores[account], score)
                account_signals[account].append(detection_type)
        
        # Rank accounts by temporal risk
        ranked_accounts = sorted(account_risk_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'forecast_type': 'temporal_predictive',
            'lookback_days': self.lookback_days,
            'forecast_days': self.forecast_days,
            'baseline_statistics': {
                'total_accounts_baselined': len(self.baselines),
                'mean_baseline_transaction': float(np.mean([b['avg_out_amount'] for b in self.baselines.values() if b['avg_out_amount'] > 0]) if self.baselines else 0)
            },
            'detection_summary': {
                'volume_acceleration_alerts': len(volume_accel),
                'behavioral_shift_alerts': len(behavior_shifts),
                'risk_escalation_predictions': len(risk_escalations),
                'temporal_concentration_alerts': len(temporal_bursts),
                'cycle_emergence_predictions': len(cycle_predictions),
                'total_flagged_accounts': len(account_risk_scores)
            },
            'highest_risk_accounts': [
                {
                    'account': account,
                    'temporal_risk_score': float(score),
                    'signal_count': len(account_signals[account]),
                    'signals': list(set(account_signals[account]))
                }
                for account, score in ranked_accounts[:10]
            ],
            'all_predictions': all_predictions if include_all_detections else None
        }


class SequenceAnalyzer:
    """Analyzes transaction sequences to predict multi-step suspicious patterns."""

    def __init__(self):
        self.patterns = {}
        self.pattern_history = defaultdict(list)

    def detect_structuring_sequence(self, df: pd.DataFrame,
                                   threshold_amount: float = 10000,
                                   just_below_threshold: float = 9000,
                                   time_window_days: int = 7) -> List[Dict[str, Any]]:
        """Detect structuring patterns (breaking large amounts into smaller ones).
        
        Args:
            df: Transaction DataFrame
            threshold_amount: Regulatory threshold (e.g., 10k for CTR)
            just_below_threshold: Amount range below threshold
            time_window_days: Time window for pattern grouping
            
        Returns:
            List of structuring detections
        """
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        flags = []
        
        for account in df['source'].unique():
            txs = df[df['source'] == account].sort_values('timestamp')
            
            # Find sequences of just-below-threshold transactions
            within_range = txs[(txs['amount'] >= just_below_threshold) & 
                              (txs['amount'] < threshold_amount)]
            
            if len(within_range) < 3:
                continue
            
            # Check time clustering
            time_span = (within_range['timestamp'].max() - within_range['timestamp'].min()).days
            
            if time_span <= time_window_days and len(within_range) >= 3:
                total_structured = within_range['amount'].sum()
                structuring_sequence_score = min(100, (len(within_range) - 2) * 20)
                
                flags.append({
                    'account': account,
                    'type': 'structuring_sequence',
                    'transaction_count': len(within_range),
                    'time_window_days': int(time_span),
                    'total_structured_amount': float(total_structured),
                    'avg_transaction_amount': float(within_range['amount'].mean()),
                    'amount_range': f"{just_below_threshold}-{threshold_amount}",
                    'score': float(structuring_sequence_score),
                    'reason': f"Detected {len(within_range)} transactions in amount range {just_below_threshold}-{threshold_amount} within {time_span} days"
                })
        
        return flags
