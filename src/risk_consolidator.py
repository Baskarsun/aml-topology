"""Risk Consolidation: Unified risk scoring across all AML detection phases.

This module combines signals from spatial (graph), behavioral, temporal, and predictive
(LSTM) phases into a single parameterized risk score per account.

Design:
-------
- Each phase contributes signals (detections, scores, alerts).
- Signals are normalized to [0, 1] range.
- Configurable weights control the influence of each phase.
- Final risk score is a weighted sum of normalized phase scores.
- Ranks accounts by consolidated risk for triage.

Parameters:
-----------
weights: dict
    Phase weights: {'spatial': 0.25, 'behavioral': 0.15, 'temporal': 0.35, 'lstm': 0.25}
    Sum of weights should ideally be 1.0 for normalized output.

signal_thresholds: dict
    Control when a signal contributes (e.g., min_lstm_prob=0.5).

Examples:
---------
consolidator = RiskConsolidator(weights={'spatial': 0.3, 'temporal': 0.4, 'lstm': 0.3})
scores = consolidator.consolidate_risks(spatial_dict, behavioral_dict, temporal_dict, lstm_dict)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class RiskConsolidator:
    """Parameterized multi-phase risk consolidation engine."""
    
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 signal_thresholds: Optional[Dict[str, float]] = None,
                 normalize_output: bool = True):
        """
        Args:
            weights: Phase weights (spatial, behavioral, temporal, lstm, cyber).
                    Default: uniform weights across 5 phases.
            signal_thresholds: Min thresholds for signals to contribute
                              (e.g., lstm_prob_min=0.5).
            normalize_output: If True, normalize final scores to [0, 1].
        """
        self.weights = weights or {
            'spatial': 0.20,       # cycles, fan-in/out, centrality
            'behavioral': 0.10,    # cyber alerts (credential stuffing, brute force, impossible travel)
            'temporal': 0.35,      # volume accel, behavioral shift, risk escalation, concentration
            'lstm': 0.25,          # emerging link predictions
            'cyber': 0.10          # cyber behavioral detections
        }
        
        self.signal_thresholds = signal_thresholds or {
            'cycle_detected': 0,           # binary: 0 or 1
            'fan_in_threshold': 5,         # indegree
            'fan_out_threshold': 5,        # outdegree
            'centrality_percentile': 75,   # top 25% by betweenness
            'behavioral_alert': 0,         # binary
            'temporal_concentration': 0.5, # pct threshold
            'lstm_prob_min': 0.5,          # min link prob to count
            'cyber_alert': 0               # binary
        }
        
        self.normalize_output = normalize_output
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate and normalize weights if needed."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"[RiskConsolidator] Warning: weights sum to {total:.3f}, not 1.0. "
                  f"Consider normalizing.")
    
    def consolidate_risks(self,
                         spatial_signals: Dict,
                         behavioral_signals: Dict,
                         temporal_signals: Dict,
                         lstm_signals: Dict,
                         all_nodes: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Consolidate multi-phase risk scores into unified ranking.
        
        Args:
            spatial_signals: dict with keys like 'cycles', 'fan_ins', 'fan_outs', 'centrality'
                            cycles: list of cycle tuples
                            fan_ins: list of dicts with 'node' and 'burst_size'
                            fan_outs: list of dicts with 'node' and 'burst_size'
                            centrality: list of (node, score) tuples
            behavioral_signals: dict with keys like 'cyber_alerts'
                               cyber_alerts: list of alert dicts with 'user_id'
            temporal_signals: dict with keys like 'risk_predictions', 'concentration', 'cycle_pred'
                             risk_predictions: list of dicts with 'account' and 'predicted_risk_probability'
                             concentration: list of dicts with 'account' and 'concentration_pct'
                             cycle_pred: list of dicts with 'account' and 'predicted_cycle_probability'
            lstm_signals: dict with keys like 'emerging_links'
                         emerging_links: list of dicts with 'source', 'target', 'probability'
            all_nodes: optional list of all node IDs to ensure complete coverage
        
        Returns:
            Dict mapping node -> {
                'spatial_score': float,
                'behavioral_score': float,
                'temporal_score': float,
                'lstm_score': float,
                'final_risk_score': float,
                'risk_rank': int,
                'signals': [list of signals that triggered],
                'signal_count': int
            }
        """
        # Initialize scores for all nodes
        all_accounts = set(all_nodes) if all_nodes else set()
        
        # Collect all mentioned accounts from signals
        all_accounts.update(self._extract_accounts(spatial_signals))
        all_accounts.update(self._extract_accounts(behavioral_signals))
        all_accounts.update(self._extract_accounts(temporal_signals))
        all_accounts.update(self._extract_accounts(lstm_signals))
        
        all_accounts = sorted(list(all_accounts))
        
        # Initialize result dict
        results = {node: {
            'spatial_score': 0.0,
            'behavioral_score': 0.0,
            'temporal_score': 0.0,
            'lstm_score': 0.0,
            'final_risk_score': 0.0,
            'signals': [],
            'signal_count': 0
        } for node in all_accounts}
        
        # Phase 1: Spatial Signals
        self._score_spatial(results, spatial_signals)
        
        # Phase 2: Behavioral Signals
        self._score_behavioral(results, behavioral_signals)
        
        # Phase 3: Temporal Signals
        self._score_temporal(results, temporal_signals)
        
        # Phase 4: LSTM Signals
        self._score_lstm(results, lstm_signals)
        
        # Consolidate final scores
        weight_sum = sum(self.weights.values())
        for node in results:
            weighted_sum = (
                results[node]['spatial_score'] * self.weights.get('spatial', 0) +
                results[node]['behavioral_score'] * self.weights.get('behavioral', 0) +
                results[node]['temporal_score'] * self.weights.get('temporal', 0) +
                results[node]['lstm_score'] * self.weights.get('lstm', 0)
            )
            final_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
            
            if self.normalize_output:
                final_score = min(1.0, max(0.0, final_score))
            
            results[node]['final_risk_score'] = final_score
        
        # Rank by final risk score
        sorted_nodes = sorted(all_accounts, 
                            key=lambda n: results[n]['final_risk_score'], 
                            reverse=True)
        
        for rank, node in enumerate(sorted_nodes, 1):
            results[node]['risk_rank'] = rank
        
        return results
    
    def _extract_accounts(self, signals: Dict) -> set:
        """Extract all account mentions from signal dict."""
        accounts = set()
        
        if not signals:
            return accounts
        
        # Cycles
        for cycle in signals.get('cycles', []):
            accounts.update(cycle)
        
        # Fan-in/out
        for item in signals.get('fan_ins', []):
            if isinstance(item, dict) and 'node' in item:
                accounts.add(item['node'])
        for item in signals.get('fan_outs', []):
            if isinstance(item, dict) and 'node' in item:
                accounts.add(item['node'])
        
        # Centrality
        for node, _ in signals.get('centrality', []):
            accounts.add(node)
        
        # Risk predictions
        for pred in signals.get('risk_predictions', []):
            if isinstance(pred, dict) and 'account' in pred:
                accounts.add(pred['account'])
        
        # Concentration
        for item in signals.get('concentration', []):
            if isinstance(item, dict) and 'account' in item:
                accounts.add(item['account'])
        
        # Cycle predictions
        for pred in signals.get('cycle_pred', []):
            if isinstance(pred, dict) and 'account' in pred:
                accounts.add(pred['account'])
        
        # Cyber alerts
        for alert in signals.get('cyber_alerts', []):
            if isinstance(alert, dict) and 'user_id' in alert:
                accounts.add(alert['user_id'])
        
        # Emerging links
        for link in signals.get('emerging_links', []):
            if isinstance(link, tuple) and len(link) >= 2:
                accounts.add(link[0][0])  # (u, v), prob
                accounts.add(link[0][1])
        
        return accounts
    
    def _score_spatial(self, results: Dict, signals: Dict):
        """Score spatial (graph) signals: cycles, fan-in/out, centrality."""
        if not signals:
            return
        
        # Cycles: each node in a cycle gets +0.3
        for cycle in signals.get('cycles', []):
            for node in cycle:
                if node in results:
                    results[node]['spatial_score'] += 0.3
                    if 'cycle' not in results[node]['signals']:
                        results[node]['signals'].append('cycle_detection')
        
        # Fan-in (structuring): +0.25 per incident
        for item in signals.get('fan_ins', []):
            if isinstance(item, dict) and 'node' in item:
                node = item['node']
                if node in results:
                    results[node]['spatial_score'] += 0.25
                    if 'fan_in' not in results[node]['signals']:
                        results[node]['signals'].append('fan_in_detected')
        
        # Fan-out (integration): +0.25 per incident
        for item in signals.get('fan_outs', []):
            if isinstance(item, dict) and 'node' in item:
                node = item['node']
                if node in results:
                    results[node]['spatial_score'] += 0.25
                    if 'fan_out' not in results[node]['signals']:
                        results[node]['signals'].append('fan_out_detected')
        
        # Centrality: top 25% get +0.2
        centrality_list = signals.get('centrality', [])
        if len(centrality_list) > 0:
            cutoff_idx = max(1, len(centrality_list) // 4)
            top_nodes = [node for node, _ in centrality_list[:cutoff_idx]]
            for node in top_nodes:
                if node in results:
                    results[node]['spatial_score'] += 0.2
                    if 'high_centrality' not in results[node]['signals']:
                        results[node]['signals'].append('high_centrality')
        
        # Cap at 1.0
        for node in results:
            results[node]['spatial_score'] = min(1.0, results[node]['spatial_score'])
    
    def _score_behavioral(self, results: Dict, signals: Dict):
        """Score behavioral (cyber) signals."""
        if not signals:
            return
        
        cyber_alerts = signals.get('cyber_alerts', [])
        if not cyber_alerts:
            return
        
        alert_types = {}
        for alert in cyber_alerts:
            if isinstance(alert, dict):
                user = alert.get('user_id')
                alert_type = alert.get('type', 'unknown')
                if user and user in results:
                    if user not in alert_types:
                        alert_types[user] = []
                    alert_types[user].append(alert_type)
        
        for user, types in alert_types.items():
            if user in results:
                # Each cyber alert type: +0.2
                results[user]['behavioral_score'] += 0.2 * len(set(types))
                for t in set(types):
                    results[user]['signals'].append(f'cyber_{t}')
        
        # Cap at 1.0
        for node in results:
            results[node]['behavioral_score'] = min(1.0, results[node]['behavioral_score'])
    
    def _score_temporal(self, results: Dict, signals: Dict):
        """Score temporal signals: risk escalation, concentration, cycle predictions."""
        if not signals:
            return
        
        # Risk escalation predictions: score = predicted_risk_probability
        for pred in signals.get('risk_predictions', []):
            if isinstance(pred, dict) and 'account' in pred:
                node = pred['account']
                if node in results:
                    risk_prob = float(pred.get('predicted_risk_probability', 0.0))
                    results[node]['temporal_score'] += risk_prob * 0.6
                    if 'risk_escalation' not in results[node]['signals']:
                        results[node]['signals'].append('risk_escalation')
        
        # Temporal concentration: +0.2 if concentration > 50%
        for item in signals.get('concentration', []):
            if isinstance(item, dict) and 'account' in item:
                node = item['account']
                conc = float(item.get('concentration_pct', 0.0))
                if conc > self.signal_thresholds.get('temporal_concentration', 0.5) * 100:
                    if node in results:
                        results[node]['temporal_score'] += 0.2
                        if 'temporal_concentration' not in results[node]['signals']:
                            results[node]['signals'].append('temporal_concentration')
        
        # Cycle emergence predictions: score = predicted_cycle_probability
        for pred in signals.get('cycle_pred', []):
            if isinstance(pred, dict) and 'account' in pred:
                node = pred['account']
                if node in results:
                    cycle_prob = float(pred.get('predicted_cycle_probability', 0.0))
                    results[node]['temporal_score'] += cycle_prob * 0.2
                    if 'cycle_emergence' not in results[node]['signals']:
                        results[node]['signals'].append('cycle_emergence')
        
        # Cap at 1.0
        for node in results:
            results[node]['temporal_score'] = min(1.0, results[node]['temporal_score'])
    
    def _score_lstm(self, results: Dict, signals: Dict):
        """Score LSTM link prediction signals."""
        if not signals:
            return
        
        emerging_links = signals.get('emerging_links', [])
        link_count = {}
        link_scores = {}
        
        for link_tuple in emerging_links:
            if isinstance(link_tuple, tuple) and len(link_tuple) == 2:
                (u, v), prob = link_tuple
                prob = float(prob)
                
                # Only count links above threshold
                if prob >= self.signal_thresholds.get('lstm_prob_min', 0.5):
                    link_count[u] = link_count.get(u, 0) + 1
                    link_count[v] = link_count.get(v, 0) + 1
                    
                    # Track max prob for each node
                    link_scores[u] = max(link_scores.get(u, 0), prob)
                    link_scores[v] = max(link_scores.get(v, 0), prob)
        
        for node, count in link_count.items():
            if node in results:
                # Score based on count (capped at 0.5) + max emerging link prob (up to 0.5)
                count_score = min(0.5, count * 0.1)  # max 0.5 for 5+ links
                prob_score = link_scores.get(node, 0.0) * 0.5
                results[node]['lstm_score'] = count_score + prob_score
                if 'emerging_link' not in results[node]['signals']:
                    results[node]['signals'].append('emerging_link')
        
        # Cap at 1.0
        for node in results:
            results[node]['lstm_score'] = min(1.0, results[node]['lstm_score'])
    
    def get_top_accounts(self, results: Dict, top_n: int = 10) -> List[Tuple[str, Dict]]:
        """Return top N highest-risk accounts."""
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['final_risk_score'],
            reverse=True
        )
        return sorted_results[:top_n]
    
    def get_summary_report(self, results: Dict) -> Dict:
        """Generate summary statistics."""
        if not results:
            return {}
        
        scores = [r['final_risk_score'] for r in results.values()]
        signal_counts = [r['signal_count'] for r in results.values()]
        
        for node in results:
            results[node]['signal_count'] = len(results[node]['signals'])
        
        return {
            'total_accounts': len(results),
            'high_risk_count': len([s for s in scores if s > 0.7]),
            'medium_risk_count': len([s for s in scores if 0.4 < s <= 0.7]),
            'low_risk_count': len([s for s in scores if 0.0 < s <= 0.4]),
            'clean_count': len([s for s in scores if s == 0.0]),
            'avg_risk_score': np.mean(scores),
            'max_risk_score': max(scores) if scores else 0.0,
            'median_signal_count': np.median(signal_counts) if signal_counts else 0
        }
