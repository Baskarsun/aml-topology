"""
Inference API Service for AML Detection Pipeline

Provides REST endpoints to:
1. Score individual transactions via GBDT
2. Analyze event sequences via Sequence Detector
3. Perform graph-based analysis via GNN
4. Predict emerging links via LSTM
5. Consolidate all signals into final risk scores

Models are loaded once at startup for performance.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Ensure workspace imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify

try:
    import flask
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False
    print("Warning: Flask not installed. Install with: pip install flask")

from src.gbdt_detector import load_gbdt_model, score_transaction
from src.sequence_detector import load_sequence_model, SequenceDataset, EVENT2IDX
from src.lstm_link_predictor import load_model as load_lstm_model
from src.risk_consolidator import RiskConsolidator
from src.metrics_logger import get_metrics_logger
from src.gnn_embedding_cache import get_gnn_cache

# Try loading GNN (optional)
try:
    from src.gnn_trainer import load_gnn_model
    _HAS_GNN = True
except ImportError:
    _HAS_GNN = False


class InferenceEngine:
    """Loads persisted models and provides inference methods."""
    
    def __init__(self):
        """Initialize inference engine and load all models."""
        self.models = {}
        self.metadata = {}
        self.consolidator = None
        self.feature_maps = {}
        self.gnn_cache = None
        self.load_models()
        
        # Initialize GNN embedding cache
        try:
            self.gnn_cache = get_gnn_cache()
            cache_stats = self.gnn_cache.get_stats()
            print(f"✓ GNN Embedding Cache loaded")
            print(f"  Embeddings available: {cache_stats['file_cache_size']}")
            print(f"  Redis status: {'Connected' if cache_stats['redis_available'] else 'Offline'}")
        except Exception as e:
            print(f"✗ GNN Cache initialization failed: {e}")
            self.gnn_cache = None
    
    def load_models(self):
        """Load all persisted models with error handling."""
        print("Loading inference models...")
        
        # Load GBDT
        try:
            self.models['gbdt'], self.metadata['gbdt'] = load_gbdt_model()
            print("✓ GBDT model loaded")
        except Exception as e:
            print(f"✗ GBDT loading failed: {e}")
            self.models['gbdt'] = None
        
        # Load Sequence Detector
        try:
            self.models['sequence'], self.metadata['sequence'] = load_sequence_model()
            print("✓ Sequence Detector model loaded")
        except Exception as e:
            print(f"✗ Sequence Detector loading failed: {e}")
            self.models['sequence'] = None
        
        # Load LSTM Link Predictor
        try:
            lstm_path = os.path.join(ROOT, 'models', 'lstm_link_predictor.pt')
            self.models['lstm'], self.metadata['lstm'] = load_lstm_model(lstm_path)
            print("✓ LSTM Link Predictor model loaded")
        except Exception as e:
            print(f"✗ LSTM loading failed: {e}")
            self.models['lstm'] = None
        
        # Load GNN (optional)
        if _HAS_GNN:
            try:
                self.models['gnn'], self.metadata['gnn'] = load_gnn_model()
                print("✓ GNN model loaded")
            except Exception as e:
                print(f"✗ GNN loading failed: {e}")
                self.models['gnn'] = None
        
        # Load Risk Consolidator
        try:
            config_path = os.path.join(ROOT, 'models', 'consolidation_config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.consolidator = RiskConsolidator(weights=config.get('weights', {}))
            self.metadata['consolidator'] = config
            print("✓ Risk Consolidator loaded")
        except Exception as e:
            print(f"✗ Risk Consolidator loading failed: {e}")
            self.consolidator = None
        
        print(f"Inference engine ready. {len(self.models)} models loaded.\n")
    
    def score_transaction(self, transaction: Dict) -> Dict:
        """
        Score a single transaction using GBDT.
        
        Args:
            transaction: Dict with keys: amount, mcc, payment_type, device_change,
                        ip_risk, count_1h, sum_24h, uniq_payees_24h, country
        
        Returns:
            Dict with GBDT score and metadata
        """
        if self.models['gbdt'] is None:
            return {'error': 'GBDT model not loaded'}
        
        try:
            # Score using GBDT
            score = score_transaction(transaction, {}, self.models['gbdt'])
            
            return {
                'transaction': transaction,
                'gbdt_score': float(score),
                'gbdt_risk_level': self._score_to_risk_level(float(score)),
                'model': 'gbdt',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': f'Scoring failed: {str(e)}'}
    
    def score_event_sequence(self, events: List[str]) -> Dict:
        """
        Score an event sequence (e.g., login, transfer, logout).
        
        Args:
            events: List of event type strings (e.g., ['login_success', 'transfer', 'logout'])
        
        Returns:
            Dict with anomaly score and metadata
        """
        if self.models['sequence'] is None:
            return {'error': 'Sequence Detector not loaded'}
        
        try:
            # Encode events to indices
            seq_indices = [EVENT2IDX.get(e, 0) for e in events]
            
            # Pad to max length
            max_len = 20
            if len(seq_indices) < max_len:
                seq_indices = seq_indices + [0] * (max_len - len(seq_indices))
            else:
                seq_indices = seq_indices[:max_len]
            
            # Create tensor and score
            X = torch.tensor([seq_indices], dtype=torch.long).unsqueeze(1).float()
            
            with torch.no_grad():
                score = self.models['sequence'](X).item()
            
            return {
                'events': events,
                'sequence_score': float(score),
                'anomaly_risk_level': self._score_to_risk_level(float(score)),
                'model': 'sequence_detector',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': f'Sequence scoring failed: {str(e)}'}
    
    def consolidate_risks(self, transaction: Dict, events: List[str] = None, 
                         account_id: str = None) -> Dict:
        """
        Consolidate GBDT + Sequence + GNN scores into final risk assessment.
        
        Args:
            transaction: Transaction data for GBDT scoring
            events: Optional event sequence for Sequence Detector
            account_id: Account identifier for tracking
        
        Returns:
            Dict with consolidated risk scores from all available models
        """
        if self.consolidator is None:
            return {'error': 'Risk Consolidator not loaded'}
        
        results = {
            'account_id': account_id or 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'component_scores': {}
        }
        
        # Score transaction with GBDT
        if self.models['gbdt'] is not None:
            try:
                gbdt_score = score_transaction(transaction, {}, self.models['gbdt'])
                results['component_scores']['gbdt'] = {
                    'score': float(gbdt_score),
                    'weight': self.consolidator.weights.get('cyber', 0.1),
                    'status': 'success'
                }
            except Exception as e:
                results['component_scores']['gbdt'] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Score event sequence
        if events and self.models['sequence'] is not None:
            try:
                seq_result = self.score_event_sequence(events)
                if 'sequence_score' in seq_result:
                    results['component_scores']['sequence'] = {
                        'score': seq_result['sequence_score'],
                        'weight': self.consolidator.weights.get('temporal', 0.35),
                        'status': 'success'
                    }
            except Exception as e:
                results['component_scores']['sequence'] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # === NEW: GNN EMBEDDING LOOKUP ===
        gnn_score = 0.0
        gnn_available = False
        
        if self.gnn_cache is not None and account_id:
            try:
                embedding = self.gnn_cache.get(account_id)
                
                if embedding is not None:
                    # Compute risk score from embedding
                    # Method 1: Use L2 norm (simple)
                    gnn_score = float(np.linalg.norm(embedding))
                    gnn_score = min(1.0, gnn_score / 10.0)  # Normalize to [0, 1]
                    
                    # Method 2: Use mean of embedding values
                    # gnn_score = float(np.mean(np.abs(embedding)))
                    
                    gnn_available = True
                    
                    results['component_scores']['gnn'] = {
                        'score': gnn_score,
                        'weight': 0.35,  # High weight when available
                        'status': 'success',
                        'embedding_dim': len(embedding),
                        'method': 'l2_norm'
                    }
            except Exception as e:
                results['component_scores']['gnn'] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # === ADAPTIVE WEIGHT ADJUSTMENT ===
        if gnn_available:
            # GNN available: use GNN-heavy weights
            weights = {
                'gbdt': 0.20,
                'sequence': 0.10,
                'gnn': 0.40,       # High weight for GNN
                'temporal': 0.20,
                'lstm': 0.10
            }
        else:
            # GNN unavailable: fallback weights
            weights = {
                'gbdt': 0.30,
                'sequence': 0.20,
                'gnn': 0.0,
                'temporal': 0.30,
                'lstm': 0.20
            }
        
        # === COMPUTE FINAL SCORE ===
        consolidated_score = 0.0
        total_weight = 0.0
        
        for component, data in results['component_scores'].items():
            if data.get('status') == 'success' and 'score' in data:
                score = data['score']
                weight = weights.get(component, 0.0)
                consolidated_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            consolidated_score /= total_weight
        
        consolidated_score = min(1.0, max(0.0, consolidated_score))
        
        results['consolidated_risk_score'] = consolidated_score
        results['risk_level'] = self._score_to_risk_level(consolidated_score)
        results['recommendation'] = self._get_recommendation(consolidated_score)
        results['gnn_enhanced'] = gnn_available
        results['weights_used'] = weights
        
        return results
    
    def _compute_consolidated_score(self, component_scores: Dict) -> float:
        """Compute weighted average of available component scores."""
        if not component_scores:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for component, data in component_scores.items():
            if data.get('status') == 'success' and 'score' in data:
                weight = data.get('weight', 0.2)
                score = data['score']
                weighted_sum += weight * score
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return min(1.0, weighted_sum / total_weight)
    
    def _score_to_risk_level(self, score: float) -> str:
        """Convert numeric score to risk level."""
        if score >= 0.7:
            return 'HIGH'
        elif score >= 0.4:
            return 'MEDIUM'
        elif score > 0.0:
            return 'LOW'
        else:
            return 'CLEAN'
    
    def _get_recommendation(self, score: float) -> str:
        """Get action recommendation based on risk score."""
        if score >= 0.7:
            return 'Block or require additional verification'
        elif score >= 0.4:
            return 'Monitor closely, may require review'
        elif score > 0.0:
            return 'Log for monitoring'
        else:
            return 'Allow - no suspicious activity detected'
    
    def health_check(self) -> Dict:
        """Return status of loaded models."""
        health_data = {
            'status': 'healthy' if self.models.get('gbdt') else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': {
                'gbdt': self.models.get('gbdt') is not None,
                'sequence': self.models.get('sequence') is not None,
                'lstm': self.models.get('lstm') is not None,
                'gnn': self.models.get('gnn') is not None,
                'consolidator': self.consolidator is not None
            },
            'metadata': {k: {**v, 'timestamp': str(v.get('timestamp', 'N/A'))} 
                        for k, v in self.metadata.items()}
        }
        
        # Add GNN cache stats
        if self.gnn_cache is not None:
            try:
                cache_stats = self.gnn_cache.get_stats()
                health_data['gnn_cache'] = {
                    'available': True,
                    'redis_connected': cache_stats['redis_available'],
                    'embeddings_count': cache_stats['file_cache_size'],
                    'embedding_dim': cache_stats['embedding_dim'],
                    'last_update': cache_stats['last_update']
                }
            except Exception as e:
                health_data['gnn_cache'] = {
                    'available': False,
                    'error': str(e)
                }
        else:
            health_data['gnn_cache'] = {
                'available': False,
                'reason': 'Cache not initialized'
            }
        
        return health_data


def create_app(inference_engine: InferenceEngine = None) -> Flask:
    """Create and configure Flask application."""
    
    if not _HAS_FLASK:
        raise RuntimeError("Flask is required. Install with: pip install flask")
    
    app = Flask(__name__)
    
    # Initialize inference engine
    engine = inference_engine or InferenceEngine()
    
    # Initialize metrics logger
    metrics = get_metrics_logger()
    
    # ==================== ENDPOINTS ====================
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify(engine.health_check())
    
    @app.route('/score/transaction', methods=['POST'])
    def score_transaction_endpoint():
        """
        Score a single transaction.
        
        Expected JSON:
        {
            "amount": 5000.0,
            "mcc": "5411",
            "payment_type": "card",
            "device_change": 0,
            "ip_risk": 0.1,
            "count_1h": 3,
            "sum_24h": 15000.0,
            "uniq_payees_24h": 2,
            "country": "US"
        }
        
        Response:
        {
            "transaction": {...},
            "gbdt_score": 0.45,
            "gbdt_risk_level": "MEDIUM",
            "timestamp": "2026-01-09T..."
        }
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            result = engine.score_transaction(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/score/sequence', methods=['POST'])
    def score_sequence_endpoint():
        """
        Score an event sequence.
        
        Expected JSON:
        {
            "events": ["login_success", "view_account", "transfer", "logout"]
        }
        
        Response:
        {
            "events": [...],
            "sequence_score": 0.25,
            "anomaly_risk_level": "LOW",
            "timestamp": "2026-01-09T..."
        }
        """
        try:
            data = request.get_json()
            if not data or 'events' not in data:
                return jsonify({'error': 'Missing "events" field'}), 400
            
            events = data['events']
            if not isinstance(events, list):
                return jsonify({'error': '"events" must be a list'}), 400
            
            result = engine.score_event_sequence(events)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/score/consolidate', methods=['POST'])
    def consolidate_endpoint():
        """
        Consolidate all signals into final risk score.
        
        Expected JSON:
        {
            "account_id": "ACC_0025",
            "transaction": {
                "amount": 5000.0,
                "mcc": "5411",
                ...
            },
            "events": ["login_success", "transfer", "logout"]
        }
        
        Response:
        {
            "account_id": "ACC_0025",
            "component_scores": {
                "gbdt": {"score": 0.45, "weight": 0.1, "status": "success"},
                "sequence": {"score": 0.25, "weight": 0.35, "status": "success"}
            },
            "consolidated_risk_score": 0.38,
            "risk_level": "LOW",
            "recommendation": "Allow - no suspicious activity detected",
            "timestamp": "2026-01-09T..."
        }
        """
        start_time = datetime.now()
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            account_id = data.get('account_id')
            transaction = data.get('transaction', {})
            events = data.get('events', None)
            
            result = engine.consolidate_risks(transaction, events, account_id)
            
            # Log metrics
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            metrics.log_inference({
                'timestamp': result.get('timestamp'),
                'account_id': account_id,
                'endpoint': '/score/consolidate',
                'engine': 'consolidated',
                'latency_ms': latency_ms,
                'risk_score': result.get('consolidated_risk_score'),
                'risk_level': result.get('risk_level'),
                'component_scores': result.get('component_scores'),
                'status': 'success'
            })
            
            # Log engine activities
            for engine_name, engine_data in result.get('component_scores', {}).items():
                if engine_data.get('status') == 'success':
                    metrics.log_engine_activity(
                        engine=engine_name,
                        operation='score',
                        latency_ms=latency_ms / len(result.get('component_scores', {}))
                    )
            
            return jsonify(result)
        except Exception as e:
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            metrics.log_inference({
                'timestamp': datetime.now().isoformat(),
                'account_id': data.get('account_id') if 'data' in locals() else None,
                'endpoint': '/score/consolidate',
                'engine': 'consolidated',
                'latency_ms': latency_ms,
                'status': 'error',
                'error': str(e)
            })
            return jsonify({'error': str(e)}), 500
    
    @app.route('/batch/score', methods=['POST'])
    def batch_score_endpoint():
        """
        Score multiple transactions in batch.
        
        Expected JSON:
        {
            "transactions": [
                {
                    "account_id": "ACC_0001",
                    "transaction": {...},
                    "events": [...]
                },
                ...
            ]
        }
        
        Response:
        {
            "batch_id": "batch_2026-01-09_...",
            "total": 3,
            "results": [
                {"account_id": "ACC_0001", "consolidated_risk_score": 0.45, ...},
                ...
            ],
            "summary": {
                "high_risk": 1,
                "medium_risk": 1,
                "low_risk": 1,
                "clean": 0
            }
        }
        """
        try:
            data = request.get_json()
            if not data or 'transactions' not in data:
                return jsonify({'error': 'Missing "transactions" field'}), 400
            
            transactions = data['transactions']
            if not isinstance(transactions, list):
                return jsonify({'error': '"transactions" must be a list'}), 400
            
            results = []
            risk_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'CLEAN': 0}
            
            for tx_data in transactions:
                account_id = tx_data.get('account_id', f'TXN_{len(results)}')
                transaction = tx_data.get('transaction', {})
                events = tx_data.get('events', None)
                
                result = engine.consolidate_risks(transaction, events, account_id)
                results.append(result)
                
                risk_level = result.get('risk_level', 'CLEAN')
                risk_counts[risk_level] += 1
            
            return jsonify({
                'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'total': len(transactions),
                'results': results,
                'summary': {
                    'high_risk': risk_counts['HIGH'],
                    'medium_risk': risk_counts['MEDIUM'],
                    'low_risk': risk_counts['LOW'],
                    'clean': risk_counts['CLEAN']
                },
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/models/info', methods=['GET'])
    def models_info_endpoint():
        """Get information about loaded models."""
        return jsonify({
            'available_models': list(engine.models.keys()),
            'metadata': {k: {**v, 'timestamp': str(v.get('timestamp', 'N/A'))} 
                        for k, v in engine.metadata.items()},
            'consolidator_weights': engine.consolidator.weights if engine.consolidator else None,
            'timestamp': datetime.now().isoformat()
        })
    
    return app


def main():
    """Run inference API server."""
    if not _HAS_FLASK:
        print("Error: Flask is required to run the API server.")
        print("Install with: pip install flask")
        return
    
    print("Starting AML Inference API Server...")
    engine = InferenceEngine()
    app = create_app(engine)
    
    # Run on localhost:5000
    print("\nAPI Server running on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /health                      - Health check")
    print("  POST /score/transaction           - Score single transaction")
    print("  POST /score/sequence              - Score event sequence")
    print("  POST /score/consolidate           - Consolidate all signals")
    print("  POST /batch/score                 - Batch scoring")
    print("  GET  /models/info                 - Model information")
    print("\nPress Ctrl+C to stop.\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
