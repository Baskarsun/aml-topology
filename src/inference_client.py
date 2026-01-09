"""
Inference API Client

Simple Python client for interacting with the AML Inference API.

Usage:
    from src.inference_client import InferenceClient
    
    client = InferenceClient('http://localhost:5000')
    
    # Score a transaction
    result = client.score_transaction({
        'amount': 5000.0,
        'mcc': '5411',
        'payment_type': 'card',
        ...
    })
    
    # Score an event sequence
    result = client.score_sequence(['login_success', 'transfer', 'logout'])
    
    # Consolidate all signals
    result = client.consolidate_risks(
        account_id='ACC_0025',
        transaction={...},
        events=[...]
    )
    
    # Batch score multiple accounts
    results = client.batch_score(transactions)
"""

import requests
import json
from typing import Dict, List, Any, Optional


class InferenceClient:
    """Client for AML Inference API."""
    
    def __init__(self, base_url: str = 'http://localhost:5000'):
        """
        Initialize client.
        
        Args:
            base_url: Base URL of inference API (default: localhost:5000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """
        Check API server health.
        
        Returns:
            Dict with model status and metadata
        """
        response = self.session.get(f'{self.base_url}/health')
        return response.json()
    
    def score_transaction(self, transaction: Dict) -> Dict:
        """
        Score a single transaction using GBDT model.
        
        Args:
            transaction: Dict with transaction features
                - amount (float)
                - mcc (str)
                - payment_type (str)
                - device_change (int)
                - ip_risk (float)
                - count_1h (int)
                - sum_24h (float)
                - uniq_payees_24h (int)
                - country (str)
        
        Returns:
            Dict with GBDT score and risk level
        """
        response = self.session.post(
            f'{self.base_url}/score/transaction',
            json=transaction
        )
        return response.json()
    
    def score_sequence(self, events: List[str]) -> Dict:
        """
        Score an event sequence using Sequence Detector model.
        
        Args:
            events: List of event type strings
                Valid types: login_success, login_failed, password_change,
                           add_payee, navigate_help, view_account, transfer,
                           max_transfer, logout
        
        Returns:
            Dict with sequence anomaly score and risk level
        """
        response = self.session.post(
            f'{self.base_url}/score/sequence',
            json={'events': events}
        )
        return response.json()
    
    def consolidate_risks(self, transaction: Dict = None, events: List[str] = None,
                         account_id: str = None) -> Dict:
        """
        Consolidate all signals into final risk score.
        
        Args:
            transaction: Transaction data (optional, for GBDT scoring)
            events: Event sequence (optional, for Sequence Detector)
            account_id: Account identifier (for tracking)
        
        Returns:
            Dict with consolidated risk score and recommendation
        """
        payload = {
            'account_id': account_id or 'UNKNOWN',
            'transaction': transaction or {},
            'events': events or []
        }
        
        response = self.session.post(
            f'{self.base_url}/score/consolidate',
            json=payload
        )
        return response.json()
    
    def batch_score(self, transactions: List[Dict]) -> Dict:
        """
        Score multiple transactions in batch.
        
        Args:
            transactions: List of dicts with:
                - account_id (str)
                - transaction (Dict with features)
                - events (List[str], optional)
        
        Returns:
            Dict with batch results and summary statistics
        """
        response = self.session.post(
            f'{self.base_url}/batch/score',
            json={'transactions': transactions}
        )
        return response.json()
    
    def get_models_info(self) -> Dict:
        """Get information about loaded models."""
        response = self.session.get(f'{self.base_url}/models/info')
        return response.json()
    
    def close(self):
        """Close client session."""
        self.session.close()


class InferenceClientAsync:
    """Async client for AML Inference API (requires aiohttp)."""
    
    def __init__(self, base_url: str = 'http://localhost:5000'):
        """Initialize async client."""
        try:
            import aiohttp
            self.aiohttp = aiohttp
        except ImportError:
            raise ImportError("aiohttp required for async client. Install: pip install aiohttp")
        
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = self.aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        await self.session.close()
    
    async def health_check(self) -> Dict:
        """Check API server health (async)."""
        async with self.session.get(f'{self.base_url}/health') as resp:
            return await resp.json()
    
    async def score_transaction(self, transaction: Dict) -> Dict:
        """Score a single transaction (async)."""
        async with self.session.post(
            f'{self.base_url}/score/transaction',
            json=transaction
        ) as resp:
            return await resp.json()
    
    async def score_sequence(self, events: List[str]) -> Dict:
        """Score an event sequence (async)."""
        async with self.session.post(
            f'{self.base_url}/score/sequence',
            json={'events': events}
        ) as resp:
            return await resp.json()
    
    async def consolidate_risks(self, transaction: Dict = None, events: List[str] = None,
                               account_id: str = None) -> Dict:
        """Consolidate all signals (async)."""
        payload = {
            'account_id': account_id or 'UNKNOWN',
            'transaction': transaction or {},
            'events': events or []
        }
        
        async with self.session.post(
            f'{self.base_url}/score/consolidate',
            json=payload
        ) as resp:
            return await resp.json()
    
    async def batch_score(self, transactions: List[Dict]) -> Dict:
        """Score multiple transactions in batch (async)."""
        async with self.session.post(
            f'{self.base_url}/batch/score',
            json={'transactions': transactions}
        ) as resp:
            return await resp.json()
    
    async def get_models_info(self) -> Dict:
        """Get information about loaded models (async)."""
        async with self.session.get(f'{self.base_url}/models/info') as resp:
            return await resp.json()


if __name__ == '__main__':
    # Example usage
    client = InferenceClient()
    
    # Health check
    print("Health check:")
    print(json.dumps(client.health_check(), indent=2))
    
    # Score a transaction
    print("\nScoring transaction:")
    tx = {
        'amount': 5000.0,
        'mcc': '5411',
        'payment_type': 'card',
        'device_change': 0,
        'ip_risk': 0.1,
        'count_1h': 3,
        'sum_24h': 15000.0,
        'uniq_payees_24h': 2,
        'country': 'US'
    }
    result = client.score_transaction(tx)
    print(json.dumps(result, indent=2))
    
    # Score a sequence
    print("\nScoring event sequence:")
    events = ['login_success', 'view_account', 'transfer', 'logout']
    result = client.score_sequence(events)
    print(json.dumps(result, indent=2))
    
    # Consolidate
    print("\nConsolidating risks:")
    result = client.consolidate_risks(
        account_id='ACC_0025',
        transaction=tx,
        events=events
    )
    print(json.dumps(result, indent=2))
    
    client.close()
