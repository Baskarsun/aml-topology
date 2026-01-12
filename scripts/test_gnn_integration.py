"""
Test GNN integration with inference API.
"""

import requests
import json
import sys

def test_gnn_integration():
    """Test GNN embedding integration in inference pipeline."""
    
    print("="*60)
    print("GNN Integration Test Suite")
    print("="*60)
    
    api_url = 'http://localhost:5000'
    
    # Check API health
    print("\n[Test 0] API Health Check")
    try:
        response = requests.get(f'{api_url}/health', timeout=5)
        if response.status_code == 200:
            print("✓ API is running")
            health = response.json()
            print(f"  Status: {health.get('status', 'unknown')}")
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        print("  Make sure the API is running: python src/inference_api.py")
        return False
    
    # Test 1: Score without GNN (new account)
    print("\n[Test 1] New Account (no GNN embedding)")
    print("-" * 60)
    
    test_data_1 = {
        'account_id': 'TEST_NEW_001',
        'transaction': {
            'amount': 5000,
            'mcc': '6011',
            'payment_type': 'crypto',
            'device_change': True,
            'ip_risk': 0.8,
            'count_1h': 10,
            'sum_24h': 25000,
            'uniq_payees_24h': 15,
            'country': 'RU'
        },
        'events': ['login_success', 'transfer', 'logout']
    }
    
    try:
        response = requests.post(
            f'{api_url}/score/consolidate', 
            json=test_data_1,
            timeout=10
        )
        
        if response.status_code == 200:
            result1 = response.json()
            print(f"✓ Request successful")
            print(f"  Account ID: {result1.get('account_id')}")
            print(f"  GNN Enhanced: {result1.get('gnn_enhanced', False)}")
            print(f"  Risk Score: {result1.get('consolidated_risk_score', 0):.3f}")
            print(f"  Risk Level: {result1.get('risk_level', 'UNKNOWN')}")
            print(f"  Recommendation: {result1.get('recommendation', 'N/A')}")
            
            if not result1.get('gnn_enhanced'):
                print("  ℹ️  As expected, GNN not available for new account")
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Request error: {e}")
    
    # Test 2: Score with GNN (existing account from cache)
    print("\n[Test 2] Existing Account (with GNN embedding)")
    print("-" * 60)
    
    test_data_2 = {
        'account_id': 'ACC_1001',  # From simulator pool
        'transaction': {
            'amount': 5000,
            'mcc': '6011',
            'payment_type': 'crypto',
            'device_change': True,
            'ip_risk': 0.8,
            'count_1h': 10,
            'sum_24h': 25000,
            'uniq_payees_24h': 15,
            'country': 'RU'
        },
        'events': ['login_success', 'transfer', 'logout']
    }
    
    try:
        response = requests.post(
            f'{api_url}/score/consolidate', 
            json=test_data_2,
            timeout=10
        )
        
        if response.status_code == 200:
            result2 = response.json()
            print(f"✓ Request successful")
            print(f"  Account ID: {result2.get('account_id')}")
            print(f"  GNN Enhanced: {result2.get('gnn_enhanced', False)}")
            print(f"  Risk Score: {result2.get('consolidated_risk_score', 0):.3f}")
            print(f"  Risk Level: {result2.get('risk_level', 'UNKNOWN')}")
            
            if result2.get('gnn_enhanced'):
                print("  ✓ GNN embedding found and used!")
                
                component_scores = result2.get('component_scores', {})
                if 'gnn' in component_scores:
                    gnn_data = component_scores['gnn']
                    print(f"  GNN Score: {gnn_data.get('score', 0):.3f}")
                    print(f"  GNN Weight: {result2.get('weights_used', {}).get('gnn', 0)}")
                    print(f"  Embedding Dim: {gnn_data.get('embedding_dim', 'N/A')}")
                    print(f"  Method: {gnn_data.get('method', 'N/A')}")
            else:
                print("  ℹ️  GNN embedding not found for this account")
                print("  Run: python scripts/generate_gnn_embeddings.py")
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Request error: {e}")
    
    # Test 3: Cache stats
    print("\n[Test 3] Cache Statistics")
    print("-" * 60)
    
    try:
        response = requests.get(f'{api_url}/health', timeout=5)
        if response.status_code == 200:
            health = response.json()
            
            # Look for GNN cache info
            gnn_info = health.get('gnn_cache')
            if gnn_info:
                print("✓ GNN Cache information found:")
                print(json.dumps(gnn_info, indent=2))
            else:
                print("ℹ️  No GNN cache information in health endpoint")
                print("  This is expected if GNN integration is not yet complete")
        else:
            print(f"✗ Health check failed")
    except Exception as e:
        print(f"✗ Error fetching health: {e}")
    
    # Test 4: Component score breakdown
    print("\n[Test 4] Component Score Analysis")
    print("-" * 60)
    
    try:
        response = requests.post(
            f'{api_url}/score/consolidate', 
            json=test_data_2,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            component_scores = result.get('component_scores', {})
            
            print("Component Scores:")
            for component, data in component_scores.items():
                status = data.get('status', 'unknown')
                if status == 'success':
                    score = data.get('score', 0)
                    weight = result.get('weights_used', {}).get(component, 0)
                    print(f"  {component.upper():12s} | Score: {score:.3f} | Weight: {weight:.2f}")
                else:
                    print(f"  {component.upper():12s} | Status: {status}")
            
            print(f"\nConsolidated Score: {result.get('consolidated_risk_score', 0):.3f}")
            print(f"Risk Level: {result.get('risk_level', 'UNKNOWN')}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)
    
    return True

if __name__ == '__main__':
    success = test_gnn_integration()
    sys.exit(0 if success else 1)
