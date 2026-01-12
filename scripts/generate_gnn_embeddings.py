"""
Nightly batch job to generate GNN embeddings for all accounts.
Run via cron: 0 23 * * * python scripts/generate_gnn_embeddings.py
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import redis

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.gnn_trainer import load_gnn_model, build_node_features, adjacency_sparse_from_nx
from src.graph_analyzer import AMLGraphAnalyzer
import networkx as nx

def load_recent_transactions(db_path='metrics.db', days=30):
    """Load transaction data from metrics database."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    
    query = f"""
    SELECT 
        account_id,
        json_extract(component_scores, '$.target_account') as target_account,
        risk_score,
        timestamp
    FROM inference_logs
    WHERE datetime(timestamp) >= datetime('now', '-{days} days')
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def build_graph_from_transactions(df_tx, days=30):
    """Build NetworkX graph from transaction dataframe."""
    G = nx.DiGraph()
    
    # Add edges from transactions
    for _, row in df_tx.iterrows():
        source = row['account_id']
        target = row.get('target_account', 'EXTERNAL')
        
        if pd.notna(target) and target != 'EXTERNAL':
            G.add_edge(source, target, 
                      amount=row.get('risk_score', 0) * 10000,
                      timestamp=row['timestamp'])
    
    # Generate synthetic profiles for nodes
    profiles = {}
    for node in G.nodes():
        profiles[node] = {
            'age_days': 365,
            'balance': 5000.0,
            'tx_count': G.degree(node)
        }
    
    labels = {node: 0 for node in G.nodes()}  # Dummy labels
    
    return G, labels, profiles

def generate_embeddings():
    """Main function to generate and cache GNN embeddings."""
    
    print("="*60)
    print("GNN Embedding Generation - Nightly Batch Job")
    print("="*60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load GNN model
    print("\n[1/6] Loading GNN model...")
    try:
        model, metadata = load_gnn_model()
        model.eval()
        print(f"✓ Model loaded: {metadata.get('model_type', 'GraphSage')}")
    except Exception as e:
        print(f"✗ Failed to load GNN model: {e}")
        print("  Run: python src/gnn_trainer.py --save_model to train first")
        return False
    
    # Step 2: Load transaction data
    print("\n[2/6] Loading transaction data (last 30 days)...")
    try:
        df_tx = load_recent_transactions(days=30)
        print(f"✓ Loaded {len(df_tx)} transactions")
        print(f"  Unique accounts: {df_tx['account_id'].nunique()}")
    except Exception as e:
        print(f"✗ Failed to load transactions: {e}")
        return False
    
    if len(df_tx) == 0:
        print("⚠️  No transactions found. Skipping embedding generation.")
        return False
    
    # Step 3: Build graph
    print("\n[3/6] Building transaction graph...")
    try:
        G, labels, profiles = build_graph_from_transactions(df_tx, days=30)
        print(f"✓ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"✗ Failed to build graph: {e}")
        return False
    
    if G.number_of_nodes() == 0:
        print("⚠️  Empty graph. Skipping embedding generation.")
        return False
    
    # Step 4: Extract features
    print("\n[4/6] Extracting node features...")
    try:
        features, node_index, raw_features = build_node_features(G, df_tx, profiles)
        adj = adjacency_sparse_from_nx(G, node_index)
        print(f"✓ Features extracted: shape {features.shape}")
    except Exception as e:
        print(f"✗ Failed to extract features: {e}")
        return False
    
    # Step 5: GNN inference
    print("\n[5/6] Running GNN inference...")
    try:
        X = torch.from_numpy(features).float()
        
        with torch.no_grad():
            embeddings = model(X, adj)  # Forward pass
            
            if len(embeddings.shape) == 1:
                # Model outputs risk scores, use them as embeddings
                embeddings = embeddings.unsqueeze(1)
        
        print(f"✓ Generated embeddings: shape {embeddings.shape}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
    except Exception as e:
        print(f"✗ GNN inference failed: {e}")
        return False
    
    # Step 6: Cache embeddings
    print("\n[6/6] Caching embeddings...")
    
    # Connect to Redis
    try:
        cache = redis.Redis(host='localhost', port=6379, decode_responses=False)
        cache.ping()
        redis_available = True
        print("✓ Connected to Redis")
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("  Embeddings will be saved to file only")
        redis_available = False
    
    # Prepare embedding dict for JSON export
    embedding_dict = {}
    cached_count = 0
    
    nodes_sorted = sorted(node_index, key=lambda x: node_index[x])
    
    for node_id in nodes_sorted:
        idx = node_index[node_id]
        embedding = embeddings[idx].numpy()
        
        # Store in Redis
        if redis_available:
            try:
                cache.setex(
                    f"gnn_emb:{node_id}",
                    25 * 3600,  # 25 hour expiry (refresh before next run)
                    embedding.tobytes()
                )
                cached_count += 1
            except Exception as e:
                print(f"⚠️  Redis cache failed for {node_id}: {e}")
        
        # Store in dict for JSON export
        embedding_dict[node_id] = embedding.tolist()
    
    if redis_available:
        print(f"✓ Cached {cached_count} embeddings in Redis")
    
    # Fallback: Save to JSON file
    cache_file = os.path.join(ROOT, 'models', 'gnn_embeddings_cache.json')
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_accounts': len(embedding_dict),
                'embed_dim': embeddings.shape[1],
                'embeddings': embedding_dict
            }, f, indent=2)
        print(f"✓ Saved embeddings to {cache_file}")
    except Exception as e:
        print(f"✗ Failed to save JSON: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ Embedding generation complete!")
    print(f"  Total embeddings: {len(embedding_dict)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Cache expiry: 25 hours")
    print("="*60)
    
    return True

if __name__ == '__main__':
    success = generate_embeddings()
    sys.exit(0 if success else 1)
