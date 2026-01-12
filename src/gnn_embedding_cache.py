"""
GNN Embedding Cache - Fast lookup for pre-computed embeddings.
"""

import os
import json
import numpy as np
from typing import Optional
import redis

class GNNEmbeddingCache:
    """Manages GNN embedding cache with Redis and file fallback."""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_available = False
        self.embedding_cache = {}
        self.embedding_dim = None
        self.last_update = None
        
        # Try to connect to Redis
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                decode_responses=False,
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            self.redis_available = True
            print(f"✓ GNN Cache: Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            print(f"⚠️  GNN Cache: Redis unavailable ({e}), using file fallback")
            self.redis_client = None
        
        # Load fallback cache from file
        self._load_file_cache()
    
    def _load_file_cache(self):
        """Load embeddings from JSON file (fallback)."""
        cache_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'models', 
            'gnn_embeddings_cache.json'
        )
        
        if not os.path.exists(cache_file):
            print(f"⚠️  GNN Cache: No cache file found at {cache_file}")
            return
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            self.embedding_cache = data.get('embeddings', {})
            self.embedding_dim = data.get('embed_dim')
            self.last_update = data.get('generated_at')
            
            print(f"✓ GNN Cache: Loaded {len(self.embedding_cache)} embeddings from file")
            print(f"  Generated at: {self.last_update}")
            print(f"  Dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"✗ GNN Cache: Failed to load file cache: {e}")
    
    def get(self, account_id: str) -> Optional[np.ndarray]:
        """
        Retrieve GNN embedding for account.
        
        Returns:
            numpy array of embedding, or None if not found
        """
        # Try Redis first (fastest)
        if self.redis_available:
            try:
                key = f"gnn_emb:{account_id}"
                emb_bytes = self.redis_client.get(key)
                if emb_bytes:
                    return np.frombuffer(emb_bytes, dtype=np.float32)
            except Exception as e:
                # Redis failed, fall through to file cache
                pass
        
        # Fallback to in-memory dict
        if account_id in self.embedding_cache:
            return np.array(self.embedding_cache[account_id], dtype=np.float32)
        
        return None
    
    def has_embedding(self, account_id: str) -> bool:
        """Check if embedding exists for account."""
        return self.get(account_id) is not None
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'redis_available': self.redis_available,
            'file_cache_size': len(self.embedding_cache),
            'embedding_dim': self.embedding_dim,
            'last_update': self.last_update
        }

# Global singleton
_cache_instance = None

def get_gnn_cache() -> GNNEmbeddingCache:
    """Get or create global GNN cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = GNNEmbeddingCache()
    return _cache_instance
