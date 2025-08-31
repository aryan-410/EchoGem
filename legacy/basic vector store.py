# Legacy: Basic Vector Store (v0.1.0-beta)
# Simple in-memory vector storage using numpy
# Replaced by Pinecone integration in v0.2.0

import numpy as np
from typing import List, Dict, Optional
import pickle
import os

class BasicVectorStore:
    """Simple in-memory vector storage - replaced by Pinecone"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.vectors = {}  # id -> vector
        self.metadata = {}  # id -> metadata
        self.vector_matrix = None  # numpy array for similarity search
    
    def add_vector(self, vector_id: str, vector: np.ndarray, metadata: Dict):
        """Add a vector to the store"""
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension {vector.shape[0]} != {self.dimension}")
        
        self.vectors[vector_id] = vector
        self.metadata[vector_id] = metadata
        self._rebuild_matrix()
    
    def _rebuild_matrix(self):
        """Rebuild the numpy matrix for similarity search"""
        if not self.vectors:
            self.vector_matrix = None
            return
        
        vectors_list = list(self.vectors.values())
        self.vector_matrix = np.vstack(vectors_list)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Simple cosine similarity search"""
        if self.vector_matrix is None:
            return []
        
        # Normalize vectors
        query_norm = query_vector / np.linalg.norm(query_vector)
        matrix_norm = self.vector_matrix / np.linalg.norm(self.vector_matrix, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(matrix_norm, query_norm)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        vector_ids = list(self.vectors.keys())
        for idx in top_indices:
            results.append({
                'id': vector_ids[idx],
                'similarity': similarities[idx],
                'metadata': self.metadata[vector_ids[idx]]
            })
        
        return results
    
    def save(self, filepath: str):
        """Save to pickle file"""
        data = {
            'vectors': self.vectors,
            'metadata': self.metadata,
            'dimension': self.dimension
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load from pickle file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data['vectors']
                self.metadata = data['metadata']
                self.dimension = data['dimension']
                self._rebuild_matrix()

# PROBLEMS WITH THIS APPROACH:
# - In-memory storage limited by RAM
# - No persistence across sessions
# - No distributed access
# - Poor scalability for large datasets
# - No built-in similarity search optimization
# - Manual matrix rebuilding on every update

# REPLACED BY: Pinecone integration in ChunkVectorDB class
