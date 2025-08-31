# Legacy: Experimental Retrieval Methods (v0.1.0-rc2)
# Different retrieval and scoring approaches tested
# Replaced by current scoring system in v0.2.0

import numpy as np
from typing import List, Dict, Tuple
import math
from datetime import datetime, timedelta

class NaiveRetrieval:
    """Simple retrieval without sophisticated scoring"""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = []
    
    def add_chunk(self, chunk: str, embedding: np.ndarray):
        """Add chunk and its embedding"""
        self.chunks.append(chunk)
        self.embeddings.append(embedding)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """Simple cosine similarity search"""
        if not self.embeddings:
            return []
        
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append(sim)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

class BM25Retrieval:
    """BM25 scoring for text retrieval"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.avg_length = 0
        self.idf = {}
    
    def fit(self, chunks: List[str]):
        """Calculate IDF scores"""
        self.chunks = chunks
        total_chunks = len(chunks)
        
        # Calculate document frequencies
        word_freq = {}
        total_length = 0
        
        for chunk in chunks:
            words = chunk.lower().split()
            total_length += len(words)
            
            for word in set(words):
                word_freq[word] = word_freq.get(word, 0) + 1
        
        self.avg_length = total_length / total_chunks
        
        # Calculate IDF
        for word, freq in word_freq.items():
            self.idf[word] = math.log((total_chunks - freq + 0.5) / (freq + 0.5))
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """BM25 search"""
        query_words = query.lower().split()
        scores = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_words = chunk.lower().split()
            chunk_length = len(chunk_words)
            
            score = 0
            for word in query_words:
                if word in self.idf:
                    tf = chunk_words.count(word)
                    score += self.idf[word] * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * chunk_length / self.avg_length))
            
            scores.append((chunk, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class HybridRetrieval:
    """Combination of semantic and keyword search"""
    
    def __init__(self, semantic_weight: float = 0.7):
        self.semantic_weight = semantic_weight
        self.chunks = []
        self.embeddings = []
        self.keywords = []
    
    def add_chunk(self, chunk: str, embedding: np.ndarray, keywords: List[str]):
        """Add chunk with embedding and keywords"""
        self.chunks.append(chunk)
        self.embeddings.append(embedding)
        self.keywords.append(keywords)
    
    def search(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """Hybrid search combining semantic and keyword matching"""
        if not self.chunks:
            return []
        
        # Semantic similarity
        semantic_scores = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            semantic_scores.append(sim)
        
        # Keyword matching
        query_words = set(query.lower().split())
        keyword_scores = []
        for chunk_keywords in self.keywords:
            overlap = len(query_words.intersection(set(chunk_keywords)))
            keyword_scores.append(overlap / len(query_words) if query_words else 0)
        
        # Combine scores
        combined_scores = []
        for i in range(len(self.chunks)):
            combined = (self.semantic_weight * semantic_scores[i] + 
                       (1 - self.semantic_weight) * keyword_scores[i])
            combined_scores.append((self.chunks[i], combined))
        
        # Sort and return top-k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in combined_scores[:top_k]]

class TimeAwareRetrieval:
    """Retrieval considering temporal aspects"""
    
    def __init__(self):
        self.chunks = []
        self.timestamps = []
        self.embeddings = []
    
    def add_chunk(self, chunk: str, embedding: np.ndarray, timestamp: datetime):
        """Add chunk with timestamp"""
        self.chunks.append(chunk)
        self.embeddings.append(embedding)
        self.timestamps.append(timestamp)
    
    def search(self, query_embedding: np.ndarray, query_time: datetime, 
               top_k: int = 5, time_weight: float = 0.3) -> List[str]:
        """Search with temporal relevance"""
        if not self.chunks:
            return []
        
        # Semantic similarity
        semantic_scores = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            semantic_scores.append(sim)
        
        # Temporal relevance (closer in time = higher score)
        time_scores = []
        for timestamp in self.timestamps:
            time_diff = abs((query_time - timestamp).total_seconds())
            time_score = 1.0 / (1.0 + time_diff / 3600)  # Normalize by hour
            time_scores.append(time_score)
        
        # Combine scores
        combined_scores = []
        for i in range(len(self.chunks)):
            combined = ((1 - time_weight) * semantic_scores[i] + 
                       time_weight * time_scores[i])
            combined_scores.append((self.chunks[i], combined))
        
        # Sort and return top-k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in combined_scores[:top_k]]

# TESTING RESULTS:
# - Naive: Fast but poor quality results
# - BM25: Good for keyword matching, poor semantic understanding
# - Hybrid: Better than individual methods but complex tuning
# - TimeAware: Good for temporal data, but not always relevant

# REPLACED BY: Current scoring system with similarity, entropy, and recency
