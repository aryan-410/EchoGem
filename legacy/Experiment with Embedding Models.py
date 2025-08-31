# Legacy: Old Embedding Models (v0.1.0-rc1)
# Various embedding approaches tested during development
# Replaced by GoogleGenerativeAIEmbeddings in v0.2.0

import numpy as np
from typing import List, Optional
import hashlib
import re

class TFIDFEmbeddings:
    """Term Frequency-Inverse Document Frequency embeddings"""
    
    def __init__(self):
        self.word_freq = {}
        self.doc_freq = {}
        self.total_docs = 0
        self.vocab = set()
    
    def fit(self, documents: List[str]):
        """Build vocabulary and calculate TF-IDF"""
        self.total_docs = len(documents)
        
        # Count word frequencies per document
        for doc in documents:
            words = re.findall(r'\w+', doc.lower())
            doc_words = set(words)
            
            for word in doc_words:
                self.doc_freq[word] = self.doc_freq.get(word, 0) + 1
                self.word_freq[word] = self.word_freq.get(word, 0) + words.count(word)
                self.vocab.add(word)
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector"""
        words = re.findall(r'\w+', text.lower())
        
        # Create vector
        vector = np.zeros(len(self.vocab))
        vocab_list = list(self.vocab)
        
        for word in words:
            if word in self.vocab:
                idx = vocab_list.index(word)
                tf = words.count(word) / len(words)
                idf = np.log(self.total_docs / self.doc_freq[word])
                vector[idx] = tf * idf
        
        return vector

class HashEmbeddings:
    """Simple hash-based embeddings for testing"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    def embed(self, text: str) -> np.ndarray:
        """Create deterministic hash-based embedding"""
        # Hash the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to vector
        vector = np.zeros(self.dimension)
        for i, char in enumerate(text_hash):
            if i >= self.dimension:
                break
            vector[i] = ord(char) / 255.0
        
        return vector

class RandomEmbeddings:
    """Random embeddings for baseline testing"""
    
    def __init__(self, dimension: int = 768, seed: int = 42):
        self.dimension = dimension
        np.random.seed(seed)
    
    def embed(self, text: str) -> np.ndarray:
        """Generate random embedding based on text length"""
        # Use text length as seed for reproducible randomness
        seed = hash(text) % 10000
        np.random.seed(seed)
        return np.random.randn(self.dimension)

class Word2VecMock:
    """Mock Word2Vec implementation for testing"""
    
    def __init__(self, dimension: int = 100):
        self.dimension = dimension
        self.word_vectors = {}
    
    def fit(self, documents: List[str]):
        """Mock training - just create random vectors for words"""
        words = set()
        for doc in documents:
            words.update(re.findall(r'\w+', doc.lower()))
        
        for word in words:
            self.word_vectors[word] = np.random.randn(self.dimension)
    
    def embed(self, text: str) -> np.ndarray:
        """Average word vectors"""
        words = re.findall(r'\w+', text.lower())
        if not words:
            return np.zeros(self.dimension)
        
        vectors = []
        for word in words:
            if word in self.word_vectors:
                vectors.append(self.word_vectors[word])
        
        if not vectors:
            return np.zeros(self.dimension)
        
        return np.mean(vectors, axis=0)

# TESTING RESULTS:
# - TFIDF: Good for keyword matching, poor semantic understanding
# - Hash: Fast but no semantic meaning, poor similarity search
# - Random: Baseline performance, no meaningful relationships
# - Word2Vec Mock: Better than random, but limited vocabulary

# REPLACED BY: GoogleGenerativeAIEmbeddings for semantic understanding
