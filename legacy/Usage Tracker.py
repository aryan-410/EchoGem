# Legacy: Old Usage Tracking System (v0.1.0-rc3)
# Different approaches to tracking chunk usage
# Replaced by CSV-based UsageCache in v0.2.0

import json
import sqlite3
import pickle
from typing import Dict, List, Optional
from datetime import datetime
import os

class InMemoryUsageTracker:
    """Simple in-memory usage tracking - lost on restart"""
    
    def __init__(self):
        self.usage_counts = {}  # chunk_id -> count
        self.last_used = {}     # chunk_id -> timestamp
        self.query_history = [] # list of queries
    
    def record_usage(self, chunk_id: str, query: str):
        """Record that a chunk was used"""
        self.usage_counts[chunk_id] = self.usage_counts.get(chunk_id, 0) + 1
        self.last_used[chunk_id] = datetime.now().isoformat()
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'chunk_id': chunk_id
        })
    
    def get_usage_stats(self, chunk_id: str) -> Dict:
        """Get usage statistics for a chunk"""
        return {
            'count': self.usage_counts.get(chunk_id, 0),
            'last_used': self.last_used.get(chunk_id),
            'queries': [q for q in self.query_history if q['chunk_id'] == chunk_id]
        }
    
    def get_popular_chunks(self, top_k: int = 10) -> List[Dict]:
        """Get most popular chunks"""
        sorted_chunks = sorted(
            self.usage_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [
            {'chunk_id': chunk_id, 'count': count}
            for chunk_id, count in sorted_chunks
        ]

class SQLiteUsageTracker:
    """SQLite-based usage tracking - more persistent but complex"""
    
    def __init__(self, db_path: str = "usage_tracker.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_usage (
                chunk_id TEXT PRIMARY KEY,
                usage_count INTEGER DEFAULT 0,
                first_used TIMESTAMP,
                last_used TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                query TEXT,
                chunk_id TEXT,
                FOREIGN KEY (chunk_id) REFERENCES chunk_usage (chunk_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_usage(self, chunk_id: str, query: str):
        """Record chunk usage in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        # Update or insert chunk usage
        cursor.execute('''
            INSERT OR REPLACE INTO chunk_usage (chunk_id, usage_count, first_used, last_used)
            VALUES (?, 
                    COALESCE((SELECT usage_count FROM chunk_usage WHERE chunk_id = ?), 0) + 1,
                    COALESCE((SELECT first_used FROM chunk_usage WHERE chunk_id = ?), ?),
                    ?)
        ''', (chunk_id, chunk_id, chunk_id, now, now))
        
        # Add query to history
        cursor.execute('''
            INSERT INTO query_history (timestamp, query, chunk_id)
            VALUES (?, ?, ?)
        ''', (now, query, chunk_id))
        
        conn.commit()
        conn.close()
    
    def get_usage_stats(self, chunk_id: str) -> Dict:
        """Get usage statistics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT usage_count, first_used, last_used
            FROM chunk_usage WHERE chunk_id = ?
        ''', (chunk_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            count, first_used, last_used = result
            return {
                'count': count,
                'first_used': first_used,
                'last_used': last_used
            }
        return {'count': 0, 'first_used': None, 'last_used': None}

class PickleUsageTracker:
    """Pickle-based usage tracking - simple but not scalable"""
    
    def __init__(self, filepath: str = "usage_tracker.pkl"):
        self.filepath = filepath
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load data from pickle file"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {
            'usage_counts': {},
            'last_used': {},
            'query_history': []
        }
    
    def _save_data(self):
        """Save data to pickle file"""
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.data, f)
    
    def record_usage(self, chunk_id: str, query: str):
        """Record chunk usage"""
        self.data['usage_counts'][chunk_id] = self.data['usage_counts'].get(chunk_id, 0) + 1
        self.data['last_used'][chunk_id] = datetime.now().isoformat()
        self.data['query_history'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'chunk_id': chunk_id
        })
        self._save_data()
    
    def get_usage_stats(self, chunk_id: str) -> Dict:
        """Get usage statistics"""
        return {
            'count': self.data['usage_counts'].get(chunk_id, 0),
            'last_used': self.data['last_used'].get(chunk_id)
        }

class RedisUsageTracker:
    """Redis-based usage tracking - fast but requires external service"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.prefix = "usage:"
    
    def record_usage(self, chunk_id: str, query: str):
        """Record usage in Redis"""
        now = datetime.now().isoformat()
        
        # Increment usage count
        self.redis.hincrby(f"{self.prefix}counts", chunk_id, 1)
        
        # Update last used
        self.redis.hset(f"{self.prefix}last_used", chunk_id, now)
        
        # Add to query history (with TTL)
        history_key = f"{self.prefix}history:{chunk_id}"
        self.redis.lpush(history_key, json.dumps({
            'timestamp': now,
            'query': query
        }))
        self.redis.expire(history_key, 86400)  # 24 hour TTL
    
    def get_usage_stats(self, chunk_id: str) -> Dict:
        """Get usage statistics from Redis"""
        count = self.redis.hget(f"{self.prefix}counts", chunk_id)
        last_used = self.redis.hget(f"{self.prefix}last_used", chunk_id)
        
        return {
            'count': int(count) if count else 0,
            'last_used': last_used.decode() if last_used else None
        }

# TESTING RESULTS:
# - InMemory: Fast but data loss on restart
# - SQLite: Persistent but complex setup and queries
# - Pickle: Simple but not scalable, file corruption risk
# - Redis: Fast and scalable but external dependency

# REPLACED BY: CSV-based UsageCache for simplicity and portability
