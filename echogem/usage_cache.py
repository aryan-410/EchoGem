"""
Usage tracking and caching for EchoGem chunks and responses.
"""

import csv
import json
import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from .models import Chunk


def sha16(text: str) -> str:
    """Generate a 16-character hash from text"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


class UsageCache:
    """
    Stores chunk usage data in CSV format with rich metadata.
    
    Supports both rich CSV format (chunk_id,title,content,keywords,named_entities,
    timestamp_range,last_used,usage_count) and legacy format (chunk_id,last_used,times_used).
    """
    
    RICH_HEADERS = [
        "chunk_id", "title", "content", "keywords", "named_entities",
        "timestamp_range", "last_used", "usage_count"
    ]
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize the usage cache
        
        Args:
            csv_path: Path to CSV file for persistence
        """
        self.csv_path = csv_path
        # internal store by chunk_id
        self.rows: Dict[str, Dict[str, Any]] = {}
        if self.csv_path:
            self._load_from_csv()

    def _now(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now(timezone.utc).isoformat()

    def _blank_row(self, chunk_id: str) -> Dict[str, Any]:
        """Create a blank row for a new chunk"""
        return {
            "chunk_id": chunk_id,
            "title": "",
            "content": "",
            "keywords": "[]",
            "named_entities": "[]",
            "timestamp_range": "",
            "last_used": "",
            "usage_count": 0,
        }

    def push_chunks(self, chunks: List[Chunk]) -> List[str]:
        """
        Insert new chunks (or refresh existing) with rich metadata.
        
        Args:
            chunks: List of chunks to cache
            
        Returns:
            List of chunk IDs that were cached
        """
        ids: List[str] = []
        now = self._now()

        for ch in chunks:
            cid = sha16(ch.content or "")
            ids.append(cid)
            row = self.rows.get(cid) or self._blank_row(cid)

            # Fill/refresh rich fields from the Chunk
            row["title"] = ch.title or row.get("title", "") or ""
            row["content"] = ch.content or row.get("content", "") or ""
            row["keywords"] = json.dumps(list(ch.keywords or []), ensure_ascii=False)
            row["named_entities"] = json.dumps(list(ch.named_entities or []), ensure_ascii=False)
            row["timestamp_range"] = ch.timestamp_range or row.get("timestamp_range", "") or ""

            # Initialize recency/usage if not present
            if not row.get("last_used"):
                row["last_used"] = now
            row["usage_count"] = int(row.get("usage_count", 0) or 0)

            self.rows[cid] = row

        self._save_to_csv()
        print(f"Cached {len(chunks)} chunks")
        return ids

    def update_usage(self, chunk_id: str) -> None:
        """
        Update usage statistics for a chunk
        
        Args:
            chunk_id: ID of the chunk to update
        """
        row = self.rows.get(chunk_id)
        if not row:
            row = self._blank_row(chunk_id)
            self.rows[chunk_id] = row
        row["last_used"] = self._now()
        row["usage_count"] = int(row.get("usage_count", 0) or 0) + 1
        self._save_to_csv()

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get chunk data by ID
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Chunk data dictionary or None if not found
        """
        return self.rows.get(chunk_id)

    def delete_chunk(self, chunk_id: str) -> None:
        """
        Delete a chunk from the cache
        
        Args:
            chunk_id: ID of the chunk to delete
        """
        if chunk_id in self.rows:
            del self.rows[chunk_id]
            self._save_to_csv()

    def get_all_chunks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all chunks in the cache
        
        Returns:
            Dictionary mapping chunk IDs to chunk data
        """
        return self.rows

    def _save_to_csv(self) -> None:
        """Save cache data to CSV file"""
        if not self.csv_path:
            return
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.csv_path) if os.path.dirname(self.csv_path) else ".", exist_ok=True)
        
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.RICH_HEADERS)
            w.writeheader()
            for cid, row in self.rows.items():
                # ensure JSON strings are strings
                kw = row.get("keywords", "[]")
                ne = row.get("named_entities", "[]")
                if not isinstance(kw, str): 
                    kw = json.dumps(kw, ensure_ascii=False)
                if not isinstance(ne, str): 
                    ne = json.dumps(ne, ensure_ascii=False)

                w.writerow({
                    "chunk_id": cid,
                    "title": row.get("title", "") or "",
                    "content": row.get("content", "") or "",
                    "keywords": kw,
                    "named_entities": ne,
                    "timestamp_range": row.get("timestamp_range", "") or "",
                    "last_used": row.get("last_used", "") or "",
                    "usage_count": int(row.get("usage_count", 0) or 0),
                })

    def _load_from_csv(self) -> None:
        """Load cache data from CSV file"""
        try:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                cols = [c.strip() for c in (reader.fieldnames or [])]
                rich = set(self.RICH_HEADERS).issubset(set(cols))

                if rich:
                    for row in reader:
                        cid = row.get("chunk_id", "") or ""
                        if not cid: 
                            continue
                        self.rows[cid] = {
                            "chunk_id": cid,
                            "title": row.get("title", "") or "",
                            "content": row.get("content", "") or "",
                            "keywords": row.get("keywords", "") or "[]",
                            "named_entities": row.get("named_entities", "") or "[]",
                            "timestamp_range": row.get("timestamp_range", "") or "",
                            "last_used": row.get("last_used", "") or "",
                            "usage_count": int(row.get("usage_count", 0) or 0),
                        }
                    print(f"Loaded {len(self.rows)} chunks from CSV (rich).")
                else:
                    # old minimal CSV: chunk_id,last_used,times_used
                    for row in reader:
                        cid = row.get("chunk_id", "") or ""
                        if not cid: 
                            continue
                        self.rows[cid] = {
                            "chunk_id": cid,
                            "title": "",
                            "content": "",
                            "keywords": "[]",
                            "named_entities": "[]",
                            "timestamp_range": "",
                            "last_used": row.get("last_used", "") or "",
                            "usage_count": int(row.get("times_used", 0) or 0),
                        }
                    print(f"Loaded {len(self.rows)} chunks from CSV (legacy). Will upgrade on next save.")
        except FileNotFoundError:
            print("No CSV cache found. Starting fresh.")
        except Exception as e:
            print(f"Error loading cache from CSV: {e}")

    def clear(self) -> None:
        """Clear all cached data"""
        self.rows.clear()
        if self.csv_path and os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        print("Usage cache cleared")
