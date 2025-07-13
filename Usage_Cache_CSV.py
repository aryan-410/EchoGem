import csv
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from chunk_response_class_ex import Chunk

class UsageCache:
    def __init__(self, csv_path: Optional[str] = None):
        self.store: Dict[str, Dict[str, Any]] = {}
        self.csv_path = csv_path

        if self.csv_path:
            self._load_from_csv()

    def _key(self, chunk_id: str) -> str:
        return f"chunk:{chunk_id}"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def set_chunk(self, chunk_id: str, metadata: Dict[str, Any]) -> None:
        self.store[self._key(chunk_id)] = metadata
        self._save_to_csv()

    def update_usage(self, chunk_id: str) -> None:
        key = self._key(chunk_id)
        if key not in self.store:
            self.store[key] = {"last_used": self._now(), "times_used": 1}
        else:
            self.store[key]["last_used"] = self._now()
            self.store[key]["times_used"] = int(self.store[key].get("times_used", 0)) + 1
        self._save_to_csv()

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, str]]:
        return self.store.get(self._key(chunk_id))

    def delete_chunk(self, chunk_id: str) -> None:
        key = self._key(chunk_id)
        if key in self.store:
            del self.store[key]
            self._save_to_csv()

    def get_all_chunks(self) -> Dict[str, Dict[str, str]]:
        return {k.split("chunk:")[1]: v for k, v in self.store.items()}

    def push_chunks(self, chunks: List[Chunk]) -> List[str]:
        ids: List[str] = []
        now = self._now()

        for chunk in chunks:
            id_ = hashlib.sha256(chunk.content.encode()).hexdigest()
            metadata = {"last_used": now, "times_used": 0}
            self.store[self._key(id_)] = metadata
            ids.append(id_)

        self._save_to_csv()
        print(f"Cached {len(chunks)} chunks")
        return ids

    def _save_to_csv(self) -> None:
        if not self.csv_path:
            return

        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["chunk_id", "last_used", "times_used"])
            for key, metadata in self.store.items():
                chunk_id = key.split("chunk:")[1]
                writer.writerow([
                    chunk_id,
                    metadata.get("last_used", ""),
                    metadata.get("times_used", 0)
                ])

    def _load_from_csv(self) -> None:
        try:
            with open(self.csv_path, mode="r", newline="", encoding="utf-8") as file: #type: ignore
                reader = csv.DictReader(file)
                for row in reader:
                    key = self._key(row["chunk_id"])
                    self.store[key] = {
                        "last_used": row.get("last_used", ""),
                        "times_used": int(row.get("times_used", 0))
                    }
            print(f"Loaded {len(self.store)} chunks from CSV.")
        except FileNotFoundError:
            print("No CSV cache found. Starting fresh.")
        except Exception as e:
            print(f"Error loading cache from CSV: {e}")
