# %%

# %%
import redis
import json
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from chunk_response_class_ex import Chunk, ChunkResponse

# %%
import redis, uuid, hashlib, json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# %%
class UsageCache:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.client.ping()

    def _key(self, chunk_id: str) -> str:
        return f"chunk:{chunk_id}"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def set_chunk(self, chunk_id: str, metadata: Dict[str, Any]) -> None:
        self.client.hset(self._key(chunk_id), mapping=metadata)

    def update_usage(self, chunk_id: str) -> None:
        pipe = self.client.pipeline()
        pipe.hset(self._key(chunk_id), "last_used", self._now())
        pipe.hincrby(self._key(chunk_id), "times_used", 1)
        pipe.execute()

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, str]]:
        data = self.client.hgetall(self._key(chunk_id))
        return data or None #type: ignore

    def delete_chunk(self, chunk_id: str) -> None:
        self.client.delete(self._key(chunk_id))

    def get_all_chunks(self) -> Dict[str, Dict[str, str]]:
        out = {}
        for key in self.client.scan_iter(match="chunk:*"):
            cid = key.split("chunk:")[1]
            out[cid] = self.client.hgetall(key)
        return out

    def push_chunks(self, chunks: List["Chunk"]) -> List[str]:
        ids: List[str] = []
        now = self._now()
        pipe = self.client.pipeline()

        for chunk in chunks:
            # id_ = str(uuid.uuid4())
            id_ = hashlib.sha256(chunk.content.encode()).hexdigest()

            md = dict({})
            md.update({"last_used": now, "times_used": 0})
            pipe.hset(self._key(id_), mapping=md)
            ids.append(id_)

        pipe.execute()
        print(f"Cached {len(chunks)} chunks")
        return ids



