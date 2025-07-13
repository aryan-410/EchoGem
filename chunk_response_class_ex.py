# %%
from pydantic import BaseModel

# %%
class Chunk(BaseModel):
    title: str
    content: str
    keywords: list[str]
    named_entities: list[str]
    timestamp_range: str

class ChunkResponse(BaseModel):
    chunks: list[Chunk]


