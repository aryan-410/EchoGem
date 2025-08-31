"""
Data models for EchoGem library
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Chunk(BaseModel):
    """Represents a chunk of transcript text with metadata"""
    title: str = Field(..., description="2-5 word summary of the chunk")
    content: str = Field(..., description="The actual text content of the chunk")
    keywords: List[str] = Field(default_factory=list, description="Important terms from the chunk")
    named_entities: List[str] = Field(default_factory=list, description="Named entities mentioned in the chunk")
    timestamp_range: str = Field(default="", description="Estimated timestamp range (e.g., '00:00-01:30')")
    chunk_id: Optional[str] = Field(None, description="Unique identifier for the chunk")


class PAPair(BaseModel):
    """Represents a prompt-answer pair"""
    prompt: str = Field(..., description="The user's question or prompt")
    answer: str = Field(..., description="The generated answer")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks used to generate the answer")
    timestamp: Optional[str] = Field(None, description="When this Q&A was created")
    usage_count: int = Field(default=0, description="Number of times this pair has been used")


class ChunkResponse(BaseModel):
    """Response containing multiple chunks"""
    chunks: List[Chunk] = Field(..., description="List of chunks")


class QueryResult(BaseModel):
    """Result of a query with answer and metadata"""
    answer: str = Field(..., description="The generated answer")
    chunks_used: List[Chunk] = Field(..., description="Chunks used to generate the answer")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks used")
    confidence: float = Field(..., description="Confidence score of the answer")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ChunkingOptions(BaseModel):
    """Options for chunking transcripts"""
    max_tokens: int = Field(default=2000, description="Maximum tokens per chunk")
    similarity_threshold: float = Field(default=0.82, description="Similarity threshold for chunking")
    coherence_threshold: float = Field(default=0.75, description="Coherence threshold for chunking")
    output_chunks: bool = Field(default=False, description="Whether to output chunk details")


class QueryOptions(BaseModel):
    """Options for querying the system"""
    k: int = Field(default=5, description="Number of chunks to retrieve")
    entropy_weight: float = Field(default=0.25, description="Weight for entropy scoring")
    recency_weight: float = Field(default=0.25, description="Weight for recency scoring")
    show_chunks: bool = Field(default=False, description="Whether to show retrieved chunks")
    show_metadata: bool = Field(default=False, description="Whether to show chunk metadata")
