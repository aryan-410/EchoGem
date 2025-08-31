"""
EchoGem - Intelligent Transcript Processing and Question Answering Library

A powerful library for processing transcripts, chunking them intelligently,
and answering questions using Google Gemini and vector search.
"""

from .chunker import Chunker
from .vector_store import ChunkVectorDB
from .usage_cache import UsageCache
from .prompt_answer_store import PromptAnswerVectorDB, PAPair
from .processor import Processor
from .models import Chunk, ChunkResponse, QueryResult, ChunkingOptions, QueryOptions
from .graphe import GraphVisualizer

__version__ = "0.1.9"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "Chunker",
    "ChunkVectorDB",
    "UsageCache",
    "PromptAnswerVectorDB",
    "PAPair",
    "Processor",
    "Chunk",
    "ChunkResponse",
    "QueryResult",
    "ChunkingOptions",
    "QueryOptions",
    "GraphVisualizer"
]
