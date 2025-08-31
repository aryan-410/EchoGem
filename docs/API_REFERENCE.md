# EchoGem API Reference

## 📖 Overview

This document provides comprehensive API documentation for EchoGem, including all public classes, methods, and data models. The API is designed to be intuitive and follows Python best practices.

## 🏗️ Core Classes

### Processor

The main orchestrator class that coordinates all EchoGem operations.

```python
class Processor:
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        weights: Optional[List[float]] = None,
        chunk_index_name: str = "echogem-chunks",
        pa_index_name: str = "echogem-pa",
        usage_cache_path: str = "usage_cache_store.csv"
    )
```

**Parameters:**
- `google_api_key`: Google Gemini API key (defaults to `GOOGLE_API_KEY` env var)
- `pinecone_api_key`: Pinecone API key (defaults to `PINECONE_API_KEY` env var)
- `embedding_model`: Sentence transformer model name (default: "all-MiniLM-L6-v2")
- `weights`: Scoring weights for chunk selection (default: [1.0] * 7)
- `chunk_index_name`: Pinecone index name for chunks
- `pa_index_name`: Pinecone index name for Q&A pairs
- `usage_cache_path`: Path to usage cache CSV file

**Example:**
```python
from echogem import Processor

# Initialize with environment variables
processor = Processor()

# Initialize with custom configuration
processor = Processor(
    embedding_model="all-mpnet-base-v2",
    chunk_index_name="my-chunks",
    weights=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
)
```

#### Methods

##### `process_transcript()`
Processes a transcript file and creates intelligent chunks.

```python
def process_transcript(
    self, 
    file_path: str, 
    options: Optional[ChunkingOptions] = None
) -> ChunkResponse
```

**Parameters:**
- `file_path`: Path to transcript file
- `options`: Optional chunking configuration

**Returns:** `ChunkResponse` object with processing results

**Example:**
```python
# Basic processing
response = processor.process_transcript("transcript.txt")

# With custom options
options = ChunkingOptions(
    max_tokens=3000,
    show_chunks=True,
    show_metadata=True
)
response = processor.process_transcript("transcript.txt", options)

print(f"Created {response.num_chunks} chunks")
print(f"Processing time: {response.processing_time:.2f}s")
```

##### `query()`
Answers questions based on processed transcript content.

```python
def query(
    self, 
    question: str, 
    options: Optional[QueryOptions] = None
) -> QueryResult
```

**Parameters:**
- `question`: User's question or query
- `options`: Optional query configuration

**Returns:** `QueryResult` object with answer and metadata

**Example:**
```python
# Basic query
result = processor.query("What is the main topic discussed?")

# With custom options
options = QueryOptions(
    max_chunks=10,
    show_chunks=True,
    show_metadata=True
)
result = processor.query("What are the key findings?", options)

print(f"Answer: {result.answer}")
print(f"Used {len(result.chunks_used)} chunks")
print(f"Confidence: {result.confidence:.2f}")
```

##### `pick_chunks()`
Retrieves the most relevant chunks for a given prompt.

```python
def pick_chunks(self, prompt: str, k: int = 5) -> Optional[List[Chunk]]
```

**Parameters:**
- `prompt`: Text prompt for chunk selection
- `k`: Number of chunks to retrieve

**Returns:** List of relevant `Chunk` objects or `None`

**Example:**
```python
chunks = processor.pick_chunks("machine learning algorithms", k=3)
if chunks:
    for chunk in chunks:
        print(f"Chunk: {chunk.title}")
        print(f"Content: {chunk.content[:100]}...")
```

##### `get_usage_statistics()`
Retrieves system usage statistics and performance metrics.

```python
def get_usage_statistics(self) -> Dict[str, Any]
```

**Returns:** Dictionary containing usage statistics

**Example:**
```python
stats = processor.get_usage_statistics()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Total queries: {stats['total_queries']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}")
```

### Chunker

Intelligent transcript segmentation using LLM-based semantic analysis.

```python
class Chunker:
    def __init__(
        self,
        api_key: Optional[str] = None,
        embed_model: str = "all-MiniLM-L6-v2",
        max_tokens: int = 2000,
        similarity_threshold: float = 0.82,
        coherence_threshold: float = 0.75,
    )
```

**Parameters:**
- `api_key`: Google Gemini API key
- `embed_model`: Sentence transformer model name
- `max_tokens`: Maximum tokens per chunk
- `similarity_threshold`: Threshold for chunk similarity
- `coherence_threshold`: Threshold for chunk coherence

#### Methods

##### `chunk_transcript()`
Creates intelligent chunks from transcript text.

```python
def chunk_transcript(self, transcript: str) -> List[Chunk]
```

**Parameters:**
- `transcript`: Raw transcript text

**Returns:** List of `Chunk` objects

**Example:**
```python
from echogem import Chunker

chunker = Chunker(api_key="your_api_key")
chunks = chunker.chunk_transcript(transcript_text)

for chunk in chunks:
    print(f"Title: {chunk.title}")
    print(f"Keywords: {', '.join(chunk.keywords)}")
    print(f"Content: {chunk.content[:200]}...")
```

### ChunkVectorDB

Pinecone integration for vector storage and retrieval.

```python
class ChunkVectorDB:
    def __init__(
        self,
        embedding_model,
        api_key: Optional[str] = None,
        index_name: str = "echogem-chunks",
        region: str = "us-east-1",
        dimension: int = 384
    )
```

**Parameters:**
- `embedding_model`: Sentence transformer model
- `api_key`: Pinecone API key
- `index_name`: Pinecone index name
- `region`: Pinecone region
- `dimension`: Vector dimension

#### Methods

##### `add_chunks()`
Adds chunks to the vector database.

```python
def add_chunks(self, chunks: List[Chunk]) -> bool
```

**Parameters:**
- `chunks`: List of `Chunk` objects to add

**Returns:** `True` if successful, `False` otherwise

**Example:**
```python
from echogem import ChunkVectorDB

vector_db = ChunkVectorDB(embedding_model, api_key="your_key")
success = vector_db.add_chunks(chunks)
if success:
    print("Chunks added successfully")
```

##### `search_similar()`
Searches for similar chunks using vector similarity.

```python
def search_similar(self, query: str, k: int = 5) -> List[Chunk]
```

**Parameters:**
- `query`: Search query text
- `k`: Number of results to return

**Returns:** List of similar `Chunk` objects

**Example:**
```python
similar_chunks = vector_db.search_similar("artificial intelligence", k=3)
for chunk in similar_chunks:
    print(f"Similarity: {chunk.similarity_score:.3f}")
    print(f"Content: {chunk.content[:100]}...")
```

### PromptAnswerVectorDB

Stores and retrieves Q&A pairs with vector embeddings.

```python
class PromptAnswerVectorDB:
    def __init__(
        self,
        embedding_model,
        api_key: Optional[str] = None,
        index_name: str = "echogem-pa",
        region: str = "us-east-1",
        dimension: int = 768,
        namespace: str = "pa_pairs",
        use_prompt_plus_answer: bool = True
    )
```

**Parameters:**
- `embedding_model`: Sentence transformer model
- `api_key`: Pinecone API key
- `index_name`: Pinecone index name
- `region`: Pinecone region
- `dimension`: Vector dimension
- `namespace`: Pinecone namespace
- `use_prompt_plus_answer`: Whether to embed prompt + answer together

#### Methods

##### `add_pair()`
Adds a new Q&A pair to the database.

```python
def add_pair(self, prompt: str, answer: str, chunk_ids: List[str]) -> str
```

**Parameters:**
- `prompt`: User's question
- `answer`: Generated answer
- `chunk_ids`: IDs of chunks used

**Returns:** Unique identifier for the pair

**Example:**
```python
from echogem import PromptAnswerVectorDB

pa_db = PromptAnswerVectorDB(embedding_model, api_key="your_key")
pair_id = pa_db.add_pair(
    "What is machine learning?",
    "Machine learning is a subset of AI...",
    ["chunk_001", "chunk_002"]
)
print(f"Added pair with ID: {pair_id}")
```

##### `search_similar_questions()`
Finds similar questions in the database.

```python
def search_similar_questions(self, question: str, k: int = 5) -> List[PAPair]
```

**Parameters:**
- `question`: Question to search for
- `k`: Number of results to return

**Returns:** List of similar `PAPair` objects

**Example:**
```python
similar_questions = pa_db.search_similar_questions("What is AI?", k=3)
for pair in similar_questions:
    print(f"Question: {pair.prompt}")
    print(f"Answer: {pair.answer[:100]}...")
```

### UsageCache

Tracks usage patterns and performance metrics.

```python
class UsageCache:
    def __init__(self, csv_path: str = "usage_cache_store.csv")
```

**Parameters:**
- `csv_path`: Path to CSV cache file

#### Methods

##### `record_chunk_usage()`
Records when a chunk is used for answering a query.

```python
def record_chunk_usage(self, chunk_id: str, prompt: str, score: float) -> None
```

**Parameters:**
- `chunk_id`: ID of the chunk used
- `prompt`: Query prompt
- `score`: Relevance score

**Example:**
```python
from echogem import UsageCache

cache = UsageCache()
cache.record_chunk_usage("chunk_001", "What is ML?", 0.95)
```

##### `get_chunk_usage_stats()`
Retrieves usage statistics for a specific chunk.

```python
def get_chunk_usage_stats(self, chunk_id: str) -> Dict[str, Any]
```

**Parameters:**
- `chunk_id`: ID of the chunk

**Returns:** Dictionary with usage statistics

**Example:**
```python
stats = cache.get_chunk_usage_stats("chunk_001")
print(f"Usage count: {stats['usage_count']}")
print(f"Average score: {stats['average_score']:.3f}")
print(f"Last used: {stats['last_used']}")
```

### GraphVisualizer

Interactive visualization of information flow and relationships.

```python
class GraphVisualizer:
    def __init__(self, processor: Processor)
```

**Parameters:**
- `processor`: Initialized Processor instance

#### Methods

##### `run()`
Starts the interactive graph visualization.

```python
def run(self) -> None
```

**Example:**
```python
from echogem import GraphVisualizer

visualizer = GraphVisualizer(processor)
visualizer.run()  # Opens interactive Pygame window
```

## 📊 Data Models

### Chunk

Represents a segment of transcript content.

```python
class Chunk(BaseModel):
    id: str = Field(..., description="Unique chunk identifier")
    title: str = Field(..., description="Chunk title/summary")
    content: str = Field(..., description="Chunk content text")
    keywords: List[str] = Field(default_factory=list, description="Key terms")
    named_entities: List[str] = Field(default_factory=list, description="Named entities")
    timestamp_range: Optional[str] = Field(None, description="Time range")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
```

**Example:**
```python
from echogem import Chunk

chunk = Chunk(
    id="chunk_001",
    title="Introduction to AI",
    content="Artificial Intelligence is a field of computer science...",
    keywords=["AI", "computer science", "intelligence"],
    named_entities=["Alan Turing", "MIT"],
    timestamp_range="00:00-02:30"
)
```

### ChunkResponse

Response from transcript processing operation.

```python
class ChunkResponse(BaseModel):
    success: bool = Field(..., description="Operation success status")
    num_chunks: int = Field(..., description="Number of chunks created")
    chunks: List[Chunk] = Field(default_factory=list, description="Created chunks")
    processing_time: float = Field(..., description="Processing time in seconds")
    file_path: str = Field(..., description="Processed file path")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    show_chunks: bool = Field(default=False, description="Whether to show chunks")
    show_metadata: bool = Field(default=False, description="Whether to show metadata")
```

**Example:**
```python
from echogem import ChunkResponse

response = ChunkResponse(
    success=True,
    num_chunks=15,
    chunks=chunks,
    processing_time=2.45,
    file_path="transcript.txt"
)
```

### QueryResult

Result from a question-answering operation.

```python
class QueryResult(BaseModel):
    success: bool = Field(..., description="Operation success status")
    question: str = Field(..., description="User's question")
    answer: str = Field(..., description="Generated answer")
    chunks_used: List[Chunk] = Field(default_factory=list, description="Chunks used")
    confidence: float = Field(..., description="Answer confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    show_chunks: bool = Field(default=False, description="Whether to show chunks")
    show_metadata: bool = Field(default=False, description="Whether to show metadata")
```

**Example:**
```python
from echogem import QueryResult

result = QueryResult(
    success=True,
    question="What is the main topic?",
    answer="The main topic is artificial intelligence...",
    chunks_used=used_chunks,
    confidence=0.87,
    processing_time=1.23
)
```

### PAPair

Represents a prompt-answer pair.

```python
class PAPair(BaseModel):
    prompt: str = Field(..., description="The user's question or prompt")
    answer: str = Field(..., description="The generated answer")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks used")
    timestamp: Optional[str] = Field(None, description="When this Q&A was created")
    usage_count: int = Field(default=0, description="Number of times this pair has been used")
```

**Example:**
```python
from echogem import PAPair

pair = PAPair(
    prompt="What is machine learning?",
    answer="Machine learning is a subset of AI...",
    chunk_ids=["chunk_001", "chunk_002"],
    timestamp="2025-08-31T10:30:00Z",
    usage_count=5
)
```

### ChunkingOptions

Configuration options for transcript chunking.

```python
class ChunkingOptions(BaseModel):
    max_tokens: int = Field(default=2000, description="Maximum tokens per chunk")
    similarity_threshold: float = Field(default=0.82, description="Similarity threshold")
    coherence_threshold: float = Field(default=0.75, description="Coherence threshold")
    show_chunks: bool = Field(default=False, description="Whether to show chunks")
    show_metadata: bool = Field(default=False, description="Whether to show metadata")
```

**Example:**
```python
from echogem import ChunkingOptions

options = ChunkingOptions(
    max_tokens=3000,
    similarity_threshold=0.85,
    show_chunks=True
)
```

### QueryOptions

Configuration options for query processing.

```python
class QueryOptions(BaseModel):
    max_chunks: int = Field(default=5, description="Maximum chunks to use")
    similarity_threshold: float = Field(default=0.75, description="Similarity threshold")
    show_chunks: bool = Field(default=False, description="Whether to show chunks")
    show_metadata: bool = Field(default=False, description="Whether to show metadata")
```

**Example:**
```python
from echogem import QueryOptions

options = QueryOptions(
    max_chunks=10,
    similarity_threshold=0.80,
    show_chunks=True
)
```

## 🔧 Configuration

### Environment Variables

EchoGem uses environment variables for configuration:

```bash
# Required
export GOOGLE_API_KEY="your_gemini_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"

# Optional
export ECHOGEM_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export ECHOGEM_CHUNK_INDEX="echogem-chunks"
export ECHOGEM_PA_INDEX="echogem-pa"
```

### Default Values

```python
# Default configuration
DEFAULT_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_index_name": "echogem-chunks",
    "pa_index_name": "echogem-pa",
    "usage_cache_path": "usage_cache_store.csv",
    "max_tokens": 2000,
    "similarity_threshold": 0.82,
    "coherence_threshold": 0.75,
    "max_chunks": 5,
    "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}
```

## 🚀 Advanced Usage

### Custom Embedding Models

```python
from sentence_transformers import SentenceTransformer

# Use custom model
custom_model = SentenceTransformer("all-mpnet-base-v2")
processor = Processor(embedding_model=custom_model)
```

### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    return processor.process_transcript(file_path)

files = ["transcript1.txt", "transcript2.txt", "transcript3.txt"]
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_file, files))
```

### Custom Scoring Weights

```python
# Customize chunk selection weights
weights = [
    0.4,  # Content relevance
    0.2,  # Keyword match
    0.2,  # Named entity match
    0.1,  # Usage frequency
    0.05, # Recency
    0.03, # Length
    0.02  # Metadata richness
]

processor = Processor(weights=weights)
```

### Error Handling

```python
try:
    response = processor.process_transcript("transcript.txt")
    if response.success:
        print(f"Processed {response.num_chunks} chunks")
    else:
        print(f"Error: {response.error_message}")
except Exception as e:
    print(f"Exception occurred: {e}")
```

This API reference provides comprehensive coverage of all public interfaces in EchoGem. For additional examples and use cases, refer to the demos and examples folders.
