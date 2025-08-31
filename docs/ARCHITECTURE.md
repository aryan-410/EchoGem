# EchoGem Architecture Overview

## 🏗️ System Architecture

EchoGem is built with a modular, layered architecture designed for scalability, maintainability, and extensibility. The system follows the **Separation of Concerns** principle, with each component handling a specific aspect of transcript processing.

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface (cli.py)  │  Graph Visualization (graphe.py) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                      │
├─────────────────────────────────────────────────────────────┤
│                    Processor (processor.py)                 │
│              Main workflow coordinator and API              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Processing Layer                    │
├─────────────────────────────────────────────────────────────┤
│  Chunker (chunker.py)  │  Vector Store (vector_store.py)   │
│  Semantic chunking     │  Pinecone integration            │
└─────────────────────────────────────────────────────────────┤
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Management Layer                     │
├─────────────────────────────────────────────────────────────┤
│ Usage Cache (usage_cache.py) │ Prompt-Answer Store        │
│ Performance tracking         │ (prompt_answer_store.py)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Services                        │
├─────────────────────────────────────────────────────────────┤
│  Google Gemini API  │  Pinecone Vector DB  │  Embeddings  │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Core Components

### 1. **Processor (processor.py)**
The central orchestrator that coordinates all operations.

**Responsibilities:**
- Manages the complete transcript processing workflow
- Coordinates between chunking, vector storage, and Q&A systems
- Handles API key validation and service initialization
- Provides a unified interface for all operations

**Key Methods:**
```python
class Processor:
    def process_transcript(self, file_path: str, options: Optional[ChunkingOptions] = None) -> ChunkResponse
    def query(self, question: str, options: Optional[QueryOptions] = None) -> QueryResult
    def pick_chunks(self, prompt: str, k: int = 5) -> Optional[List[Chunk]]
    def get_usage_statistics(self) -> Dict[str, Any]
```

### 2. **Chunker (chunker.py)**
Intelligent transcript segmentation using LLM-based semantic analysis.

**Responsibilities:**
- Analyzes transcript content for semantic boundaries
- Creates meaningful chunks based on content structure
- Maintains context between segments
- Provides fallback chunking for reliability

**Key Methods:**
```python
class Chunker:
    def chunk_transcript(self, transcript: str) -> List[Chunk]
    def _create_prompt(self, transcript: str) -> str
    def _parse_chunk_response(self, response: str) -> List[Chunk]
    def _fallback_chunking(self, transcript: str) -> List[Chunk]
    def get_embeddings(self, text: str) -> List[float]
```

### 3. **Vector Store (vector_store.py)**
Pinecone integration for efficient vector storage and retrieval.

**Responsibilities:**
- Stores transcript chunks as vector embeddings
- Performs similarity search for content retrieval
- Manages Pinecone index operations
- Handles batch operations for efficiency

**Key Methods:**
```python
class ChunkVectorDB:
    def add_chunks(self, chunks: List[Chunk]) -> bool
    def search_similar(self, query: str, k: int = 5) -> List[Chunk]
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]
    def delete_chunk(self, chunk_id: str) -> bool
    def clear_index(self) -> bool
```

### 4. **Prompt-Answer Store (prompt_answer_store.py)**
Stores and retrieves Q&A pairs for context-aware responses.

**Responsibilities:**
- Maintains history of questions and answers
- Links responses to source chunks
- Provides context for future queries
- Optimizes storage with vector embeddings

**Key Methods:**
```python
class PromptAnswerVectorDB:
    def add_pair(self, prompt: str, answer: str, chunk_ids: List[str]) -> str
    def search_similar_questions(self, question: str, k: int = 5) -> List[PAPair]
    def get_pair(self, pair_id: str) -> Optional[PAPair]
    def update_usage_count(self, pair_id: str) -> bool
```

### 5. **Usage Cache (usage_cache.py)**
Tracks usage patterns and performance metrics.

**Responsibilities:**
- Monitors chunk and response utilization
- Provides analytics and insights
- Optimizes caching strategies
- Tracks performance metrics

**Key Methods:**
```python
class UsageCache:
    def record_chunk_usage(self, chunk_id: str, prompt: str, score: float) -> None
    def record_response_usage(self, response_id: str, prompt: str) -> None
    def get_chunk_usage_stats(self, chunk_id: str) -> Dict[str, Any]
    def get_top_chunks(self, limit: int = 10) -> List[Tuple[str, int]]
    def clear_cache(self) -> None
```

### 6. **Graph Visualizer (graphe.py)**
Interactive visualization of information flow and relationships.

**Responsibilities:**
- Displays chunk relationships and connections
- Shows Q&A pair usage patterns
- Provides interactive exploration interface
- Visualizes system performance metrics

**Key Methods:**
```python
class GraphVisualizer:
    def __init__(self, processor: Processor)
    def run(self) -> None
    def _create_graph_data(self) -> Tuple[List[GraphNode], List[GraphEdge]]
    def _draw_graph(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> None
    def _handle_events(self) -> None
```

## 🔄 Data Flow

### Transcript Processing Flow
```
1. User Input → Processor.process_transcript()
2. File Reading → Text extraction and validation
3. Semantic Chunking → Chunker.chunk_transcript()
4. Vector Embedding → Sentence transformers
5. Storage → Vector Store + Usage Cache
6. Response → ChunkResponse with metadata
```

### Query Processing Flow
```
1. User Question → Processor.query()
2. Query Embedding → Vector similarity search
3. Chunk Retrieval → Top-k relevant chunks
4. Context Assembly → Chunk combination
5. LLM Generation → Google Gemini API
6. Response Storage → Prompt-Answer Store
7. Usage Tracking → Usage Cache update
8. Result Return → QueryResult with details
```

## 🏛️ Design Patterns

### 1. **Factory Pattern**
Used in the Processor class to create appropriate instances of services.

```python
def __init__(self, ...):
    self.chunker = Chunker(api_key=self.google_api_key)
    self.vector_db = ChunkVectorDB(...)
    self.usage_cache = UsageCache(...)
    self.pa_db = PromptAnswerStore(...)
```

### 2. **Strategy Pattern**
Different chunking strategies (LLM-based vs. fallback) can be selected based on context.

```python
def chunk_transcript(self, transcript: str) -> List[Chunk]:
    try:
        return self._llm_chunking(transcript)
    except Exception:
        return self._fallback_chunking(transcript)
```

### 3. **Observer Pattern**
Usage tracking observes and records all system activities for analytics.

```python
def record_chunk_usage(self, chunk_id: str, prompt: str, score: float):
    # Observer pattern for usage tracking
    self._update_usage_stats(chunk_id, score)
    self._log_usage_event(chunk_id, prompt, score)
```

### 4. **Command Pattern**
CLI commands are implemented as separate command classes for modularity.

```python
class ProcessCommand:
    def execute(self, args) -> None
class QueryCommand:
    def execute(self, args) -> None
class GraphCommand:
    def execute(self, args) -> None
```

## 🔐 Security & Configuration

### API Key Management
- Environment variable-based configuration
- Secure storage in `.pypirc` for PyPI operations
- No hardcoded credentials in source code

### Error Handling
- Comprehensive exception handling at all levels
- Graceful degradation for service failures
- User-friendly error messages

### Data Privacy
- Local processing of sensitive content
- Optional cloud storage with user control
- Secure transmission to external APIs

## 📊 Performance Characteristics

### Time Complexity
- **Chunking**: O(n) where n is transcript length
- **Vector Search**: O(log n) for similarity search
- **Q&A Generation**: O(1) for cached responses, O(n) for new queries

### Space Complexity
- **Memory**: O(k) where k is number of active chunks
- **Storage**: O(n) for transcript data, O(m) for embeddings
- **Cache**: O(u) where u is usage history size

### Scalability Features
- **Horizontal Scaling**: Multiple processor instances
- **Vertical Scaling**: Configurable chunk sizes and batch processing
- **Caching**: Multi-level caching for performance optimization

## 🔧 Configuration Options

### Chunking Configuration
```python
class ChunkingOptions:
    max_tokens: int = 2000
    similarity_threshold: float = 0.82
    coherence_threshold: float = 0.75
    show_chunks: bool = False
    show_metadata: bool = False
```

### Query Configuration
```python
class QueryOptions:
    max_chunks: int = 5
    similarity_threshold: float = 0.75
    show_chunks: bool = False
    show_metadata: bool = False
```

### System Configuration
```python
class ProcessorConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_index_name: str = "echogem-chunks"
    pa_index_name: str = "echogem-pa"
    usage_cache_path: str = "usage_cache_store.csv"
```

## 🚀 Extension Points

### Custom Chunking Strategies
```python
class CustomChunker(Chunker):
    def chunk_transcript(self, transcript: str) -> List[Chunk]:
        # Custom implementation
        pass
```

### Custom Vector Stores
```python
class CustomVectorStore:
    def add_chunks(self, chunks: List[Chunk]) -> bool:
        # Custom implementation
        pass
```

### Custom Usage Tracking
```python
class CustomUsageCache(UsageCache):
    def record_chunk_usage(self, chunk_id: str, prompt: str, score: float):
        # Custom implementation
        pass
```

## 🔍 Monitoring & Observability

### Metrics Collection
- API call counts and response times
- Chunk processing performance
- Vector search efficiency
- Cache hit rates

### Logging
- Structured logging for all operations
- Error tracking and debugging information
- Performance profiling data

### Health Checks
- Service availability monitoring
- API key validation
- Database connectivity checks

This architecture provides a solid foundation for EchoGem's current functionality while maintaining flexibility for future enhancements and customizations.
