# EchoGem User Guide

## 🚀 Getting Started

### Installation

EchoGem is available on PyPI and can be installed with pip:

```bash
pip install echogem
```

### Quick Start

```python
from echogem import Processor

# Initialize the processor
processor = Processor()

# Process a transcript
response = processor.process_transcript("transcript.txt")

# Ask questions
result = processor.query("What is this transcript about?")
print(result.answer)
```

## 📋 Prerequisites

### Required Services

1. **Google Gemini API Key**
   - Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set as environment variable: `export GOOGLE_API_KEY="your_key"`

2. **Pinecone API Key**
   - Create account at [Pinecone](https://app.pinecone.io/)
   - Get your API key from the console
   - Set as environment variable: `export PINECONE_API_KEY="your_key"`

### Environment Setup

```bash
# Set your API keys
export GOOGLE_API_KEY="your_gemini_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"

# Optional: Customize EchoGem settings
export ECHOGEM_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export ECHOGEM_CHUNK_INDEX="echogem-chunks"
export ECHOGEM_PA_INDEX="echogem-pa"
```

## 🔧 Basic Usage

### 1. Processing Transcripts

#### Simple Processing
```python
from echogem import Processor

processor = Processor()
response = processor.process_transcript("transcript.txt")

if response.success:
    print(f"Successfully processed {response.num_chunks} chunks")
    print(f"Processing time: {response.processing_time:.2f} seconds")
else:
    print(f"Error: {response.error_message}")
```

#### Advanced Processing with Options
```python
from echogem import ChunkingOptions

options = ChunkingOptions(
    max_tokens=3000,           # Larger chunks
    similarity_threshold=0.85,  # Higher similarity requirement
    show_chunks=True,          # Display chunk details
    show_metadata=True         # Show metadata
)

response = processor.process_transcript("transcript.txt", options)

# Display chunk information
if options.show_chunks:
    for chunk in response.chunks:
        print(f"\nChunk: {chunk.title}")
        print(f"Keywords: {', '.join(chunk.keywords)}")
        print(f"Content: {chunk.content[:200]}...")
```

### 2. Asking Questions

#### Basic Question Answering
```python
# Simple question
result = processor.query("What is the main topic discussed?")

if result.success:
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Processing time: {result.processing_time:.2f}s")
```

#### Advanced Querying with Options
```python
from echogem import QueryOptions

options = QueryOptions(
    max_chunks=10,             # Use more chunks for context
    similarity_threshold=0.80,  # Lower similarity threshold
    show_chunks=True,          # Show which chunks were used
    show_metadata=True         # Show chunk metadata
)

result = processor.query("What are the key findings?", options)

# Display chunk usage information
if options.show_chunks:
    print(f"\nUsed {len(result.chunks_used)} chunks:")
    for chunk in result.chunks_used:
        print(f"- {chunk.title}: {chunk.content[:100]}...")
```

### 3. Retrieving Specific Chunks

```python
# Get most relevant chunks for a topic
chunks = processor.pick_chunks("machine learning algorithms", k=3)

if chunks:
    print(f"Found {len(chunks)} relevant chunks:")
    for chunk in chunks:
        print(f"\nTitle: {chunk.title}")
        print(f"Keywords: {', '.join(chunk.keywords)}")
        print(f"Content: {chunk.content[:150]}...")
```

### 4. Getting System Statistics

```python
# Get usage and performance statistics
stats = processor.get_usage_statistics()

print("System Statistics:")
print(f"Total chunks: {stats.get('total_chunks', 0)}")
print(f"Total queries: {stats.get('total_queries', 0)}")
print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2f}")
print(f"Average response time: {stats.get('avg_response_time', 0):.2f}s")
```

## 🎯 Command Line Interface

EchoGem provides a powerful command-line interface for easy use.

### Basic Commands

```bash
# Show help
py -m echogem.cli --help

# Process a transcript
py -m echogem.cli process transcript.txt

# Ask a question
py -m echogem.cli ask "What is this about?"

# Show system statistics
py -m echogem.cli stats

# Clear all data
py -m echogem.cli clear

# Visualize information graph
py -m echogem.cli graph
```

### Advanced CLI Usage

#### Process with Options
```bash
# Process with chunk details
py -m echogem.cli process transcript.txt --show-chunks --show-metadata

# Process with custom settings
py -m echogem.cli process transcript.txt --max-tokens 3000 --similarity-threshold 0.85
```

#### Query with Options
```bash
# Ask with chunk details
py -m echogem.cli ask "What are the key findings?" --show-chunks --show-metadata

# Use more context
py -m echogem.cli ask "Explain the methodology" --max-chunks 10
```

#### Interactive Mode
```bash
# Launch interactive CLI
py -m echogem.cli interactive
```

This opens an interactive menu where you can:
- Process transcripts
- Ask questions
- View statistics
- Explore the graph visualization
- Clear data

## 🎨 Graph Visualization

EchoGem includes an interactive graph visualization system.

### Launching the Visualizer

```python
from echogem import GraphVisualizer

visualizer = GraphVisualizer(processor)
visualizer.run()
```

### Using the Graph Interface

1. **Navigation**
   - **Mouse**: Click and drag to move around
   - **Scroll**: Zoom in/out
   - **Right-click**: Context menu

2. **Layouts**
   - **Force-directed**: Natural node positioning
   - **Circular**: Organized circular arrangement
   - **Hierarchical**: Tree-like structure

3. **Node Types**
   - **Blue nodes**: Transcript chunks
   - **Green nodes**: Q&A pairs
   - **Node size**: Indicates usage frequency

4. **Edge Types**
   - **Solid lines**: Direct relationships
   - **Dashed lines**: Indirect connections
   - **Line thickness**: Relationship strength

### Keyboard Shortcuts

- **F**: Toggle force-directed layout
- **C**: Toggle circular layout
- **H**: Toggle hierarchical layout
- **R**: Reset view
- **S**: Save graph image
- **ESC**: Exit visualization

## 🔧 Configuration

### Customizing EchoGem

#### Processor Configuration
```python
from echogem import Processor

# Custom embedding model
processor = Processor(
    embedding_model="all-mpnet-base-v2",  # Higher quality embeddings
    chunk_index_name="my-chunks",         # Custom index names
    pa_index_name="my-qa-pairs",
    usage_cache_path="my_cache.csv"
)

# Custom scoring weights
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

#### Chunking Configuration
```python
from echogem import ChunkingOptions

# Fine-tune chunking behavior
options = ChunkingOptions(
    max_tokens=2500,           # Larger chunks for complex content
    similarity_threshold=0.88,  # Higher similarity for better coherence
    coherence_threshold=0.80,   # Higher coherence requirement
    show_chunks=True,          # Always show chunk details
    show_metadata=True         # Always show metadata
)
```

#### Query Configuration
```python
from echogem import QueryOptions

# Optimize query processing
options = QueryOptions(
    max_chunks=8,              # Use more chunks for comprehensive answers
    similarity_threshold=0.70,  # Lower threshold for broader context
    show_chunks=True,          # Show chunk usage
    show_metadata=True         # Show metadata
)
```

### Environment Configuration

```bash
# EchoGem configuration
export ECHOGEM_EMBEDDING_MODEL="all-mpnet-base-v2"
export ECHOGEM_CHUNK_INDEX="echogem-chunks"
export ECHOGEM_PA_INDEX="echogem-pa"
export ECHOGEM_USAGE_CACHE="usage_cache.csv"

# Performance tuning
export ECHOGEM_MAX_WORKERS="4"
export ECHOGEM_BATCH_SIZE="100"
export ECHOGEM_CACHE_SIZE="1000"
```

## 📊 Performance Optimization

### Best Practices

1. **Chunk Size Optimization**
   - **Small chunks (1000-2000 tokens)**: Better for specific questions
   - **Large chunks (2000-4000 tokens)**: Better for comprehensive answers
   - **Very large chunks (4000+ tokens)**: May lose semantic coherence

2. **Similarity Thresholds**
   - **High threshold (0.85+)**: Very precise, fewer results
   - **Medium threshold (0.75-0.85)**: Balanced precision and recall
   - **Low threshold (0.70-)**: Broader context, more results

3. **Batch Processing**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   def process_file(file_path):
       return processor.process_transcript(file_path)
   
   files = ["transcript1.txt", "transcript2.txt", "transcript3.txt"]
   with ThreadPoolExecutor(max_workers=3) as executor:
       results = list(executor.map(process_file, files))
   ```

### Monitoring Performance

```python
# Get detailed performance metrics
stats = processor.get_usage_statistics()

print("Performance Metrics:")
print(f"Average chunking time: {stats.get('avg_chunking_time', 0):.2f}s")
print(f"Average query time: {stats.get('avg_query_time', 0):.2f}s")
print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2f}")
print(f"Memory usage: {stats.get('memory_usage_mb', 0):.1f} MB")
```

## 🚨 Troubleshooting

### Common Issues

#### 1. API Key Errors
```bash
Error: Google API key required. Set GOOGLE_API_KEY environment variable.
```

**Solution:**
```bash
export GOOGLE_API_KEY="your_actual_api_key"
echo $GOOGLE_API_KEY  # Verify it's set
```

#### 2. Pinecone Connection Issues
```bash
Error: Failed to connect to Pinecone
```

**Solution:**
- Verify your Pinecone API key
- Check your internet connection
- Ensure Pinecone service is available

#### 3. Memory Issues
```bash
Error: Out of memory during processing
```

**Solution:**
- Reduce chunk size: `max_tokens=1500`
- Process smaller files
- Increase system memory

#### 4. Slow Performance
**Symptoms:** Long processing times, slow responses

**Solutions:**
- Use smaller embedding models
- Reduce similarity thresholds
- Enable caching
- Use batch processing

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Process with verbose output
response = processor.process_transcript("transcript.txt")
```

### Getting Help

1. **Check the logs** for detailed error information
2. **Verify API keys** are correctly set
3. **Test with small files** first
4. **Check system resources** (memory, disk space)
5. **Review configuration** settings

## 🔄 Advanced Workflows

### 1. Multi-Document Processing

```python
import os
from pathlib import Path

def process_directory(directory_path):
    """Process all transcript files in a directory"""
    transcript_dir = Path(directory_path)
    results = {}
    
    for file_path in transcript_dir.glob("*.txt"):
        print(f"Processing {file_path.name}...")
        try:
            response = processor.process_transcript(str(file_path))
            results[file_path.name] = response
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    return results

# Process all transcripts
results = process_directory("./transcripts")
```

### 2. Incremental Processing

```python
def process_incrementally(file_path, existing_chunks):
    """Process only new content"""
    # Check if file was already processed
    if file_path in existing_chunks:
        print(f"File {file_path} already processed")
        return existing_chunks[file_path]
    
    # Process new file
    response = processor.process_transcript(file_path)
    existing_chunks[file_path] = response
    return response
```

### 3. Custom Chunking Strategies

```python
from echogem import Chunker

class CustomChunker(Chunker):
    def chunk_transcript(self, transcript: str):
        # Custom chunking logic
        if len(transcript) < 1000:
            # Small transcript: single chunk
            return [self._create_chunk(transcript, "Full Transcript")]
        else:
            # Large transcript: use parent logic
            return super().chunk_transcript(transcript)
    
    def _create_chunk(self, content, title):
        # Create chunk with custom logic
        pass

# Use custom chunker
custom_chunker = CustomChunker(api_key="your_key")
chunks = custom_chunker.chunk_transcript(transcript_text)
```

### 4. Integration with Other Systems

```python
def export_to_database(processor, file_path, db_connection):
    """Export processed chunks to external database"""
    response = processor.process_transcript(file_path)
    
    if response.success:
        for chunk in response.chunks:
            # Insert into database
            db_connection.execute("""
                INSERT INTO chunks (id, title, content, keywords, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                chunk.id,
                chunk.title,
                chunk.content,
                ','.join(chunk.keywords),
                str(chunk.metadata)
            ))
        
        db_connection.commit()
        print(f"Exported {len(response.chunks)} chunks to database")
```

## 📚 Examples

### Example 1: Academic Paper Analysis

```python
from echogem import Processor, ChunkingOptions

# Initialize with academic paper settings
processor = Processor()
options = ChunkingOptions(
    max_tokens=3000,           # Larger chunks for academic content
    similarity_threshold=0.85,  # High similarity for academic rigor
    show_chunks=True
)

# Process academic paper
response = processor.process_transcript("research_paper.txt", options)

# Ask academic questions
questions = [
    "What is the research methodology?",
    "What are the key findings?",
    "What are the limitations of this study?",
    "How does this relate to previous research?"
]

for question in questions:
    result = processor.query(question)
    print(f"\nQ: {question}")
    print(f"A: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
```

### Example 2: Meeting Transcript Analysis

```python
from echogem import Processor, QueryOptions

processor = Processor()

# Process meeting transcript
response = processor.process_transcript("meeting_transcript.txt")

# Meeting-specific queries
meeting_queries = [
    "What decisions were made?",
    "What action items were assigned?",
    "Who attended the meeting?",
    "What were the main discussion points?",
    "What are the next steps?"
]

options = QueryOptions(
    max_chunks=10,             # Use more context for meetings
    show_chunks=True,          # Show which parts were referenced
    show_metadata=True
)

for query in meeting_queries:
    result = processor.query(query, options)
    print(f"\nQ: {query}")
    print(f"A: {result.answer}")
    
    if options.show_chunks:
        print("Referenced sections:")
        for chunk in result.chunks_used:
            print(f"- {chunk.title}")
```

### Example 3: Podcast Transcript Analysis

```python
from echogem import Processor

processor = Processor()

# Process podcast transcript
response = processor.process_transcript("podcast_transcript.txt")

# Podcast-specific analysis
podcast_analysis = [
    "What is the main topic of this episode?",
    "Who are the guests and what are their backgrounds?",
    "What are the key insights shared?",
    "What questions were discussed?",
    "What are the main takeaways?"
]

for question in podcast_analysis:
    result = processor.query(question)
    print(f"\nQ: {question}")
    print(f"A: {result.answer}")
    print(f"Processing time: {result.processing_time:.2f}s")
```

## 🔮 Future Features

EchoGem is actively developed with planned features including:

- **Real-time processing** of live transcripts
- **Multi-language support** for international content
- **Advanced analytics** and insights
- **Web interface** for browser-based usage
- **API endpoints** for integration with other systems
- **Mobile applications** for on-the-go usage

## 📞 Support

For additional support and resources:

- **Documentation**: Check the docs folder for detailed guides
- **Examples**: Review the examples and demos folders
- **Issues**: Report bugs and request features on GitHub
- **Community**: Join discussions and share use cases

This user guide covers the essential aspects of using EchoGem. For more advanced topics and technical details, refer to the API Reference and Architecture documentation.
