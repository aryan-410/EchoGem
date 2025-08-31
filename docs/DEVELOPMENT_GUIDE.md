# EchoGem Development Guide

## 🚀 Getting Started with Development

### Prerequisites

- **Python 3.8+** (recommended: Python 3.11+)
- **Git** for version control
- **pip** for package management
- **Virtual environment** (venv or conda)

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/echogem.git
cd echogem

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

```bash
# Core development tools
pip install -e ".[dev]"

# Additional development tools
pip install pre-commit
pip install black[jupyter]
pip install isort[profile-black]
pip install mypy-extensions
```

## 🏗️ Project Structure

```
echogem/
├── echogem/                 # Main package
│   ├── __init__.py         # Package initialization
│   ├── chunker.py          # Transcript chunking logic
│   ├── vector_store.py     # Vector database operations
│   ├── prompt_answer_store.py # Q&A pair storage
│   ├── usage_cache.py      # Usage tracking
│   ├── processor.py        # Main orchestrator
│   ├── models.py           # Data models
│   ├── cli.py              # Command-line interface
│   └── graphe.py           # Graph visualization
├── tests/                   # Test suite
├── docs/                    # Documentation
├── examples/                # Basic examples
├── demos/                   # Comprehensive demonstrations
├── legacy/                  # Development history
├── setup.py                 # Package setup
├── pyproject.toml          # Project configuration
├── requirements.txt         # Dependencies
└── README.md               # Project overview
```

## 🔧 Development Tools

### Code Quality Tools

#### Black (Code Formatter)
```bash
# Format all Python files
black echogem/ tests/ examples/ demos/

# Check formatting without changes
black --check echogem/ tests/ examples/ demos/

# Format specific file
black echogem/processor.py
```

#### isort (Import Sorter)
```bash
# Sort imports in all files
isort echogem/ tests/ examples/ demos/

# Check import sorting
isort --check-only echogem/ tests/ examples/ demos/
```

#### Flake8 (Linter)
```bash
# Run linting
flake8 echogem/ tests/ examples/ demos/

# Run with specific configuration
flake8 --config .flake8 echogem/
```

#### MyPy (Type Checker)
```bash
# Run type checking
mypy echogem/

# Run with specific configuration
mypy --config-file mypy.ini echogem/
```

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before commits:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

### Testing

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=echogem

# Run specific test file
pytest tests/test_processor.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

#### Test Structure
```
tests/
├── conftest.py             # Test configuration and fixtures
├── test_processor.py       # Processor class tests
├── test_chunker.py         # Chunker class tests
├── test_vector_store.py    # Vector store tests
├── test_models.py          # Data model tests
├── test_cli.py             # CLI interface tests
└── test_integration.py     # Integration tests
```

#### Writing Tests
```python
import pytest
from echogem import Processor, Chunk

class TestProcessor:
    def test_processor_initialization(self):
        """Test Processor initialization with default parameters"""
        processor = Processor()
        assert processor is not None
        assert processor.google_api_key is not None
    
    def test_process_transcript_success(self, sample_transcript):
        """Test successful transcript processing"""
        processor = Processor()
        response = processor.process_transcript(sample_transcript)
        
        assert response.success is True
        assert response.num_chunks > 0
        assert len(response.chunks) == response.num_chunks
    
    def test_process_transcript_invalid_file(self):
        """Test transcript processing with invalid file"""
        processor = Processor()
        response = processor.process_transcript("nonexistent.txt")
        
        assert response.success is False
        assert response.error_message is not None

@pytest.fixture
def sample_transcript():
    """Sample transcript for testing"""
    return """
    This is a sample transcript for testing purposes.
    It contains multiple sentences and should be chunked appropriately.
    The chunking algorithm should identify semantic boundaries.
    """
```

## 📝 Code Style Guidelines

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these EchoGem-specific additions:

#### Naming Conventions
```python
# Classes: PascalCase
class TranscriptProcessor:
    pass

# Functions and methods: snake_case
def process_transcript():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CHUNK_SIZE = 2000
DEFAULT_SIMILARITY_THRESHOLD = 0.82

# Private methods: leading underscore
def _internal_helper():
    pass
```

#### Import Organization
```python
# Standard library imports
import os
import sys
from typing import List, Optional, Dict, Any

# Third-party imports
import numpy as np
import pinecone
from sentence_transformers import SentenceTransformer

# Local imports
from .models import Chunk, ChunkResponse
from .utils import validate_file_path
```

#### Docstrings
```python
def process_transcript(
    self, 
    file_path: str, 
    options: Optional[ChunkingOptions] = None
) -> ChunkResponse:
    """
    Process a transcript file and create intelligent chunks.
    
    Args:
        file_path: Path to the transcript file to process
        options: Optional configuration for chunking behavior
        
    Returns:
        ChunkResponse object containing processing results
        
    Raises:
        FileNotFoundError: If the transcript file doesn't exist
        ValueError: If the file is empty or invalid
        
    Example:
        >>> processor = Processor()
        >>> response = processor.process_transcript("transcript.txt")
        >>> print(f"Created {response.num_chunks} chunks")
    """
    pass
```

#### Type Hints
```python
from typing import List, Optional, Dict, Any, Tuple

def search_similar_chunks(
    self, 
    query: str, 
    k: int = 5
) -> List[Chunk]:
    """Search for chunks similar to the query."""
    pass

def get_usage_statistics(self) -> Dict[str, Any]:
    """Get system usage statistics."""
    pass
```

### Error Handling

#### Exception Types
```python
class EchoGemError(Exception):
    """Base exception for EchoGem."""
    pass

class ChunkingError(EchoGemError):
    """Raised when chunking fails."""
    pass

class VectorStoreError(EchoGemError):
    """Raised when vector store operations fail."""
    pass

class APIError(EchoGemError):
    """Raised when external API calls fail."""
    pass
```

#### Error Handling Patterns
```python
def process_transcript(self, file_path: str) -> ChunkResponse:
    """Process transcript with comprehensive error handling."""
    try:
        # Validate input
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            raise ValueError("Transcript file is empty")
        
        # Process content
        chunks = self.chunker.chunk_transcript(content)
        
        return ChunkResponse(
            success=True,
            num_chunks=len(chunks),
            chunks=chunks,
            processing_time=time.time() - start_time,
            file_path=file_path
        )
        
    except FileNotFoundError as e:
        return ChunkResponse(
            success=False,
            num_chunks=0,
            chunks=[],
            processing_time=0,
            file_path=file_path,
            error_message=str(e)
        )
    except Exception as e:
        return ChunkResponse(
            success=False,
            num_chunks=0,
            chunks=[],
            processing_time=0,
            file_path=file_path,
            error_message=f"Unexpected error: {str(e)}"
        )
```

## 🔄 Development Workflow

### Git Workflow

#### Branch Naming
```bash
# Feature branches
feature/add-custom-chunking
feature/enhance-graph-visualization
feature/add-batch-processing

# Bug fix branches
fix/memory-leak-in-vector-store
fix/cli-argument-parsing-issue

# Documentation branches
docs/update-api-reference
docs/add-performance-guide
```

#### Commit Messages
```bash
# Format: <type>(<scope>): <description>
feat(processor): add batch processing capability
fix(vector-store): resolve memory leak in large datasets
docs(api): update method documentation with examples
test(chunker): add comprehensive test coverage
refactor(cli): simplify command structure
```

#### Pull Request Process
1. **Create feature branch** from `main`
2. **Make changes** following coding standards
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Run quality checks** locally
6. **Create pull request** with detailed description
7. **Address review comments** and iterate
8. **Merge** after approval

### Testing Strategy

#### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test system performance
5. **Error Handling Tests**: Test error conditions

#### Test Coverage Goals
- **Minimum coverage**: 80%
- **Target coverage**: 90%+
- **Critical paths**: 100%

#### Test Data Management
```python
# Use fixtures for test data
@pytest.fixture
def sample_transcript():
    return "Sample transcript content for testing."

@pytest.fixture
def mock_processor():
    return Processor(
        google_api_key="test_key",
        pinecone_api_key="test_key"
    )

# Use temporary files for file operations
@pytest.fixture
def temp_transcript_file(tmp_path):
    file_path = tmp_path / "test_transcript.txt"
    file_path.write_text("Test transcript content")
    return str(file_path)
```

## 🚀 Adding New Features

### Feature Development Process

1. **Design Phase**
   - Define requirements and specifications
   - Design API and data structures
   - Plan testing strategy

2. **Implementation Phase**
   - Implement core functionality
   - Add comprehensive tests
   - Update documentation

3. **Review Phase**
   - Code review and testing
   - Performance validation
   - Security review

4. **Integration Phase**
   - Merge to main branch
   - Update package version
   - Release notes

### Example: Adding Custom Chunking Strategy

#### 1. Define Interface
```python
from abc import ABC, abstractmethod
from typing import List
from .models import Chunk

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, transcript: str) -> List[Chunk]:
        """Chunk transcript using this strategy."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass
```

#### 2. Implement Strategy
```python
class SemanticChunkingStrategy(ChunkingStrategy):
    """Semantic chunking using LLM analysis."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def chunk(self, transcript: str) -> List[Chunk]:
        # Implementation using LLM
        pass
    
    def get_name(self) -> str:
        return "semantic"

class RuleBasedChunkingStrategy(ChunkingStrategy):
    """Rule-based chunking using predefined rules."""
    
    def chunk(self, transcript: str) -> List[Chunk]:
        # Implementation using rules
        pass
    
    def get_name(self) -> str:
        return "rule-based"
```

#### 3. Update Chunker Class
```python
class Chunker:
    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy
    
    def chunk_transcript(self, transcript: str) -> List[Chunk]:
        return self.strategy.chunk(transcript)
    
    def set_strategy(self, strategy: ChunkingStrategy):
        """Change chunking strategy at runtime."""
        self.strategy = strategy
```

#### 4. Add Tests
```python
def test_chunking_strategy_interface():
    """Test chunking strategy interface."""
    strategy = SemanticChunkingStrategy("test_key")
    assert hasattr(strategy, 'chunk')
    assert hasattr(strategy, 'get_name')
    assert callable(strategy.chunk)
    assert callable(strategy.get_name)

def test_chunker_with_strategy():
    """Test chunker with different strategies."""
    semantic_strategy = SemanticChunkingStrategy("test_key")
    rule_strategy = RuleBasedChunkingStrategy()
    
    chunker = Chunker(semantic_strategy)
    chunks = chunker.chunk_transcript("Test transcript")
    
    # Test strategy switching
    chunker.set_strategy(rule_strategy)
    chunks2 = chunker.chunk_transcript("Test transcript")
    
    assert len(chunks) > 0
    assert len(chunks2) > 0
```

#### 5. Update Documentation
```markdown
## Chunking Strategies

EchoGem supports multiple chunking strategies:

### Semantic Chunking
Uses LLM analysis to identify semantic boundaries.

### Rule-Based Chunking
Uses predefined rules for consistent chunking.

### Custom Strategies
Implement your own chunking strategy by extending `ChunkingStrategy`.
```

## 🔍 Debugging and Profiling

### Debugging Tools

#### Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def process_transcript(self, file_path: str):
    logger.debug(f"Starting transcript processing: {file_path}")
    logger.info(f"Processing file: {file_path}")
    
    try:
        # Processing logic
        logger.debug("Chunking transcript...")
        chunks = self.chunker.chunk_transcript(content)
        logger.info(f"Created {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error processing transcript: {e}", exc_info=True)
        raise
```

#### Debug Mode
```python
class Processor:
    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def process_transcript(self, file_path: str):
        if self.debug:
            print(f"Debug: Processing {file_path}")
            print(f"Debug: File size: {os.path.getsize(file_path)} bytes")
```

### Profiling

#### Performance Profiling
```python
import cProfile
import pstats
from pstats import SortKey

def profile_function(func, *args, **kwargs):
    """Profile a function and print statistics."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(10)  # Top 10 functions
    
    return result

# Usage
result = profile_function(processor.process_transcript, "transcript.txt")
```

#### Memory Profiling
```python
import tracemalloc

def profile_memory(func, *args, **kwargs):
    """Profile memory usage of a function."""
    tracemalloc.start()
    
    result = func(*args, **kwargs)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
    return result
```

## 📦 Packaging and Distribution

### Building the Package

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build source distribution
python -m build --sdist

# Build wheel
python -m build --wheel

# Build both
python -m build
```

### Testing the Build

```bash
# Install from built package
pip install dist/echogem-0.1.0-py3-none-any.whl

# Test installation
python -c "import echogem; print(echogem.__version__)"

# Test CLI
echogem --help
```

### Version Management

#### Semantic Versioning
```python
# echogem/__init__.py
__version__ = "0.1.0"

# setup.py
setup(
    name="echogem",
    version="0.1.0",
    # ...
)
```

#### Version Bumping
```bash
# Patch version (bug fixes)
bump2version patch

# Minor version (new features)
bump2version minor

# Major version (breaking changes)
bump2version major
```

## 🧪 Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest --cov=echogem --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## 📚 Documentation

### Documentation Standards

#### Code Documentation
- **All public functions** must have docstrings
- **Use Google style** docstrings
- **Include examples** for complex functions
- **Document exceptions** and error conditions

#### API Documentation
- **Comprehensive coverage** of all public APIs
- **Usage examples** for common scenarios
- **Parameter descriptions** with types and constraints
- **Return value documentation** with examples

#### User Documentation
- **Getting started** guides
- **Tutorial examples** for common use cases
- **Troubleshooting** guides
- **Performance optimization** tips

### Documentation Tools

```bash
# Generate API documentation
sphinx-apidoc -o docs/source echogem/

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## 🔒 Security Considerations

### API Key Security
- **Never commit** API keys to version control
- **Use environment variables** for configuration
- **Validate API keys** before use
- **Implement rate limiting** for external APIs

### Input Validation
```python
def validate_file_path(file_path: str) -> str:
    """Validate and sanitize file path."""
    if not file_path:
        raise ValueError("File path cannot be empty")
    
    # Normalize path
    normalized_path = os.path.normpath(file_path)
    
    # Check for path traversal attempts
    if ".." in normalized_path:
        raise ValueError("Invalid file path")
    
    return normalized_path
```

### Data Privacy
- **Process sensitive data** locally when possible
- **Implement data retention** policies
- **Provide data deletion** capabilities
- **Log minimal sensitive information**

## 🚀 Performance Optimization

### Optimization Strategies

#### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_embedding_model(model_name: str):
    """Cache embedding model instances."""
    return SentenceTransformer(model_name)
```

#### Batch Processing
```python
def process_chunks_batch(self, chunks: List[Chunk], batch_size: int = 100):
    """Process chunks in batches for efficiency."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        self._process_batch(batch)
```

#### Async Processing
```python
import asyncio

async def process_transcript_async(self, file_path: str):
    """Process transcript asynchronously."""
    content = await self._read_file_async(file_path)
    chunks = await self._chunk_content_async(content)
    return chunks
```

### Performance Monitoring

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"{operation_name} took {elapsed:.2f} seconds")

# Usage
with timer("Transcript processing"):
    response = processor.process_transcript("transcript.txt")
```

## 🤝 Contributing Guidelines

### Contribution Process

1. **Fork the repository**
2. **Create feature branch**
3. **Make changes** following coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Run quality checks** locally
7. **Submit pull request** with detailed description

### Code Review Checklist

- [ ] **Code follows** style guidelines
- [ ] **Tests pass** and cover new functionality
- [ ] **Documentation updated** for new features
- [ ] **Error handling** implemented appropriately
- [ ] **Performance impact** considered
- [ ] **Security implications** reviewed
- [ ] **Backward compatibility** maintained

### Communication

- **GitHub Issues** for bug reports and feature requests
- **Pull Request discussions** for code review
- **GitHub Discussions** for general questions
- **Email** for security issues

This development guide provides comprehensive coverage of development practices for EchoGem. Follow these guidelines to ensure code quality, maintainability, and successful contributions to the project.
