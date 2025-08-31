# Legacy: Old Testing Approaches (v0.1.0-rc9)
# Different testing strategies and frameworks tested
# Replaced by current pytest-based testing in v0.2.0

import unittest
import doctest
import sys
import os
from typing import List, Dict, Any

# Approach 1: Simple Assert Testing
class SimpleTestChunker:
    """Simple test without framework"""
    
    def test_chunking(self):
        """Test basic chunking functionality"""
        # This was the first testing approach - just assertions
        text = "This is a test transcript. It has multiple sentences. Each should be a chunk."
        chunks = self.chunk_text(text)
        
        # Simple assertions
        assert len(chunks) > 0, "Should create chunks"
        assert all(len(chunk) > 0 for chunk in chunks), "Chunks should not be empty"
        assert len(chunks) >= 2, "Should create multiple chunks"
        
        print("✅ Basic chunking test passed")
    
    def chunk_text(self, text: str) -> List[str]:
        """Simple chunking for testing"""
        return text.split('. ')

# Problems: No test discovery, no reporting, manual execution

# Approach 2: Unittest Framework
class TestChunker(unittest.TestCase):
    """Unittest-based testing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_text = "First sentence. Second sentence. Third sentence."
        self.chunker = SimpleChunker()
    
    def test_basic_chunking(self):
        """Test basic chunking functionality"""
        chunks = self.chunker.chunk_text(self.test_text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_chunk_content(self):
        """Test chunk content"""
        chunks = self.chunker.chunk_text(self.test_text)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)
    
    def test_chunk_count(self):
        """Test number of chunks created"""
        chunks = self.chunker.chunk_text(self.test_text)
        self.assertGreaterEqual(len(chunks), 2)
    
    def tearDown(self):
        """Clean up after tests"""
        pass

# Problems: Verbose, requires inheritance, less intuitive

# Approach 3: Doctest Testing
class DoctestChunker:
    """Chunker with doctest examples"""
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        >>> chunker = DoctestChunker()
        >>> chunks = chunker.chunk_text("Hello. World.")
        >>> len(chunks)
        2
        >>> chunks[0]
        'Hello'
        >>> chunks[1]
        'World.'
        """
        return text.split('. ')

# Problems: Limited test scope, hard to test complex scenarios

# Approach 4: Manual Test Runner
class ManualTestRunner:
    """Manual test execution without framework"""
    
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
    
    def add_test(self, test_name: str, test_func):
        """Add a test to the suite"""
        self.tests.append((test_name, test_func))
    
    def run_tests(self):
        """Run all tests manually"""
        print("🧪 Running Manual Tests")
        print("=" * 40)
        
        for test_name, test_func in self.tests:
            try:
                test_func()
                print(f"✅ {test_name}: PASSED")
                self.passed += 1
            except Exception as e:
                print(f"❌ {test_name}: FAILED - {e}")
                self.failed += 1
        
        self._print_summary()
    
    def _print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 40)
        print(f"Tests passed: {self.passed}")
        print(f"Tests failed: {self.failed}")
        print(f"Total tests: {len(self.tests)}")
        
        if self.failed == 0:
            print("🎉 All tests passed!")
        else:
            print("💥 Some tests failed!")

# Example usage of manual test runner
def test_chunking_basic():
    """Test basic chunking"""
    chunker = SimpleChunker()
    chunks = chunker.chunk_text("Test. Text.")
    assert len(chunks) == 2

def test_chunking_empty():
    """Test chunking empty text"""
    chunker = SimpleChunker()
    chunks = chunker.chunk_text("")
    assert len(chunks) == 1

def test_chunking_single():
    """Test chunking single sentence"""
    chunker = SimpleChunker()
    chunks = chunker.chunk_text("Single sentence.")
    assert len(chunks) == 1

# Approach 5: Mock-Based Testing
class MockChunker:
    """Chunker with mock dependencies for testing"""
    
    def __init__(self, text_processor=None):
        self.text_processor = text_processor or self._default_processor
    
    def _default_processor(self, text: str) -> str:
        """Default text processing"""
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text using processor"""
        processed_text = self.text_processor(text)
        return processed_text.split('. ')

class MockTextProcessor:
    """Mock text processor for testing"""
    
    def __init__(self, return_value: str = "Mocked text"):
        self.return_value = return_value
        self.called = False
        self.call_count = 0
    
    def __call__(self, text: str) -> str:
        """Mock call"""
        self.called = True
        self.call_count += 1
        return self.return_value

# Approach 6: Property-Based Testing
class PropertyBasedTests:
    """Property-based testing approach"""
    
    def test_chunking_properties(self):
        """Test chunking properties"""
        # Property 1: Number of chunks should be >= 1
        test_cases = [
            "Single sentence.",
            "First. Second. Third.",
            "",  # Empty text
            "Very long sentence with many words and punctuation marks!"
        ]
        
        for text in test_cases:
            chunks = self._chunk_text(text)
            assert len(chunks) >= 1, f"Text: {text}, Chunks: {chunks}"
        
        # Property 2: All chunks should be strings
        for text in test_cases:
            chunks = self._chunk_text(text)
            for chunk in chunks:
                assert isinstance(chunk, str), f"Chunk {chunk} is not a string"
        
        # Property 3: Total length should be preserved (approximately)
        for text in test_cases:
            chunks = self._chunk_text(text)
            total_chunk_length = sum(len(chunk) for chunk in chunks)
            assert abs(total_chunk_length - len(text)) <= 10, "Length mismatch"
    
    def _chunk_text(self, text: str) -> List[str]:
        """Simple chunking for property testing"""
        if not text:
            return [""]
        return text.split('. ')

# Approach 7: Integration Testing
class IntegrationTestChunker:
    """Integration testing approach"""
    
    def test_full_pipeline(self):
        """Test the entire chunking pipeline"""
        # Test data
        test_transcript = """
        Welcome to the conference. Today we'll discuss AI.
        First, let's talk about machine learning.
        Then we'll cover deep learning applications.
        Finally, we'll discuss future trends.
        """
        
        # Test chunking
        chunks = self._chunk_transcript(test_transcript)
        assert len(chunks) >= 4, "Should create at least 4 chunks"
        
        # Test chunk content
        for chunk in chunks:
            assert len(chunk.strip()) > 0, "Chunks should not be empty"
            assert chunk.strip() in test_transcript, "Chunk should be in original text"
        
        # Test chunk titles
        titles = [self._generate_title(chunk) for chunk in chunks]
        for title in titles:
            assert len(title) > 0, "Titles should not be empty"
            assert len(title) <= 100, "Titles should be reasonable length"
        
        print("✅ Integration test passed")
    
    def _chunk_transcript(self, transcript: str) -> List[str]:
        """Chunk transcript text"""
        return [line.strip() for line in transcript.split('\n') if line.strip()]
    
    def _generate_title(self, chunk: str) -> str:
        """Generate title for chunk"""
        words = chunk.split()[:5]
        return ' '.join(words) + '...'

# Approach 8: Performance Testing
class PerformanceTestChunker:
    """Performance testing approach"""
    
    def test_chunking_performance(self):
        """Test chunking performance"""
        import time
        
        # Generate test data
        test_text = self._generate_large_text(10000)  # 10K words
        
        # Time chunking
        start_time = time.time()
        chunks = self._chunk_text(test_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 1.0, f"Chunking took too long: {processing_time}s"
        assert len(chunks) > 0, "Should create chunks"
        
        print(f"✅ Performance test passed: {processing_time:.3f}s for {len(chunks)} chunks")
    
    def _generate_large_text(self, word_count: int) -> str:
        """Generate large text for testing"""
        words = ["sentence"] * word_count
        return ". ".join(words) + "."
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text"""
        return text.split('. ')

# CURRENT APPROACH: Pytest
# ========================

"""
Pytest provides:
- Simple test functions (no classes needed)
- Automatic test discovery
- Rich assertion library
- Fixtures and parametrization
- Excellent reporting
- Easy to use and understand

Example of current approach:
import pytest
from echogem import Chunker

def test_chunker_creates_chunks():
    chunker = Chunker()
    chunks = chunker.chunk_text("Test. Text.")
    assert len(chunks) == 2

@pytest.fixture
def sample_transcript():
    return "First sentence. Second sentence. Third sentence."

def test_chunker_with_fixture(sample_transcript):
    chunker = Chunker()
    chunks = chunker.chunk_text(sample_transcript)
    assert len(chunks) == 3
"""

# Migration Guide
# ===============

def migrate_from_unittest():
    """Migrate from unittest to pytest"""
    # OLD (unittest):
    # class TestChunker(unittest.TestCase):
    #     def test_chunking(self):
    #         self.assertEqual(len(chunks), 2)
    
    # NEW (pytest):
    # def test_chunking():
    #     assert len(chunks) == 2
    pass

def migrate_from_manual():
    """Migrate from manual testing to pytest"""
    # OLD (manual):
    # runner = ManualTestRunner()
    # runner.add_test("test_chunking", test_chunking_basic)
    # runner.run_tests()
    
    # NEW (pytest):
    # Just run: pytest
    pass

# Why Pytest Was Chosen
# =====================

REASONS_FOR_PYTEST = [
    "Simple test functions - no classes or inheritance needed",
    "Automatic test discovery and execution",
    "Rich assertion library with clear error messages",
    "Fixtures for test data and setup",
    "Parametrization for testing multiple scenarios",
    "Excellent reporting and output",
    "Large community and ecosystem",
    "Easy to learn and use"
]
