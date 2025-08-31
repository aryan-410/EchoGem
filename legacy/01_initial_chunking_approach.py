# Legacy: Initial Chunking Approach (v0.1.0-alpha)
# This was the first attempt at transcript chunking - simple text splitting
# Replaced by LLM-based semantic chunking in v0.2.0

import re
from typing import List

class SimpleTextChunker:
    """Original chunking approach - basic text splitting by sentences/paragraphs"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
            
        return chunks

# PROBLEMS WITH THIS APPROACH:
# - Fixed chunk sizes don't respect semantic boundaries
# - Overlap was arbitrary and didn't consider context
# - No understanding of content meaning or relationships
# - Poor performance on transcripts with varying sentence lengths
# - No way to handle speaker changes or topic shifts

# REPLACED BY: LLM-based semantic chunking in Chunker class
