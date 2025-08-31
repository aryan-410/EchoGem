# Legacy: Old LLM Integration Approaches (v0.1.0-rc7)
# Different LLM providers and integration methods tested
# Replaced by Google Gemini integration in v0.2.0

import openai
import anthropic
import requests
import json
from typing import List, Dict, Optional
import os

class OpenAIChunker:
    """OpenAI GPT-based chunking - expensive but high quality"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
    
    def chunk_transcript(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Use GPT to create semantic chunks"""
        prompt = f"""
        Split the following transcript into meaningful chunks. 
        Each chunk should be a complete thought or topic.
        Maximum chunk size: {max_chunk_size} characters.
        
        Transcript:
        {text}
        
        Return only the chunks, one per line, no numbering or formatting.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            chunks = response.choices[0].message.content.strip().split('\n')
            return [chunk.strip() for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._fallback_chunking(text, max_chunk_size)
    
    def _fallback_chunking(self, text: str, max_chunk_size: int) -> List[str]:
        """Fallback to simple text splitting"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            if end < len(text):
                # Try to break at sentence boundaries
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        
        return chunks

class AnthropicChunker:
    """Anthropic Claude-based chunking - good quality, moderate cost"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def chunk_transcript(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Use Claude to create semantic chunks"""
        prompt = f"""
        Split the following transcript into meaningful chunks. 
        Each chunk should be a complete thought or topic.
        Maximum chunk size: {max_chunk_size} characters.
        
        Transcript:
        {text}
        
        Return only the chunks, one per line, no numbering or formatting.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            chunks = response.content[0].text.strip().split('\n')
            return [chunk.strip() for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return self._fallback_chunking(text, max_chunk_size)
    
    def _fallback_chunking(self, text: str, max_chunk_size: int) -> List[str]:
        """Fallback to simple text splitting"""
        return OpenAIChunker._fallback_chunking(self, text, max_chunk_size)

class LocalLLMChunker:
    """Local LLM chunking - free but lower quality"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load local LLM model"""
        try:
            # This would load a local model like llama.cpp or similar
            # For now, just a placeholder
            print("Local model loading not implemented")
        except Exception as e:
            print(f"Failed to load local model: {e}")
    
    def chunk_transcript(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Use local LLM for chunking"""
        if not self.model:
            return self._fallback_chunking(text, max_chunk_size)
        
        # Implementation would go here
        pass
    
    def _fallback_chunking(self, text: str, max_chunk_size: int) -> List[str]:
        """Fallback to simple text splitting"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            if end < len(text):
                # Try to break at sentence boundaries
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        
        return chunks

class RuleBasedChunker:
    """Rule-based chunking - fast but limited intelligence"""
    
    def __init__(self):
        self.sentence_endings = '.!?'
        self.paragraph_breaks = '\n\n'
        self.topic_indicators = [
            'first', 'second', 'third', 'finally',
            'however', 'therefore', 'meanwhile',
            'in conclusion', 'to summarize'
        ]
    
    def chunk_transcript(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Use rule-based approach for chunking"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split(self.paragraph_breaks)
        
        for paragraph in paragraphs:
            if len(paragraph) <= max_chunk_size:
                chunks.append(paragraph.strip())
            else:
                # Split long paragraphs by sentences
                sentences = self._split_sentences(paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in self.sentence_endings:
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return sentences

class HybridChunker:
    """Hybrid approach combining multiple methods"""
    
    def __init__(self, primary_chunker, fallback_chunker):
        self.primary = primary_chunker
        self.fallback = fallback_chunker
    
    def chunk_transcript(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Try primary method, fallback if it fails"""
        try:
            chunks = self.primary.chunk_transcript(text, max_chunk_size)
            if chunks and len(chunks) > 1:
                return chunks
        except Exception as e:
            print(f"Primary chunking failed: {e}")
        
        print("Falling back to secondary chunking method")
        return self.fallback.chunk_transcript(text, max_chunk_size)

# TESTING RESULTS:
# - OpenAI: High quality but expensive ($0.002 per 1K tokens)
# - Anthropic: Good quality, moderate cost ($0.003 per 1K tokens)
# - Local LLM: Free but lower quality and resource intensive
# - Rule-based: Fast and free but limited intelligence
# - Hybrid: Best of both worlds but complex

# REPLACED BY: Google Gemini for best balance of quality, cost, and reliability
