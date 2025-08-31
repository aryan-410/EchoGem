"""
Transcript chunking module using Google Gemini for intelligent segmentation.
"""

import os
import json
import google.generativeai as genai
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .models import Chunk


class Chunker:
    """
    Intelligent transcript chunking using LLM-based semantic analysis
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        embed_model: str = "all-MiniLM-L6-v2",
        max_tokens: int = 2000,
        similarity_threshold: float = 0.82,
        coherence_threshold: float = 0.75,
    ):
        """
        Initialize the chunker
        
        Args:
            api_key: Google API key for Gemini
            embed_model: Path to sentence transformer model or model name
            max_tokens: Maximum tokens per chunk
            similarity_threshold: Threshold for semantic similarity
            coherence_threshold: Threshold for coherence
        """
        # Initialize Google Gemini
        if api_key:
            genai.configure(api_key=api_key)
        else:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            else:
                raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable.")
        
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize sentence transformer
        try:
            self.embedder = SentenceTransformer(embed_model)
        except Exception as e:
            print(f"Warning: Could not load model {embed_model}: {e}")
            print("Using default model")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.max_tokens = max_tokens
        self.sim_threshold = similarity_threshold
        self.coh_threshold = coherence_threshold

    def load_transcript(self, file_path: str) -> str:
        """
        Load transcript text from file
        
        Args:
            file_path: Path to transcript file
            
        Returns:
            Transcript text content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                transcript = f.read()
            print(f"Transcript loaded ({len(transcript)} characters)")
            return transcript
        except FileNotFoundError:
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading transcript: {str(e)}")

    def chunk_transcript(self, transcript: str) -> List[Chunk]:
        """
        Chunk transcript using LLM-based semantic analysis
        
        Args:
            transcript: Transcript text to chunk
            
        Returns:
            List of Chunk objects
        """
        try:
            # Create chunking prompt
            prompt = self._create_prompt(transcript)
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse response
            chunks = self._parse_chunk_response(response.text)
            
            print(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"Error during chunking: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(transcript)

    def _create_prompt(self, transcript: str) -> str:
        """Create the chunking prompt"""
        return f"""
        **SYSTEM PROMPT**
        You are a transcript processing expert. The following transcript needs to be chunked very intelligently and logically. Ensure sensible segments and structure to be later provided as context to answer questions.

        **INSTRUCTIONS**
        1. Create as many or as few chunks as needed
        2. Each chunk should contain consecutive sentences
        3. For each chunk provide:
          - title: 2-5 word summary
          - content: exact sentences
          - keywords: 3-5 important terms
          - named_entities: any mentioned names
          - timestamp_range: estimate like "00:00-01:30"

        **TRANSCRIPT**
        {transcript[:5000]}...

        **OUTPUT FORMAT**
        You must output ONLY valid JSON in this exact format:
        {{
          "chunks": [
            {{
              "title": "Summary",
              "content": "Actual sentences",
              "keywords": ["term1", "term2"],
              "named_entities": ["Name"],
              "timestamp_range": "00:00-01:30"
            }}
          ]
        }}
        """

    def _parse_chunk_response(self, response_text: str) -> List[Chunk]:
        """Parse the LLM response into Chunk objects"""
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)
            
            chunks = []
            for chunk_data in data.get('chunks', []):
                chunk = Chunk(
                    title=chunk_data.get('title', 'Untitled'),
                    content=chunk_data.get('content', ''),
                    keywords=chunk_data.get('keywords', []),
                    named_entities=chunk_data.get('named_entities', []),
                    timestamp_range=chunk_data.get('timestamp_range', ''),
                    chunk_id=f"chunk_{len(chunks)}"
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"Error parsing chunk response: {e}")
            return []

    def _fallback_chunking(self, transcript: str) -> List[Chunk]:
        """Fallback chunking method using simple text splitting"""
        words = transcript.split()
        chunks = []
        chunk_size = self.max_tokens
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk = Chunk(
                title=f"Chunk {len(chunks) + 1}",
                content=chunk_text,
                keywords=[],
                named_entities=[],
                timestamp_range="",
                chunk_id=f"fallback_chunk_{len(chunks)}"
            )
            chunks.append(chunk)
        
        return chunks

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using sentence transformer"""
        try:
            embedding = self.embedder.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return [0.0] * 384  # Default dimension
