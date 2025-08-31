"""
Main processor class for orchestrating EchoGem workflow.
"""

import os
import numpy as np
from typing import List, Optional, Dict, Any
from .chunker import Chunker
from .vector_store import ChunkVectorDB
from .prompt_answer_store import PromptAnswerVectorDB
from .usage_cache import UsageCache
from .models import Chunk, ChunkResponse, QueryResult, ChunkingOptions, QueryOptions


class Processor:
    """
    Main processor class that orchestrates the entire EchoGem workflow.
    
    Features:
    - Intelligent transcript chunking
    - Vector database storage and retrieval
    - Question answering with retrieval-augmented generation
    - Usage tracking and analytics
    - Prompt-answer pair storage
    """
    
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        weights: Optional[List[float]] = None,
        chunk_index_name: str = "echogem-chunks",
        pa_index_name: str = "echogem-pa",
        usage_cache_path: str = "usage_cache_store.csv"
    ):
        """
        Initialize the processor
        
        Args:
            google_api_key: Google API key for Gemini (defaults to GOOGLE_API_KEY env var)
            pinecone_api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            embedding_model: Custom embedding model name (defaults to sentence-transformers)
            weights: Weights for scoring (defaults to equal weights)
            chunk_index_name: Name for chunk storage index
            pa_index_name: Name for prompt-answer storage index
            usage_cache_path: Path to usage cache CSV file
        """
        # Validate API keys
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable or pass google_api_key parameter.")

        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY environment variable or pass pinecone_api_key parameter.")

        # Initialize embedding model
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = "all-MiniLM-L6-v2"

        # Normalize weights
        if weights is None:
            weights = [1.0] * 7
        if len(weights) != 7:
            weights = weights[:7] + [1.0] * (7 - len(weights))
        self.weights = weights

        # Initialize components
        self.chunker = Chunker(api_key=self.google_api_key)
        self.vector_db = ChunkVectorDB(
            embedding_model=self.embedding_model,
            api_key=self.pinecone_api_key,
            index_name=chunk_index_name
        )
        self.usage_cache = UsageCache(usage_cache_path)
        self.pa_db = PromptAnswerVectorDB(
            embedding_model=self.embedding_model,
            api_key=self.pinecone_api_key,
            index_name=pa_index_name,
            region="us-east-1",
            dimension=768,
            namespace="pa_pairs",
            use_prompt_plus_answer=False,
        )

    def process_transcript(
        self, 
        file_path: str, 
        options: Optional[ChunkingOptions] = None
    ) -> ChunkResponse:
        """
        Process a transcript file and store chunks in vector database
        
        Args:
            file_path: Path to transcript file
            options: Chunking options
            
        Returns:
            ChunkResponse with processed chunks
        """
        try:
            # Load and chunk transcript
            transcript = self.chunker.load_transcript(file_path)
            chunks = self.chunker.chunk_transcript(transcript)
            
            # Store chunks in vector database
            for chunk in chunks:
                self.vector_db.add_chunk(chunk)
                self.usage_cache.record_chunk_usage(chunk.chunk_id)
            
            # Create response
            response = ChunkResponse(chunks=chunks)
            
            print(f"Processed transcript: {len(chunks)} chunks created and stored")
            return response
            
        except Exception as e:
            print(f"Error processing transcript: {e}")
            return ChunkResponse(chunks=[])

    def query(
        self, 
        question: str, 
        options: Optional[QueryOptions] = None
    ) -> QueryResult:
        """
        Answer a question using retrieval-augmented generation
        
        Args:
            question: User's question
            options: Query options
            
        Returns:
            QueryResult with answer and metadata
        """
        try:
            # Set default options
            if options is None:
                options = QueryOptions()
            
            # Retrieve relevant chunks
            chunks = self.pick_chunks(question, options.k)
            if not chunks:
                return QueryResult(
                    answer="I couldn't find relevant information to answer your question.",
                    chunks_used=[],
                    chunk_ids=[],
                    confidence=0.0,
                    metadata={"error": "No relevant chunks found"}
                )
            
            # Generate answer using chunks as context
            answer = self._generate_answer(question, chunks)
            
            # Record usage
            for chunk in chunks:
                self.usage_cache.record_chunk_usage(chunk.chunk_id)
            
            # Create result
            result = QueryResult(
                answer=answer,
                chunks_used=chunks,
                chunk_ids=[chunk.chunk_id for chunk in chunks],
                confidence=0.8,  # Placeholder confidence score
                metadata={"chunks_retrieved": len(chunks)}
            )
            
            return result
            
        except Exception as e:
            print(f"Error during query: {e}")
            return QueryResult(
                answer="An error occurred while processing your question.",
                chunks_used=[],
                chunk_ids=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def pick_chunks(self, prompt: str, k: int = 5) -> Optional[List[Chunk]]:
        """
        Retrieve the most relevant chunks for a given prompt
        
        Args:
            prompt: The user's question or prompt
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        try:
            # Search for similar chunks
            chunks = self.vector_db.search_chunks(prompt, limit=k)
            
            if not chunks:
                return None
            
            # Score and rank chunks
            scored_chunks = []
            for chunk in chunks:
                score = self._calculate_chunk_score(chunk, prompt)
                scored_chunks.append((chunk, score))
            
            # Sort by score and return top k
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            return [chunk for chunk, score in scored_chunks[:k]]
            
        except Exception as e:
            print(f"Error picking chunks: {e}")
            return None

    def _calculate_chunk_score(self, chunk: Chunk, prompt: str) -> float:
        """Calculate relevance score for a chunk"""
        try:
            # Simple scoring based on keyword overlap
            prompt_words = set(prompt.lower().split())
            chunk_words = set(chunk.content.lower().split())
            
            # Keyword overlap
            keyword_overlap = len(prompt_words.intersection(chunk_words))
            
            # Usage-based scoring
            usage_score = self.usage_cache.get_chunk_usage_count(chunk.chunk_id)
            
            # Combine scores
            score = keyword_overlap * 0.6 + usage_score * 0.4
            
            return score
            
        except Exception as e:
            print(f"Error calculating chunk score: {e}")
            return 0.0

    def _generate_answer(self, question: str, chunks: List[Chunk]) -> str:
        """Generate answer using retrieved chunks as context"""
        try:
            # Create context from chunks
            context = "\n\n".join([chunk.content for chunk in chunks])
            
            # Simple answer generation (placeholder)
            answer = f"Based on the available information: {question}\n\nContext: {context[:200]}..."
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I couldn't generate an answer at this time."

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for analytics"""
        try:
            return self.usage_cache.get_usage_statistics()
        except Exception as e:
            print(f"Error getting usage statistics: {e}")
            return {"error": str(e)}
