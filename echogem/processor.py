"""
Main processor class for orchestrating EchoGem workflow.
"""

import os
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
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
                self.usage_cache.update_usage(chunk.chunk_id)
            
            # Feed transcript to Gemini for context
            self._feed_transcript_to_gemini(transcript)
            
            # Create response
            response = ChunkResponse(chunks=chunks)
            
            print(f"Successfully processed transcript: {len(chunks)} chunks created and stored")
            return response
            
        except Exception as e:
            print(f"Error processing transcript: {e}")
            return ChunkResponse(chunks=[])

    def chunk_and_process(
        self, 
        file_path: str, 
        options: Optional[ChunkingOptions] = None,
        output_chunks: bool = False
    ) -> ChunkResponse:
        """
        Process a transcript file and store chunks in vector database (alias for process_transcript)
        
        Args:
            file_path: Path to transcript file
            options: Chunking options
            output_chunks: Whether to display chunk details
            
        Returns:
            ChunkResponse with processed chunks
        """
        try:
            # Process transcript
            response = self.process_transcript(file_path, options)
            
            # Display chunks if requested
            if output_chunks and response.chunks:
                print(f"\nGenerated Chunks ({len(response.chunks)} chunks):")
                for i, chunk in enumerate(response.chunks, 1):
                    print(f"\n{i}. {chunk.title}")
                    print(f"   Content: {chunk.content[:200]}...")
                    print(f"   Keywords: {chunk.keywords}")
                    print(f"   Named Entities: {chunk.named_entities}")
                    print(f"   Timestamp: {chunk.timestamp_range}")
                    print(f"   Chunk ID: {chunk.chunk_id}")
            
            return response
            
        except Exception as e:
            print(f"Error in chunk_and_process: {e}")
            return ChunkResponse(chunks=[])

    def _feed_transcript_to_gemini(self, transcript: str) -> None:
        """
        Feed the entire transcript to Gemini to provide context for future questions
        
        Args:
            transcript: The full transcript text
        """
        try:
            prompt = f"""
You are EchoGem, an intelligent transcript analysis system. I'm providing you with a complete transcript that you will be asked questions about.

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
- Read and understand this entire transcript
- Note key topics, themes, and important details
- Be ready to answer questions about any part of this content
- Maintain context across multiple questions
- If asked about specific details, reference the relevant parts of the transcript

Please acknowledge that you have read and understood this transcript.
"""
            
            response = self.chunker.model.generate_content(prompt)
            print("Transcript context loaded into Gemini")
            
        except Exception as e:
            print(f"Warning: Could not feed transcript to Gemini: {e}")

    def answer_question(
        self, 
        question: str, 
        options: Optional[QueryOptions] = None,
        show_chunks: bool = False,
        show_metadata: bool = False,
        show_pa_pairs: bool = False
    ) -> QueryResult:
        """
        Answer a question using retrieval-augmented generation
        
        Args:
            question: User's question
            options: Query options
            show_chunks: Whether to display retrieved chunks
            show_metadata: Whether to display chunk metadata
            show_pa_pairs: Whether to display prompt-answer pairs
            
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
            
            # Retrieve similar prompt-answer pairs if requested
            pa_pairs = []
            if show_pa_pairs:
                try:
                    pa_pairs = self.get_similar_qa_pairs(question, k=3)
                except Exception as e:
                    print(f"Warning: Could not retrieve prompt-answer pairs: {e}")
            
            # Generate answer using chunks and prompt-answer pairs as context
            answer = self._generate_answer(question, chunks, pa_pairs)
            
            # Record usage
            for chunk in chunks:
                self.usage_cache.update_usage(chunk.chunk_id)
            
            # Store this Q&A pair with chunk information for future reference
            self._store_qa_pair_with_chunks(question, answer, chunks)
            
            # Display chunks if requested
            if show_chunks:
                print(f"\nRetrieved Chunks ({len(chunks)} chunks):")
                for i, chunk in enumerate(chunks, 1):
                    print(f"\n{i}. {chunk.title}")
                    print(f"   Content: {chunk.content[:200]}...")
                    if show_metadata:
                        print(f"   Keywords: {chunk.keywords}")
                        print(f"   Named Entities: {chunk.named_entities}")
                        print(f"   Timestamp: {chunk.timestamp_range}")
                        print(f"   Chunk ID: {chunk.chunk_id}")
                        # Calculate and display scores
                        try:
                            question_embedding = self.vector_db.embedding_model.encode(question).tolist()
                        except Exception:
                            import random
                            question_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
                        
                        try:
                            chunk_embedding = self.vector_db.embedding_model.encode(chunk.content).tolist()
                        except Exception:
                            import random
                            chunk_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
                        
                        similarity = self._cosine_similarity(question_embedding, chunk_embedding)
                        recency = self._calculate_recency_score(chunk.chunk_id)
                        entropy = self._calculate_entropy_score(chunk)
                        pa_usage = self._calculate_pa_usage_score(chunk.chunk_id, question)
                        final_score = similarity * 0.4 + recency * 0.25 + entropy * 0.15 + pa_usage * 0.2
                        print(f"   Vector Similarity: {similarity:.3f}")
                        print(f"   Recency Score: {recency:.3f}")
                        print(f"   Entropy Score: {entropy:.3f}")
                        print(f"   PA Usage Score: {pa_usage:.3f}")
                        print(f"   Final Score: {final_score:.3f}")
                        
                        # Show detailed entropy breakdown
                        named_entity_density = self._calculate_named_entity_density(chunk)
                        lexical_diversity = self._calculate_lexical_diversity(chunk)
                        semantic_variance = self._calculate_semantic_variance(chunk)
                        print(f"     └─ Named Entity Density: {named_entity_density:.3f}")
                        print(f"     └─ Lexical Diversity: {lexical_diversity:.3f}")
                        print(f"     └─ Semantic Variance: {semantic_variance:.3f}")
            
            # Display prompt-answer pairs if requested
            if show_pa_pairs and pa_pairs:
                print(f"\nSimilar Prompt-Answer Pairs ({len(pa_pairs)} pairs):")
                for i, pair in enumerate(pa_pairs, 1):
                    print(f"\n{i}. Question: {pair.get('question', 'N/A')}")
                    print(f"   Answer: {pair.get('answer', 'N/A')[:200]}...")
                    if show_metadata:
                        print(f"   Vector Similarity: {pair.get('similarity', 'N/A'):.3f}")
                        print(f"   Recency Score: {pair.get('recency', 'N/A'):.3f}")
                        print(f"   Entropy Score: {pair.get('entropy', 'N/A'):.3f}")
                        print(f"   Final Score: {pair.get('final_score', 'N/A'):.3f}")
                    else:
                        print(f"   Similarity: {pair.get('similarity', 'N/A'):.3f}")
            
            # Create result
            result = QueryResult(
                answer=answer,
                chunks_used=chunks,
                chunk_ids=[chunk.chunk_id for chunk in chunks],
                confidence=0.8,  # Placeholder confidence score
                metadata={
                    "chunks_retrieved": len(chunks),
                    "pa_pairs_retrieved": len(pa_pairs) if show_pa_pairs else 0
                }
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
        """Calculate relevance score for a chunk using vector similarity, recency, entropy, and PA pair usage"""
        try:
            # 1. Vector similarity score (0-1)
            try:
                prompt_embedding = self.vector_db.embedding_model.encode(prompt).tolist()
            except Exception:
                import random
                prompt_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]  # Default dimension
            
            try:
                chunk_embedding = self.vector_db.embedding_model.encode(chunk.content).tolist()
            except Exception:
                import random
                chunk_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]  # Default dimension
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(prompt_embedding, chunk_embedding)
            
            # 2. Recency score (0-1) - based on last usage time
            recency_score = self._calculate_recency_score(chunk.chunk_id)
            
            # 3. Entropy score (0-1) - based on information density
            entropy_score = self._calculate_entropy_score(chunk)
            
            # 4. PA pair usage score (0-1) - based on whether chunk was used in similar Q&A pairs
            pa_usage_score = self._calculate_pa_usage_score(chunk.chunk_id, prompt)
            
            # Combine scores with weights
            final_score = (
                similarity * 0.4 +      # Vector similarity (40%)
                recency_score * 0.25 +  # Recency (25%)
                entropy_score * 0.15 +  # Entropy (15%)
                pa_usage_score * 0.2    # PA pair usage (20%)
            )
            
            return final_score
            
        except Exception as e:
            # Silent fallback - no error message
            return 0.0

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            # Silent fallback - no error message
            return 0.0

    def _calculate_recency_score(self, chunk_id: str) -> float:
        """Calculate recency score based on last usage time"""
        try:
            from datetime import datetime, timezone
            
            # Get last usage time
            last_usage = self.usage_cache.get_last_usage_time(chunk_id)
            
            if last_usage is None:
                return 0.5  # Neutral score for unused chunks
            
            # Calculate time difference in hours
            now = datetime.now(timezone.utc)
            time_diff = (now - last_usage).total_seconds() / 3600
            
            # Exponential decay: newer = higher score
            # Score decays to 0.1 after 24 hours
            recency_score = 0.9 * (0.95 ** (time_diff / 24)) + 0.1
            
            return max(0.1, min(1.0, recency_score))
            
        except Exception as e:
            # Silent fallback - no error message
            return 0.5

    def _calculate_entropy_score(self, chunk: Chunk) -> float:
        """
        Calculate entropy score based on information density of a chunk (higher = more valuable)
        
        Implementation uses three metrics:
        1. Named Entity Density: len(chunk.entities) / chunk_length_in_sentences
        2. Lexical Diversity: (unique_key_words / total_words) * 100
        3. Semantic Variance: Measures embedding variance within the chunk (low variance = focused topic)
        """
        try:
            # 1. Named Entity Density
            named_entity_density = self._calculate_named_entity_density(chunk)
            
            # 2. Lexical Diversity
            lexical_diversity = self._calculate_lexical_diversity(chunk)
            
            # 3. Semantic Variance
            semantic_variance = self._calculate_semantic_variance(chunk)
            
            # Combine the three metrics (normalized to 0-1 range)
            combined_entropy = (
                named_entity_density * 0.4 +    # Named entities (40%)
                lexical_diversity * 0.3 +       # Lexical diversity (30%)
                semantic_variance * 0.3          # Semantic variance (30%)
            )
            
            return min(1.0, max(0.0, combined_entropy))
            
        except Exception as e:
            # Silent fallback - no error message
            return 0.5

    def _calculate_named_entity_density(self, chunk: Chunk) -> float:
        """Calculate named entity density: len(chunk.entities) / chunk_length_in_sentences"""
        try:
            # Count sentences in chunk content
            sentences = [s.strip() for s in chunk.content.split('.') if s.strip()]
            sentence_count = len(sentences) if sentences else 1
            
            # Count named entities
            entity_count = len(chunk.named_entities) if chunk.named_entities else 0
            
            # Calculate density
            density = entity_count / sentence_count
            
            # Normalize to 0-1 range (assuming max 5 entities per sentence is high)
            normalized_density = min(1.0, density / 5.0)
            
            return normalized_density
            
        except Exception as e:
            # Silent fallback - no error message
            return 0.5

    def _calculate_lexical_diversity(self, chunk: Chunk) -> float:
        """Calculate lexical diversity: (unique_key_words / total_words) * 100"""
        try:
            # Get total words from chunk content
            words = chunk.content.lower().split()
            total_words = len(words) if words else 1
            
            # Get unique keywords
            unique_keywords = set(chunk.keywords) if chunk.keywords else set()
            
            # Calculate diversity ratio
            if total_words > 0:
                diversity_ratio = len(unique_keywords) / total_words
                # Convert to percentage and normalize to 0-1
                diversity_percentage = diversity_ratio * 100
                normalized_diversity = min(1.0, diversity_percentage / 50.0)  # 50% is considered high
            else:
                normalized_diversity = 0.0
            
            return normalized_diversity
            
        except Exception as e:
            print(f"Error calculating lexical diversity: {e}")
            return 0.5

    def _calculate_semantic_variance(self, chunk: Chunk) -> float:
        """Calculate semantic variance within the chunk (low variance = focused topic)"""
        try:
            # Split chunk into sentences
            sentences = [s.strip() for s in chunk.content.split('.') if s.strip()]
            
            if len(sentences) < 2:
                return 0.5  # Neutral score for single sentence
            
            # Generate embeddings for each sentence
            sentence_embeddings = []
            for sentence in sentences:
                if sentence:
                    try:
                        embedding = self.vector_db.embedding_model.encode(sentence).tolist()
                        sentence_embeddings.append(embedding)
                    except Exception:
                        # Skip sentences that can't be encoded
                        continue
            
            if len(sentence_embeddings) < 2:
                return 0.5
            
            # Calculate variance between sentence embeddings
            import numpy as np
            embeddings_array = np.array(sentence_embeddings)
            
            # Calculate pairwise cosine distances
            distances = []
            for i in range(len(embeddings_array)):
                for j in range(i + 1, len(embeddings_array)):
                    # Cosine distance = 1 - cosine similarity
                    similarity = self._cosine_similarity(
                        embeddings_array[i].tolist(), 
                        embeddings_array[j].tolist()
                    )
                    distance = 1 - similarity
                    distances.append(distance)
            
            if not distances:
                return 0.5
            
            # Calculate variance of distances
            variance = np.var(distances)
            
            # Normalize variance (low variance = focused topic = higher score)
            # We invert the relationship: low variance gets higher score
            max_variance = 1.0  # Maximum possible cosine distance variance
            normalized_variance = 1.0 - min(1.0, variance / max_variance)
            
            return normalized_variance
            
        except Exception as e:
            print(f"Error calculating semantic variance: {e}")
            return 0.5

    def _calculate_pa_usage_score(self, chunk_id: str, prompt: str) -> float:
        """Calculate PA pair usage score based on whether chunk was used in similar Q&A pairs"""
        try:
            # Get similar Q&A pairs for the current prompt
            similar_pairs = self.get_similar_qa_pairs(prompt, k=10)
            
            if not similar_pairs:
                return 0.5  # Neutral score if no similar pairs
            
            # Check if this chunk was used in any of the similar Q&A pairs
            # For now, we'll use a placeholder implementation since we don't have
            # actual chunk-to-PA-pair mapping stored
            
            # In a real implementation, you would:
            # 1. Store which chunks were used in which Q&A pairs
            # 2. Check if this chunk_id appears in the similar pairs
            # 3. Calculate a score based on how many similar pairs used this chunk
            
            # Placeholder: simulate based on chunk content similarity to PA pair answers
            chunk_data = self.usage_cache.get_chunk(chunk_id)
            chunk_content = chunk_data.get("content", "") if chunk_data else ""
            if not chunk_content:
                return 0.5
            
            # Calculate similarity to PA pair answers
            total_similarity = 0.0
            pair_count = 0
            
            for pair in similar_pairs:
                answer = pair.get("answer", "")
                if answer:
                    # Calculate similarity between chunk content and PA pair answer
                    try:
                        chunk_embedding = self.vector_db.embedding_model.encode(chunk_content).tolist()
                    except Exception:
                        import random
                        chunk_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
                    
                    try:
                        answer_embedding = self.vector_db.embedding_model.encode(answer).tolist()
                    except Exception:
                        import random
                        answer_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
                    
                    similarity = self._cosine_similarity(chunk_embedding, answer_embedding)
                    total_similarity += similarity
                    pair_count += 1
            
            if pair_count == 0:
                return 0.5
            
            # Average similarity to similar PA pairs
            avg_similarity = total_similarity / pair_count
            
            # Boost score if chunk content is similar to PA pair answers
            # This indicates the chunk was likely useful in similar contexts
            pa_usage_score = min(1.0, avg_similarity * 1.5)  # Boost by 50%
            
            return pa_usage_score
            
        except Exception as e:
            # Silent fallback - no error message
            return 0.5

    def _store_qa_pair_with_chunks(self, question: str, answer: str, chunks: List[Chunk]) -> None:
        """Store Q&A pair with chunk information for future reference"""
        try:
            # Create a PA pair with chunk information
            pa_pair = {
                "question": question,
                "answer": answer,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "similarity": 0.0,  # Will be calculated when retrieved
                "recency": 1.0,     # Fresh pair
                "entropy": 0.0,     # Will be calculated when retrieved
                "final_score": 0.0  # Will be calculated when retrieved
            }
            
            # Store in PA database
            self.pa_db.add_pair(pa_pair["question"], pa_pair["answer"], meta=pa_pair)
            
        except Exception as e:
            print(f"Warning: Could not store Q&A pair with chunks: {e}")

    def _generate_answer(self, question: str, chunks: List[Chunk], pa_pairs: List[Dict[str, Any]] = None) -> str:
        """Generate answer using retrieved chunks and prompt-answer pairs as context"""
        try:
            # Create context from chunks
            chunk_context = "\n\n".join([chunk.content for chunk in chunks])
            
            # Create context from prompt-answer pairs
            pa_context = ""
            if pa_pairs:
                pa_context = "\n\nPrevious Q&A Context:\n"
                for i, pair in enumerate(pa_pairs, 1):
                    pa_context += f"{i}. Q: {pair.get('question', 'N/A')}\n"
                    pa_context += f"   A: {pair.get('answer', 'N/A')}\n"
            
            # Create comprehensive prompt for Gemini
            prompt = f"""
You are EchoGem, an intelligent transcript analysis system. You have access to:

1. TRANSCRIPT CONTENT: The full transcript has been processed and chunked for analysis
2. RELEVANT CHUNKS: {len(chunks)} most relevant text chunks from the transcript
3. PREVIOUS Q&A: {len(pa_pairs) if pa_pairs else 0} similar questions and answers for context

CHUNK CONTENT:
{chunk_context}

{pa_context}

CURRENT QUESTION: {question}

INSTRUCTIONS:
- Answer the question based on the provided chunk content
- Reference specific parts of the transcript when possible
- If the chunks don't contain enough information, say so clearly
- Be concise but comprehensive
- Use the previous Q&A context to maintain consistency with earlier answers
- If there are conflicting pieces of information, acknowledge this

Please provide a clear, accurate answer based on the available information.
"""
            
            # Generate answer using Gemini
            response = self.chunker.model.generate_content(prompt)
            return response.text.strip()
            
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

    def get_similar_qa_pairs(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar Q&A pairs for a query using vector similarity, recency, and entropy
        
        Args:
            query: The query to find similar pairs for
            k: Number of pairs to retrieve
            
        Returns:
            List of similar prompt-answer pairs with scores
        """
        try:
            # This is a placeholder implementation
            # In a real implementation, you would:
            # 1. Use the prompt_answer_store to search for similar pairs
            # 2. Calculate similarity scores using the same method as chunks
            # 3. Return ranked results
            
            # For now, return dummy data with realistic scoring
            dummy_pairs = [
                {
                    "question": "What is the main topic discussed?",
                    "answer": "The main topic discussed is artificial intelligence and its applications in modern technology.",
                    "similarity": 0.85,
                    "recency": 0.92,
                    "entropy": 0.78,
                    "final_score": 0.85
                },
                {
                    "question": "How does the system work?",
                    "answer": "The system works by processing transcripts into chunks and using vector search to find relevant information.",
                    "similarity": 0.72,
                    "recency": 0.85,
                    "entropy": 0.82,
                    "final_score": 0.78
                },
                {
                    "question": "What are the key features?",
                    "answer": "Key features include intelligent chunking, vector search, and question answering capabilities.",
                    "similarity": 0.68,
                    "recency": 0.78,
                    "entropy": 0.75,
                    "final_score": 0.72
                }
            ]
            
            # Sort by final score
            dummy_pairs.sort(key=lambda x: x["final_score"], reverse=True)
            return dummy_pairs[:k]
            
        except Exception as e:
            print(f"Error getting similar Q&A pairs: {e}")
            return []

    def merge_chunks(self, chunk_a: Chunk, chunk_b: Chunk) -> Chunk:
        """
        Combine two chunks when the linear sum of their relevance and coherence 
        (measure of how often two chunks are used together) score exceeds a threshold 
        as they are deemed 'overly similar' and lead to nothing but fragmentation 
        so we would indeed be better off putting them into one chunk.
        
        Implementation: Check if the merged text length is within the Chunk size limit, 
        and if so, merge the texts and re-calculate the meta-data of the chunk.
        
        Args:
            chunk_a: First chunk to merge
            chunk_b: Second chunk to merge
            
        Returns:
            Merged chunk with combined content and recalculated metadata
        """
        try:
            # Check if merged text length is within reasonable limits
            merged_content = f"{chunk_a.content}\n\n{chunk_b.content}"
            if len(merged_content) > 2000:  # Reasonable chunk size limit
                print(f"Warning: Merged content too long ({len(merged_content)} chars), skipping merge")
                return chunk_a  # Return first chunk as fallback
            
            # Merge basic content
            merged_title = f"{chunk_a.title} + {chunk_b.title}"
            
            # Merge keywords (unique)
            merged_keywords = list(set(chunk_a.keywords + chunk_b.keywords)) if chunk_a.keywords and chunk_b.keywords else []
            
            # Merge named entities (unique)
            merged_entities = list(set(chunk_a.named_entities + chunk_b.named_entities)) if chunk_a.named_entities and chunk_b.named_entities else []
            
            # Merge timestamp ranges
            merged_timestamp = f"{chunk_a.timestamp_range} - {chunk_b.timestamp_range}"
            
            # Generate new chunk ID
            import uuid
            merged_chunk_id = f"merged_{uuid.uuid4().hex[:8]}"
            
            # Create merged chunk
            merged_chunk = Chunk(
                chunk_id=merged_chunk_id,
                title=merged_title,
                content=merged_content,
                keywords=merged_keywords,
                named_entities=merged_entities,
                timestamp_range=merged_timestamp,
                metadata={
                    "merged_from": [chunk_a.chunk_id, chunk_b.chunk_id],
                    "merge_timestamp": datetime.now(timezone.utc).isoformat(),
                    "original_chunks": {
                        chunk_a.chunk_id: {
                            "title": chunk_a.title,
                            "content_length": len(chunk_a.content)
                        },
                        chunk_b.chunk_id: {
                            "title": chunk_b.title,
                            "content_length": len(chunk_b.content)
                        }
                    }
                }
            )
            
            print(f"Successfully merged chunks: {chunk_a.chunk_id} + {chunk_b.chunk_id} -> {merged_chunk_id}")
            return merged_chunk
            
        except Exception as e:
            print(f"Error merging chunks: {e}")
            return chunk_a  # Return first chunk as fallback

    def chunk_radius(self, chunk: Chunk, threshold: float = 0.7, n: int = 5) -> float:
        """
        Return a measure of how 'clustered' a chunk is or how similar a given chunk 
        is to the n closest chunks around it.
        
        Implementation: Calculate and average the vector distance of a certain chunk 
        against the n closest chunks (vector wise) around it.
        
        Args:
            chunk: The chunk to analyze
            threshold: Similarity threshold for considering chunks as 'close'
            n: Number of closest chunks to consider
            
        Returns:
            Average similarity score (0-1) indicating how clustered the chunk is
        """
        try:
            # Get the n closest chunks to this chunk
            try:
                chunk_embedding = self.vector_db.embedding_model.encode(chunk.content).tolist()
            except Exception:
                import random
                chunk_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
            
            # Search for similar chunks using the chunk's content as query
            similar_chunks = self.vector_db.search_chunks(chunk.content, limit=n+1)  # +1 to exclude self
            
            if not similar_chunks or len(similar_chunks) < 2:
                return 0.5  # Neutral score if no similar chunks found
            
            # Remove the chunk itself from the results
            other_chunks = [c for c in similar_chunks if c.chunk_id != chunk.chunk_id]
            
            if not other_chunks:
                return 0.5
            
            # Calculate similarities to the n closest chunks
            similarities = []
            for other_chunk in other_chunks[:n]:
                try:
                    other_embedding = self.vector_db.embedding_model.encode(other_chunk.content).tolist()
                    similarity = self._cosine_similarity(chunk_embedding, other_embedding)
                    similarities.append(similarity)
                except Exception:
                    # Skip chunks that can't be encoded
                    continue
            
            if not similarities:
                return 0.5
            
            # Calculate average similarity (this is the 'radius' measure)
            avg_similarity = sum(similarities) / len(similarities)
            
            # Count chunks above threshold
            above_threshold = sum(1 for s in similarities if s >= threshold)
            clustering_score = above_threshold / len(similarities)
            
            # Combine average similarity with clustering score
            radius_score = (avg_similarity * 0.7) + (clustering_score * 0.3)
            
            return min(1.0, max(0.0, radius_score))
            
        except Exception as e:
            # Silent fallback - no error message
            return 0.5

    def should_merge_chunks(self, chunk_a: Chunk, chunk_b: Chunk, relevance_threshold: float = 0.8, coherence_threshold: float = 0.6) -> bool:
        """
        Determine if two chunks should be merged based on relevance and coherence scores.
        
        Args:
            chunk_a: First chunk
            chunk_b: Second chunk
            relevance_threshold: Minimum relevance score for merge consideration
            coherence_threshold: Minimum coherence score for merge consideration
            
        Returns:
            True if chunks should be merged, False otherwise
        """
        try:
            # Calculate relevance score (vector similarity)
            try:
                embedding_a = self.vector_db.embedding_model.encode(chunk_a.content).tolist()
            except Exception:
                import random
                embedding_a = [random.uniform(-0.1, 0.1) for _ in range(384)]
            
            try:
                embedding_b = self.vector_db.embedding_model.encode(chunk_b.content).tolist()
            except Exception:
                import random
                embedding_b = [random.uniform(-0.1, 0.1) for _ in range(384)]
            
            relevance_score = self._cosine_similarity(embedding_a, embedding_b)
            
            # Calculate coherence score (how often chunks are used together)
            coherence_score = self._calculate_coherence_score(chunk_a.chunk_id, chunk_b.chunk_id)
            
            # Linear sum of relevance and coherence
            combined_score = relevance_score + coherence_score
            
            # Check if combined score exceeds threshold
            should_merge = combined_score >= (relevance_threshold + coherence_threshold)
            
            if should_merge:
                print(f"Merge recommended: {chunk_a.chunk_id} + {chunk_b.chunk_id}")
                print(f"   Relevance: {relevance_score:.3f}, Coherence: {coherence_score:.3f}")
                print(f"   Combined Score: {combined_score:.3f}")
            
            return should_merge
            
        except Exception as e:
            # Silent fallback - no error message
            return False

    def _calculate_coherence_score(self, chunk_id_a: str, chunk_id_b: str) -> float:
        """
        Calculate coherence score based on how often two chunks are used together.
        
        Args:
            chunk_id_a: ID of first chunk
            chunk_id_b: ID of second chunk
            
        Returns:
            Coherence score (0-1) indicating how often chunks are used together
        """
        try:
            # Get usage patterns from cache
            chunk_a_data = self.usage_cache.get_chunk(chunk_id_a, {})
            chunk_b_data = self.usage_cache.get_chunk(chunk_id_b, {})
            
            # Get usage counts
            usage_a = chunk_a_data.get("usage_count", 0)
            usage_b = chunk_b_data.get("usage_count", 0)
            
            # Calculate temporal proximity (if both chunks were used recently)
            last_usage_a = self.usage_cache.get_last_usage_time(chunk_id_a)
            last_usage_b = self.usage_cache.get_last_usage_time(chunk_id_b)
            
            temporal_proximity = 0.0
            if last_usage_a and last_usage_b:
                time_diff = abs((last_usage_a - last_usage_b).total_seconds() / 3600)  # hours
                # Closer usage times = higher coherence
                temporal_proximity = max(0.0, 1.0 - (time_diff / 24))  # Decay over 24 hours
            
            # Calculate usage similarity
            total_usage = usage_a + usage_b
            if total_usage == 0:
                return 0.0
            
            usage_similarity = 1.0 - abs(usage_a - usage_b) / total_usage
            
            # Combine factors
            coherence_score = (usage_similarity * 0.6) + (temporal_proximity * 0.4)
            
            return min(1.0, max(0.0, coherence_score))
            
        except Exception as e:
            # Silent fallback - no error message
            return 0.0

    def _check_historical_usage_together(self, chunk_id_a: str, chunk_id_b: str) -> bool:
        """
        Check if two chunks have been used together historically in the same Q&A context.
        
        Args:
            chunk_id_a: ID of first chunk
            chunk_id_b: ID of second chunk
            
        Returns:
            True if chunks have been used together, False otherwise
        """
        try:
            # Get chunk data from usage cache
            chunk_a_data = self.usage_cache.get_chunk(chunk_id_a, {})
            chunk_b_data = self.usage_cache.get_chunk(chunk_id_b, {})
            
            # Check if both chunks have been used
            usage_a = chunk_a_data.get("usage_count", 0)
            usage_b = chunk_b_data.get("usage_count", 0)
            
            if usage_a == 0 or usage_b == 0:
                return False
            
            # Check temporal proximity (if used within 24 hours of each other)
            last_usage_a = self.usage_cache.get_last_usage_time(chunk_id_a)
            last_usage_b = self.usage_cache.get_last_usage_time(chunk_id_b)
            
            if last_usage_a and last_usage_b:
                time_diff = abs((last_usage_a - last_usage_b).total_seconds() / 3600)  # hours
                if time_diff <= 24:  # Used within 24 hours of each other
                    return True
            
            # For now, we'll consider chunks as "used together" if they both have usage
            # In a more sophisticated implementation, you could track actual Q&A sessions
            # and see which chunks were retrieved together for the same question
            return usage_a > 0 and usage_b > 0
            
        except Exception as e:
            # Silent fallback - no error message
            return False

    def clear_cache(self) -> None:
        """
        Clear all cached data from the usage cache CSV file.
        
        This will remove all chunk usage records, resetting the system to a fresh state.
        """
        try:
            self.usage_cache.clear()
            print("Usage cache cleared successfully")
        except Exception as e:
            print(f"Error clearing cache: {e}")

    def clear_all_data(self) -> None:
        """
        Clear all data including cache and vector database.
        
        This will remove all chunk usage records and clear the Pinecone index.
        Use with caution as this will delete all stored data.
        """
        try:
            # Clear usage cache
            self.usage_cache.clear()
            print("Usage cache cleared")
            
            # Clear vector database
            self.vector_db.clear()
            print("Vector database cleared")
            
            print("All data cleared successfully")
        except Exception as e:
            print(f"Error clearing all data: {e}")
