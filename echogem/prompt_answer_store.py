"""
Vector database operations for storing and retrieving prompt-answer pairs.
"""

import os
import json
import hashlib
import time
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import pinecone
from sentence_transformers import SentenceTransformer
from .models import PAPair


class PromptAnswerVectorDB:
    """
    Pinecone-backed vector database for storing and retrieving prompt-answer pairs.
    
    Features:
    - Vector similarity search for Q&A pairs
    - Usage tracking and recency scoring
    - Configurable scoring weights
    """
    
    def __init__(
        self,
        embedding_model,
        api_key: Optional[str] = None,
        index_name: str = "echogem-pa",
        region: str = "us-east-1",
        dimension: int = 768,
        namespace: str = "pa_pairs",
        use_prompt_plus_answer: bool = True
    ):
        """
        Initialize the prompt-answer vector database
        
        Args:
            embedding_model: Embedding model for text vectorization
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index
            region: AWS region for the index
            dimension: Vector dimension
            namespace: Pinecone namespace for Q&A pairs
            use_prompt_plus_answer: Whether to embed prompt+answer or just prompt
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY environment variable or pass api_key parameter.")
        
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.region = region
        self.dimension = dimension
        self.namespace = namespace
        self.use_prompt_plus_answer = use_prompt_plus_answer

        self.pc = pinecone.Pinecone(api_key=self.api_key)
        
        # Create index if it doesn't exist
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                spec=pinecone.ServerlessSpec(cloud="aws", region=self.region),
            )
            # Wait for index to be ready
            time.sleep(10)

        self.index = self.pc.Index(self.index_name)

    def add_pair(
        self, 
        prompt: str, 
        answer: str, 
        meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a prompt-answer pair to the database
        
        Args:
            prompt: User prompt/question
            answer: Generated answer
            meta: Additional metadata
            
        Returns:
            ID of the added pair
        """
        pair_id = str(uuid.uuid4())
        
        # Prepare text for embedding
        if self.use_prompt_plus_answer:
            text_to_embed = f"{prompt}\n\n{answer}"
        else:
            text_to_embed = prompt
        
        # Generate embedding
        try:
            embedding = self.embedding_model.embed_query(text_to_embed)
            embedding = list(map(float, embedding))
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Use zero vector as fallback
            embedding = [0.0] * self.dimension
        
        # Prepare metadata
        metadata = {
            "prompt": prompt,
            "answer": answer,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_used": datetime.now(timezone.utc).isoformat(),
            "usage_count": 0,
        }
        
        if meta:
            metadata.update(meta)
        
        # Upsert to Pinecone
        try:
            self.index.upsert(
                vectors=[{
                    "id": pair_id,
                    "values": embedding,
                    "metadata": metadata
                }],
                namespace=self.namespace
            )
            print(f"Added prompt-answer pair: {pair_id}")
            return pair_id
            
        except Exception as e:
            print(f"Error adding pair: {e}")
            raise

    def record_use(self, pair_id: str) -> None:
        """Record usage of a prompt-answer pair"""
        try:
            # Get current metadata
            response = self.index.fetch(ids=[pair_id], namespace=self.namespace)
            if pair_id in response.vectors:
                vector = response.vectors[pair_id]
                metadata = vector.metadata
                
                # Update usage stats
                metadata["last_used"] = datetime.now(timezone.utc).isoformat()
                metadata["usage_count"] = int(metadata.get("usage_count", 0)) + 1
                
                # Re-embed and update
                if self.use_prompt_plus_answer:
                    text_to_embed = f"{metadata['prompt']}\n\n{metadata['answer']}"
                else:
                    text_to_embed = metadata['prompt']
                
                embedding = self.embedding_model.embed_query(text_to_embed)
                embedding = list(map(float, embedding))
                
                self.index.upsert(
                    vectors=[{
                        "id": pair_id,
                        "values": embedding,
                        "metadata": metadata
                    }],
                    namespace=self.namespace
                )
                
        except Exception as e:
            print(f"Error recording use: {e}")

    def query(
        self,
        query_text: str,
        k: int = 5,
        sim_weight: float = 0.6,
        entropy_weight: float = 0.25,
        recency_weight: float = 0.15
    ) -> List[PAPair]:
        """
        Query for similar prompt-answer pairs
        
        Args:
            query_text: Query text
            k: Number of pairs to return
            sim_weight: Weight for similarity scoring
            entropy_weight: Weight for entropy scoring
            recency_weight: Weight for recency scoring
            
        Returns:
            List of (pair, score, score_breakdown) tuples
        """
        # Normalize weights
        total_weight = sim_weight + entropy_weight + recency_weight
        sim_weight /= total_weight
        entropy_weight /= total_weight
        recency_weight /= total_weight
        
        # Generate query embedding
        try:
            query_embedding = self.embedding_model.embed_query(query_text)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []
        
        # Query Pinecone
        try:
            response = self.index.query(
                vector=query_embedding,
                top_k=k * 2,  # Overfetch for better scoring
                include_metadata=True,
                namespace=self.namespace
            )
            
            matches = getattr(response, "matches", []) or []
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []
        
        if not matches:
            return []
        
        # Process and score results
        scored_pairs = []
        
        for match in matches:
            metadata = getattr(match, "metadata", {}) or {}
            similarity_score = getattr(match, "score", 0.0) or 0.0
            
            # Create PAPair object
            pair = PAPair(
                prompt=metadata.get("prompt", ""),
                answer=metadata.get("answer", ""),
                pair_id=match.id,
                metadata=metadata,
                created_at=self._parse_iso(metadata.get("created_at")),
                last_used=self._parse_iso(metadata.get("last_used")),
                usage_count=int(metadata.get("usage_count", 0))
            )
            
            # Calculate scores
            entropy_score = self._calculate_entropy(pair.prompt + " " + pair.answer)
            recency_score = self._calculate_recency(pair.last_used)
            
            # Combine scores
            final_score = (
                sim_weight * similarity_score +
                entropy_weight * entropy_score +
                recency_weight * recency_score
            )
            
            score_breakdown = {
                "sim": similarity_score,
                "ent": entropy_score,
                "rec": recency_score
            }
            
            scored_pairs.append((pair, final_score, score_breakdown))
        
        # Sort by final score and return top k
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:k]

    def _calculate_entropy(self, text: str) -> float:
        """Calculate information entropy score for text"""
        try:
            if not text.strip():
                return 0.0
            
            words = text.lower().split()
            if not words:
                return 0.0
            
            # Simple entropy calculation based on word frequency
            word_counts = {}
            total_words = len(words)
            
            for word in words:
                if word.isalpha():
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            if not word_counts:
                return 0.0
            
            # Calculate entropy
            entropy = 0.0
            for count in word_counts.values():
                p = count / total_words
                if p > 0:
                    entropy -= p * np.log2(p)
            
            # Normalize to 0-1 range
            max_entropy = np.log2(len(word_counts))
            if max_entropy > 0:
                return min(1.0, entropy / max_entropy)
            return 0.0
            
        except Exception:
            return 0.0

    def _calculate_recency(self, last_used: Optional[datetime]) -> float:
        """Calculate recency score from timestamp"""
        try:
            if not last_used:
                return 0.0
            
            now = datetime.now(timezone.utc)
            age_hours = max(0.0, (now - last_used).total_seconds() / 3600.0)
            return 1.0 / (1.0 + age_hours)
            
        except Exception:
            return 0.0

    def _parse_iso(self, iso_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp string"""
        if not iso_str:
            return None
        try:
            return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        except Exception:
            return None

    def clear(self) -> None:
        """Clear the entire index"""
        try:
            self.pc.delete_index(name=self.index_name)
            print(f"Deleted index: {self.index_name}")
        except Exception as e:
            print(f"Error deleting index: {e}")
