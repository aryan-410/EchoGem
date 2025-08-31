"""
Vector database operations for storing and retrieving transcript chunks.
"""

import os
import json
import hashlib
from typing import List, Optional, Dict, Any
import pinecone
from sentence_transformers import SentenceTransformer
from .models import Chunk
from .usage_cache import UsageCache


class ChunkVectorDB:
    """
    Pinecone-backed vector database for storing and retrieving transcript chunks.
    
    Features:
    - Intelligent chunk scoring (similarity + entropy + recency)
    - Usage tracking integration
    - Configurable weights for different scoring factors
    """
    
    def __init__(
        self,
        embedding_model,
        api_key: Optional[str] = None,
        index_name: str = "echogem-chunks",
        region: str = "us-east-1",
        dimension: int = 768
    ):
        """
        Initialize the vector database
        
        Args:
            embedding_model: Embedding model for text vectorization
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index
            region: AWS region for the index
            dimension: Vector dimension (default 768 for Google embeddings)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY environment variable or pass api_key parameter.")
        
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.region = region
        self.dimension = dimension

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

    def entropy(self, text: str) -> float:
        """
        Calculate information entropy score for text
        
        Args:
            text: Text to analyze
            
        Returns:
            Entropy score between 0 and 1
        """
        try:
            # Try to load spacy model, fallback to basic analysis
            try:
                nlp = spacy.load("en_core_web_sm")
                use_spacy = True
            except OSError:
                use_spacy = False
                print("Warning: spacy model not found. Using basic entropy calculation.")
            
            if not text.strip():
                return 0.0

            if use_spacy:
                doc = nlp(text)
                sentences = [sent.text for sent in doc.sents]
                words = [token.text.lower() for token in doc if token.is_alpha]
                entities = [ent.text for ent in doc.ents]
            else:
                # Basic fallback
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                words = [w.lower() for w in text.split() if w.isalpha()]
                entities = []

            total_words = len(words)
            if total_words == 0:
                return 0.0

            # Named entity density
            ned = len(entities) / max(1, len(sentences))

            # Lexical diversity
            unique_keywords = len(set(words))
            ld = unique_keywords / max(1, total_words)

            # Semantic variance (if sentence transformers available)
            sv = 0.0
            if sentences and len(sentences) > 1:
                try:
                    embedder = SentenceTransformer("all-MiniLM-L6-v2")
                    embeddings = embedder.encode(sentences)
                    variance = np.mean(np.var(embeddings, axis=0))
                    sv = 1.0 / (1.0 + variance)
                except Exception:
                    # Fallback to basic variance
                    sentence_lengths = [len(s.split()) for s in sentences]
                    if len(sentence_lengths) > 1:
                        variance = np.var(sentence_lengths)
                        sv = 1.0 / (1.0 + variance)

            # Weighted combination
            score = 0.4 * ned + 0.4 * ld + 0.2 * sv
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            print(f"Entropy calculation error: {e}")
            return 0.0

    def vectorize_chunks(self, chunks: List[Chunk]) -> None:
        """
        Convert chunks to vectors and store in Pinecone
        
        Args:
            chunks: List of chunks to vectorize
        """
        if not chunks:
            print("No chunks to vectorize")
            return

        texts = [c.content for c in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.embedding_model.embed_query(text)
                embeddings.append(embedding)
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(texts)} chunks")
            except Exception as e:
                print(f"Error embedding chunk {i}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * self.dimension)

        # Prepare vectors for Pinecone
        vectors = []
        for chunk, vec in zip(chunks, embeddings):
            vec = list(map(float, vec))
            
            # Generate chunk ID if not present
            chunk_id = chunk.chunk_id or str(uuid.uuid4())
            
            vectors.append({
                "id": chunk_id,
                "values": vec,
                "metadata": {
                    "chunk_text": chunk.content,
                    "title": chunk.title,
                    "keywords": json.dumps(chunk.keywords),
                    "named_entities": json.dumps(chunk.named_entities),
                    "timestamp_range": chunk.timestamp_range,
                    "entropy": self.entropy(chunk.content),
                },
            })

        # Upsert to Pinecone
        print(f"Upserting {len(vectors)} vectors to Pinecone...")
        try:
            resp = self.index.upsert(vectors=vectors, namespace="chunks")
            print(f"Upsert successful: {resp}")
            
            # Wait for index to update
            time.sleep(15)
            
            # Show stats
            stats = self.index.describe_index_stats(namespace="chunks")
            print(f"Index stats: {stats}")
            
        except Exception as e:
            print(f"Upsert failed: {e}")
            raise

    def pick_chunks(
        self,
        prompt: str,
        k: int = 10,
        entropy_weight: float = 0.25,
        recency_weight: float = 0.25,
        overfetch: int = 3,
        usage_cache: Optional[UsageCache] = None,
        usage_csv_path: str = "usage_cache_store.csv",
        max_candidate_cap: int = 200
    ) -> Optional[List[Chunk]]:
        """
        Retrieve the most relevant chunks for a prompt
        
        Args:
            prompt: User query
            k: Number of chunks to return
            entropy_weight: Weight for entropy scoring
            recency_weight: Weight for recency scoring
            overfetch: Multiplier for initial retrieval
            usage_cache: Usage cache instance
            usage_csv_path: Path to usage cache CSV
            max_candidate_cap: Maximum candidates to consider
            
        Returns:
            List of relevant chunks or None if no matches
        """
        def _minmax(xs):
            """Normalize values to 0-1 range"""
            if not xs:
                return xs
            lo, hi = min(xs), max(xs)
            if hi - lo < 1e-12:
                return [0.0] * len(xs)
            return [(x - lo) / (hi - lo) for x in xs]

        def _recency_score(last_used_iso: Optional[str]) -> float:
            """Calculate recency score from timestamp"""
            try:
                if not last_used_iso:
                    return 0.0
                ts = datetime.fromisoformat(last_used_iso)
                now = datetime.now(timezone.utc)
                age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
                return 1.0 / (1.0 + age_hours)
            except Exception:
                return 0.0

        def _content_hash(text: str) -> str:
            """Generate hash from content"""
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Initialize usage cache
        uc = usage_cache
        if uc is None:
            try:
                uc = UsageCache(usage_csv_path)
            except Exception as e:
                print(f"[usage] could not open UsageCache: {e}")
                uc = None
        usage_map = uc.get_all_chunks() if uc else {}

        # Validate parameters
        try:
            k = int(k)
        except Exception:
            k = 5
        of = max(1, int(overfetch))
        requested = min(max(k * of, k), max_candidate_cap)

        # Generate query embedding
        prompt_embedding = self.embedding_model.embed_query(prompt)
        
        # Query Pinecone
        qr = self.index.query(
            vector=prompt_embedding,
            top_k=requested,
            include_metadata=True,
            namespace="chunks"
        )
        matches = getattr(qr, "matches", None) or qr.get("matches", [])
        print(f"[pick_chunks] pinecone matches: {len(matches)} (requested {requested})")

        if not matches:
            return None

        # Process candidates
        candidates: List[Tuple[Chunk, float, float, float, str]] = []

        for m in matches:
            md = getattr(m, "metadata", None) or m.get("metadata", {})
            content = md.get("chunk_text", "") or ""
            if not content.strip():
                continue

            stable_id = md.get("chunk_id") or _content_hash(content)
            title = md.get("title", "") or ""
            ts_range = md.get("timestamp_range", "") or ""
            
            # Parse keywords and named entities
            keywords = []
            named_entities = []
            
            try:
                kw_raw = md.get("keywords", [])
                if isinstance(kw_raw, str):
                    keywords = json.loads(kw_raw)
                elif isinstance(kw_raw, list):
                    keywords = kw_raw
            except Exception:
                keywords = []
                
            try:
                ne_raw = md.get("named_entities", [])
                if isinstance(ne_raw, str):
                    named_entities = json.loads(ne_raw)
                elif isinstance(ne_raw, list):
                    named_entities = ne_raw
            except Exception:
                named_entities = []

            # Get similarity score
            sim = getattr(m, "score", None)
            if sim is None and isinstance(m, dict):
                sim = m.get("score", 0.0)
            sim = float(sim or 0.0)

            # Get entropy score
            if "entropy" in md:
                try:
                    ent = float(md["entropy"])
                except Exception:
                    ent = 0.0
            else:
                try:
                    ent = float(self.entropy(content))
                except Exception as e:
                    print(f"[entropy] error: {e}")
                    ent = 0.0

            # Get recency score
            last_used = usage_map.get(stable_id, {}).get("last_used")
            rec = _recency_score(last_used)

            # Create chunk object
            chunk = Chunk(
                title=title,
                content=content,
                keywords=keywords,
                named_entities=named_entities,
                timestamp_range=ts_range,
                chunk_id=stable_id
            )
            candidates.append((chunk, sim, ent, rec, stable_id))

        print(f"[pick_chunks] candidates built: {len(candidates)}")

        if not candidates:
            return None

        # Normalize scores
        sims = [c[1] for c in candidates]
        ents = [c[2] for c in candidates]
        recs = [c[3] for c in candidates]
        sims_n, ents_n, recs_n = _minmax(sims), _minmax(ents), _minmax(recs)

        # Apply weights
        w_ent = max(0.0, min(1.0, float(entropy_weight)))
        w_rec = max(0.0, min(1.0, float(recency_weight)))
        w_sim = max(0.0, 1.0 - (w_ent + w_rec))

        # Calculate final scores
        scored = []
        for (chunk, sim, ent, rec, sid), sn, en, rn in zip(candidates, sims_n, ents_n, recs_n):
            final = w_sim * sn + w_ent * en + w_rec * rn
            scored.append((final, sim, rec, ent, chunk, sid))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        # Deduplicate and pick top k
        seen: set[str] = set()
        picked: List[Chunk] = []
        picked_ids: List[str] = []

        for item in scored:
            chunk, sid = item[4], item[5]
            if sid in seen:
                continue
            seen.add(sid)
            picked.append(chunk)
            picked_ids.append(sid)
            if len(picked) >= k:
                break

        print(f"[pick_chunks] after dedupe -> picked: {len(picked)} (requested k={k})")

        # Update usage cache
        if uc:
            try:
                for sid in picked_ids:
                    uc.update_usage(sid)
            except Exception as e:
                print(f"[usage] update failed: {e}")

        return picked or None

    def read_vectors(self) -> None:
        """Read and display all vectors in the index"""
        all_ids = []

        for page in self.index.list_vectors():
            ids = [v.id for v in page.vectors]
            all_ids.extend(ids)

        print(f"Total IDs: {len(all_ids)}")

        batch_size = 100
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i+batch_size]
            response = self.index.fetch(ids=batch_ids)
            for vector_id, vector in response.vectors.items():
                print(f"\nID: {vector_id}")
                print(f"Values: {vector.values}")
                print(f"Metadata: {vector.metadata}")

    def clear(self) -> None:
        """Clear the entire index"""
        try:
            self.pc.delete_index(name=self.index_name)
            print(f"Deleted index: {self.index_name}")
        except Exception as e:
            print(f"Error deleting index: {e}")
