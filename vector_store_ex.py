# %%
import pinecone
from pinecone import Pinecone

import uuid

from langchain_google_genai  import GoogleGenerativeAIEmbeddings
from langchain.schema.embeddings import Embeddings
from pinecone import ServerlessSpec

import time
# %%
from pydantic import BaseModel, Field
from chunk_response_class_ex import Chunk, ChunkResponse

# %%
class ChunkVectorDB:
  def __init__(self,
               embedding_model : Embeddings,
               api_key : str = "pcsk_8eiAt_DKBYBA3H1mQg3RsGd8qRwcmh7AdGxfet3XxeE3poUVKHEt8Zpbms3q3wgXeD7Ct",
               index_name : str = "dense-index"):

    # initialize pinecone
    self.pc = Pinecone(api_key= api_key)
    self.index_name = index_name
    self.embedding_model = embedding_model

    # create pinecone index
    if not self.pc.has_index(self.index_name):
        self.pc.create_index(
            name=self.index_name,
            dimension=768,
            spec=ServerlessSpec(
              cloud="aws",
              region="us-east-1",
          )
        ) 

    self.index = self.pc.Index(self.index_name)  

  def vectorize_chunks(self, chunks: list[Chunk]):

    texts = [c.content for c in chunks]
    embeddings = [self.embedding_model.embed_query(t) for t in texts]

    vectors = []
    for chunk, vec in zip(chunks, embeddings):
        vec = list(map(float, vec))                 # ensure pure-Python floats

        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": vec,
                "metadata": {
                    "title": chunk.title,
                    "keywords": ", ".join(chunk.keywords),
                    "named_entities": ", ".join(chunk.named_entities),
                    "timestamp_range": chunk.timestamp_range,
                }
            }
        )

    resp = self.index.upsert(vectors=vectors, namespace="chunks")
    print("Upsert response:", resp)

    time.sleep(15)
    stats = self.index.describe_index_stats(namespace="chunks")
    print("Index stats AFTER upsert:", stats)     


    self.index.upsert(namespace="chunks", vectors=vectors)
    print(f"Index stats: {self.index.describe_index_stats(namespace='chunks')}")

  
  def pick_chunks(self, prompt: str, k: int = 5) -> list[Chunk]:
    prompt_embedding = self.embedding_model.embed_query(prompt)

    query_result = self.index.query(
        vector=prompt_embedding,
        top_k=k,
        include_metadata=True,
        namespace="chunks"
    )
    print("Raw Pinecone response:", query_result)

    picked_chunks = []
    for match in query_result.get("matches", []): # type: ignore
        metadata = match["metadata"]
        chunk = Chunk(
            title=metadata.get("title", ""),
            content=metadata.get("chunk_text", ""),  # ✅ watch this key!
            keywords=metadata.get("keywords", []),
            named_entities=metadata.get("named_entities", []),
            timestamp_range=metadata.get("timestamp_range", "")
        )
        picked_chunks.append(chunk)

    print(f"Picked {len(picked_chunks)} chunks for prompt: {prompt}")
    return picked_chunks if picked_chunks else None #type: ignore

  def read_vectors(self):
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

  def delete_index(self):
    self.pc.delete_index(name=self.index_name)


