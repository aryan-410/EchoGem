# %%
# %%
import sys
print("Python running from:", sys.executable)

import getpass
import os
import sys
import numpy as np
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # ← move this up

from langchain.chat_models import init_chat_model
from langchain.schema.embeddings import Embeddings

from chunker_ex import Chunker
from vector_store_ex import ChunkVectorDB
from Usage_Cache_CSV import UsageCache


# %%
class Processor:
  def __init__(self,
               embedding_model = GoogleGenerativeAIEmbeddings,
               weights : np.ndarray = np.full(7, 1 / np.sqrt(7))):

    # verify validity of api
    self.api_key = os.getenv("GOOGLE_API_KEY")

    if not self.api_key:
        print("Error: GOOGLE_API_KEY environment variable not found")
        print("Please set your API key: export GOOGLE_API_KEY='your-api-key'")
    else:
        print("API key found")

    # create embedding model
    self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document", google_api_key=self.api_key) #type: ignore

    # normalize weights
    if abs(np.linalg.norm(weights) - 1.0) < 0.01:
      weights = weights / np.linalg.norm(weights)

    self.weights = weights

    # create llm
    self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.3,
            max_retries=3,
            timeout=60)

    # create chunker
    self.chunker = Chunker()

    # create vector database and usage cache
    self.vector_db = ChunkVectorDB(self.embedding_model, index_name="my-working-index") #type: ignore
    self.usage_cache = UsageCache("usage_cache_store.csv")

  def chunkAndProcess(self, file_path : str = "transcript.txt"):
    # use chunker to create chunks
    chunks = self.chunker.ChunkTranscript(self.llm, file_path)
    
    # upload chunks to vector db and usage cache
    self.vector_db.vectorize_chunks(chunks)
    self.usage_cache.push_chunks(chunks)
  
  # place holder chunk-picking function
  def pick_chunks(self, prompt : str):
    self.vector_db.pick_chunks(prompt)

if __name__ == "__main__":
   processor = Processor()
   processor.chunkAndProcess("transcript.txt")
   print(processor.pick_chunks("How is Google adding personalization?"))








