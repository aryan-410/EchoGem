# %%
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

from chunk_response_class_ex import Chunk, ChunkResponse

# %%
class Chunker:
  def __init__(
    self,
    embed_model = "all-MiniLM-L6-v2",
    max_tokens: int = 2_000,
    similarity_threshold: float = 0.82,
    coherence_threshold: float = 0.75,):

    self.embedder = SentenceTransformer(embed_model) # type: ignore
    self.max_tokens = max_tokens
    self.sim_threshold = similarity_threshold
    self.coh_threshold = coherence_threshold

  def TranscriptTextMaker(self, file_path):
    try:
      with open(file_path, "r", encoding="utf-8") as f:
        transcript = f.read()
      print(f"Transcript loaded ({len(transcript)} characters)")
    except FileNotFoundError:
        print("Error: transcript.txt file not found")
    except Exception as e:
        print(f"Error loading transcript: {str(e)}")
        return

    return transcript # type: ignore

  def CreatePrompt(self):
    return ChatPromptTemplate.from_template("""
    **SYSTEM PROMPT**
    You are a transcript processing expert. The following transcript needs to be chunked very ingelligently and logically. Ensure sensible segments and structure to be later provided as context to answer questions.

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
    {input_text}

    **OUTPUT FORMAT**
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
    """)

  def ChunkTranscript(self, llm, file_path, output_result: bool = False) -> list[Chunk]:
    transcript = self.TranscriptTextMaker(file_path)
    try:
        prompt = self.CreatePrompt()
        structured_llm = llm.with_structured_output(ChunkResponse)
        chain = prompt | structured_llm
        response = chain.invoke({"input_text": transcript})

        if response is None:
            print("LLM response was None.")
            return []

        print(f"Generated {len(response.chunks)} chunks")
        if output_result:
            for i, chunk in enumerate(response.chunks, 1):
                print(f"  Chunk {i}: {chunk.title} ({chunk.timestamp_range}), {chunk.content}")

        return response.chunks

    except Exception as e:
        print(f"structured output failed: {str(e)}")
        return []






