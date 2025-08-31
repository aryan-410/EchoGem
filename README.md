# EchoGem — Teaching Gemini to Think in Batches by Prioritizing What Matters

> A modular, measurable long-context retrieval & batching engine with an interactive graph UI for chunk usage and prompt→answer links — designed for real, large transcripts and real constraints.

This README documents the public codebase you have here (which slightly diverges from the original proposal in naming and a few implementation details) and explains how to run, extend, and troubleshoot it on a fresh machine—especially on Windows with PowerShell. It also summarizes the intent and design as laid out in the proposal PDF that accompanies this repo.&#x20;

---

## Table of contents

* [Why EchoGem?](#why-echogem)
* [High-level architecture](#high-level-architecture)
* [What’s in this repo (scripts & folders)](#whats-in-this-repo-scripts--folders)
* [Quickstart (Windows / macOS / Linux)](#quickstart-windows--macos--linux)
* [Running the interactive graph](#running-the-interactive-graph)
* [How the pipeline works](#how-the-pipeline-works)
* [Configuration & environment](#configuration--environment)
* [Data files written at runtime](#data-files-written-at-runtime)
* [Understanding the graph messages](#understanding-the-graph-messages)
* [Common errors & fixes (Windows-focused)](#common-errors--fixes-windowsfocused)
* [Extending EchoGem (swapping components)](#extending-echogem-swapping-components)
* [Roadmap](#roadmap)
* [License & acknowledgments](#license--acknowledgments)

---

## Why EchoGem?

Long transcripts are messy. If you naively dump everything into an LLM, you waste tokens, wait longer, and risk muddier answers. EchoGem turns context construction into a **measurable, tunable** process:

* **Chunk smartly** (topic-coherent segments, not blind windows)
* **Pick ruthlessly** (only what’s relevant, recent, and information-dense)
* **Batch intentionally** (group questions to reuse context)
* **Prove it** (track token use, latency, and coherence—don’t just guess)

The design centers on modular components—**Chunker**, **RelevantInformationHandler**, **PreviousContextHandler**, and a coordinating **Processor**—so each can be swapped, benchmarked, and improved independently. The included **graph UI** shows what’s actually being picked and co-used, with similarity-gated edges so you can *see* the bridges being built (or skipped).

---

## High-level architecture

```
 transcript.txt
      │
      ▼
  Chunker (semantic segmentation, metadata, embeddings)
      │
      ├──► Vector store (e.g., Pinecone)  ── stores chunk vectors
      │
      ▼
 RelevantInformationHandler (scores chunks vs. question)
      │         ▲
      │         └─ PreviousContextHandler (reuses good past Q↔A contexts)
      ▼
   Processor (batches prompts, builds final context, calls LLM)
      │
      └─► Logging:   usage_cache_store.csv       (chunks and usage)
                     pick_log.jsonl              (co-picked chunk IDs)
                     pa_pick_log.jsonl           (picked prompt→answer IDs)
      │
      └─► Interactive Graph (pygame): see chunks/PA & bridges
```

Key ideas that drive choices here—semantic chunking, entropy, coherence, recency weighting, and adaptive scoring—are drawn from the proposal.&#x20;

---

## What’s in this repo (scripts & folders)

> Names in your working tree may differ slightly from the proposal; this README reflects the code paths that actually run.

* `processor_ex.py`
  The orchestrator. Wires together chunking, retrieval, vector DB, and PA storage. Exposes methods used by the graph app (e.g., `pick_chunks`, `answer_with_chunks_and_log`).

* `chunker_ex.py`
  Turns transcripts into coherent chunks and metadata. May compute entities/keywords, entropy, and embeddings.

* `vector_store_ex.py`
  Thin wrapper over the vector database (Pinecone in the default setup). Provides `vectorize_chunks`, `get_vector`, `embed`, `upsert`, `query`, etc.

* `prompt_answer_store.py`
  Stores and retrieves prompt→answer (PA) pairs for reuse and pa-graph visualization.

* `grapher.py` or `chunk_graph.py`
  Pygame UI that shows:

  * **Chunks tab:** picked nodes, with bridges only if cosine similarity ≥ threshold
  * **PA tab:** prompt→answer nodes and links from co-picks

  Either filename may exist in your tree; both implement the same UI contract. The code you pasted indicates `chunk_graph.py`; your run logs show `grapher.py` was used. Use whichever is present.

* Notebooks / prototypes (optional):
  E.g., `scoring_framework.ipynb`, `relevancy.ipynb`, `interaction_handler.ipynb`. These contain the experimental scaffolding that fed the production Python modules.

---

## Quickstart (Windows / macOS / Linux)

### 0) Requirements

* **Python 3.12 (recommended).**
  Python 3.13 works for many libs but native wheels and some SDKs are still catching up; 3.12 is the smooth path for Windows.

* A **Pinecone** account & API key (for vector storage), or adapt `vector_store_ex.py` to a local/vector alternative.

* A **Google Generative AI** key (`GOOGLE_API_KEY`) for Gemini (used via `google-generativeai` / `langchain_google_genai`).

* A transcript file (e.g., `transcript.txt`) to index and query.

### 1) Create a clean virtual environment

**Windows (PowerShell):**

```powershell
cd C:\Users\aryan\Documents\EchoGem
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate
python -m pip install -U pip
```

**macOS/Linux:**

```bash
cd ./EchoGem
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -U pip
```

### 2) Install Python dependencies (known-good set)

> These pins match combinations that worked in the logs you shared and avoid common wheel/API mismatches on Windows.

```bash
# Core numerical & utils
pip install numpy==1.26.4

# Pydantic pair (MUST match to avoid conflicts)
pip install --only-binary=:all: pydantic==2.11.7 pydantic-core==2.33.2

# LLM plumbing
pip install google-generativeai==0.7.2 langchain==0.3.27 langchain-core==0.3.75 langchain-google-genai==2.0.7

# Vector store (avoid the old 'pinecone-client')
pip uninstall -y pinecone-client || true
pip install --only-binary=:all: pinecone==7.2.0

# NLP stack (spaCy + small English model)
pip install spacy==3.7.2 srsly==2.4.8
python -m spacy download en_core_web_sm

# UI
pip install pygame==2.5.2
```

> If you’re on Python 3.13, ensure `pydantic==2.11.7` *pairs with* `pydantic-core==2.33.2`. Mismatched pairs cause `ResolutionImpossible` or `ModuleNotFoundError: pydantic`.

### 3) Set required environment variables

**Windows (PowerShell):**

```powershell
setx GOOGLE_API_KEY "your_google_api_key_here"
setx PINECONE_API_KEY "your_pinecone_api_key_here"
# optional: region/env specifics depending on your vector_store_ex.py configuration
```

**macOS/Linux:**

```bash
export GOOGLE_API_KEY="your_google_api_key_here"
export PINECONE_API_KEY="your_pinecone_api_key_here"
```

> Restart your shell (or start a new PowerShell) so the variables are visible to the venv.

### 4) Put a transcript in the repo root

Example:

```
EchoGem/
  transcript.txt      <-- your long transcript (lecture, podcast, debate, etc.)
```

---

## Running the interactive graph

Run whichever entrypoint exists in your tree:

**Option A (grapher.py):**

```bash
python .\grapher.py --transcript transcript.txt --persist
```

**Option B (chunk\_graph.py):**

```bash
python chunk_graph.py --transcript transcript.txt --chunks usage_cache_store.csv --events pick_log.jsonl --persist
```

### Keyboard & mouse cheatsheet (in the UI)

* **G** → type a prompt to test retrieval
* **1 / 2** → switch tabs (Chunks / PA Pairs)
* **B** → toggle edge counts
* **R** → reshuffle node positions (tab only)
* **S** → save current positions (per-tab)
* **L** → reload data from disk (CSV / JSONL)
* **Esc / Q** → quit
* **Left-drag** → move a node
* **Right-drag** → pan; **Wheel** → zoom

### What to expect on first run

* The app **chunks** your transcript and **vectorizes** those chunks.
* It **upserts** vectors to the store.
* It **persists** chunk metadata to `usage_cache_store.csv`.
* When you query (press **G**), it:

  * picks top candidate chunks and PA neighbors,
  * logs co-picked IDs to `pick_log.jsonl` / `pa_pick_log.jsonl`,
  * draws bridges only when cosine similarity ≥ the threshold (default `0.35`).

---

## How the pipeline works

### Chunker (from `chunker_ex.py`)

* Produces topic-coherent chunks (semantic boundaries, not fixed windows)
* Computes metadata: keywords, entities, entropy, etc.
* Embeds chunk text (via the configured embedding backend)
* Upserts vectors using `vector_store_ex.py`

### RelevantInformationHandler

* Scores chunks against the query with a **blend** of:

  * semantic similarity (cosine of embeddings),
  * lexical overlap (e.g., TF-IDF),
  * information value (entropy),
  * contextual coherence & cluster redundancy penalties,
  * reweighting to avoid any single metric dominating.
* Picks a **balanced** set of chunks (not just many near-duplicates).

### PreviousContextHandler

* Reuses previous prompt→answer “contexts” when semantically connected.
* Applies **exponential recency decay** so stale lines of inquiry fade out.

### Processor (from `processor_ex.py`)

* Orchestrates batching & reuse.
* Enforces token/size budgets for context.
* Logs usage & metrics.
* Exposes methods used by the UI:

  * `pick_chunks(prompt)`
  * `answer_with_chunks_and_log(prompt, k=...)`
  * (optionally) `pick_pa_pairs(prompt, k=...)`

> These roles and the scoring/entropy/decay ideas derive from the proposal’s modular plan and testing philosophy.&#x20;

---

## Configuration & environment

Most knobs live inside `processor_ex.py`, `vector_store_ex.py`, and the handlers. Look for:

* **Vector store settings** (index name, namespace, dimensions)
* **Similarity thresholds** for bridges (the graph app also accepts `--sim-threshold`)
* **Top-k & window sizes** for candidate retrieval
* **Decay half-life** and recency multipliers
* **Scoring weights** (semantic, lexical, entropy, coherence)

Set API keys in your environment (see Quickstart). If you keep per-machine secrets, consider a `.env` approach and load with `dotenv` in `processor_ex.py`.

---

## Data files written at runtime

* `usage_cache_store.csv`
  Canonical CSV for chunks and light usage stats (title, content, entities, last\_used, usage\_count, …)

* `pick_log.jsonl`
  Each line: `{"timestamp": "...", "picked_chunk_ids": [...]}`
  The grapher converts these rows into co-usage bridges **only if** it can verify cosine similarity between vectors ≥ threshold.

* `pa_pick_log.jsonl`
  Same idea, but for **prompt→answer** nodes.

* `positions_chunks_truth_v3.json`, `positions_pa_truth_v3.json`
  Per-tab node layout persistence (so your graph doesn’t reshuffle every run).

---

## Understanding the graph messages

You’ll see lines like:

* `Upsert response: {'upserted_count': 26}`
  ✅ Your chunk vectors reached the store.

* `[pick/chunk] <id> via mapped_by_content | usage=...`
  A selected chunk was matched either by its explicit `chunk_id` or by the content hash.

* `[bridge/chunk] A <-> B -> count=... sim=0.61`
  A co-pick became an **edge** because the cosine similarity of the two chunk vectors is ≥ threshold.

* `[bridge/skip] A <-> B (sim=? < 0.350)` or `[sim] Missing vectors for A or B; skipping edge (strict).`
  The app **refuses** to draw edges it can’t justify by similarity. If vectors are missing:

  * ensure `vector_db.vectorize_chunks(chunks)` was called on the same IDs the app uses,
  * make sure your vector store **namespace/index** matches what retrieval uses,
  * check `get_vector(cid)` in `vector_store_ex.py` returns the same ID you log.

---

## Common errors & fixes (Windows-focused)

### `ModuleNotFoundError: No module named 'pydantic'`

* You likely mixed versions or venvs. In your active venv:

  ```powershell
  pip uninstall -y pydantic pydantic-core
  pip cache purge
  Remove-Item -Recurse -Force .\.venv*\Lib\site-packages\pydantic* -ErrorAction SilentlyContinue
  pip install --only-binary=:all: pydantic==2.11.7 pydantic-core==2.33.2
  python -c "import pydantic, pydantic_core; print(pydantic.__version__, pydantic_core.__version__)"
  ```

### Pinecone import confusion

* Uninstall the **old** package and ensure you don’t shadow it with a local file/folder:

  ```powershell
  pip uninstall -y pinecone-client
  Get-ChildItem -Force -Name | Where-Object { $_ -like "pinecone*" }   # should show nothing project-local
  pip install --only-binary=:all: pinecone==7.2.0
  python -c "import pinecone, importlib; m=importlib.import_module('pinecone'); print(getattr(m,'__file__',None))"
  ```
* If you use gRPC features, import from `pinecone.grpc` (SDKs evolve; this avoids top-level alias surprises).

### `ModuleNotFoundError: No module named 'srsly.ujson.ujson'`

* Ensure `srsly` is present and matches spaCy:

  ```powershell
  pip install -U srsly==2.4.8 spacy==3.7.2
  python -m spacy download en_core_web_sm
  ```
* SpaCy model compatibility warnings (3.8.0 model on 3.7.2 core) are usually harmless; align versions if you prefer.

### PowerShell quoting gotchas

* Prefer **double quotes** around `python -c "..."`.
  The here-doc style `python - <<'PY'` is **Bash**, not PowerShell.

### `Missing vectors for A or B; skipping edge (strict).`

* The graph requests vectors by chunk ID. Confirm:

  * the **same IDs** were used when upserting and when logging picks,
  * your `get_vector(cid)` path returns a list/array (not `None`),
  * you didn’t change vector namespace/index between vectorization and querying.

---

## Extending EchoGem (swapping components)

EchoGem is explicitly **componentized** so you can replace parts without rewriting everything.

* **Swap the Chunker:**
  Provide a class with `build_chunks(...)` and `vectorize_chunks(...)`. Keep chunk IDs stable (e.g., `chunk_id` or content hash via `sha256`) so downstream logs and the graph can resolve them.

* **Try different scoring:**
  Adjust weights or inject your own `RelevantInformationHandler` that implements

  * `score_chunks_against(question)`
  * `goodness(...)` (combine semantic similarity, entropy, redundancy penalties, etc.)

* **Change the vector store:**
  Re-implement `vector_store_ex.py` against FAISS, Qdrant, Chroma, etc. Keep the same interface: `vectorize_chunks`, `get_vector`, `query`, and `embed`.

* **Tune reuse and decay:**
  In `PreviousContextHandler`, tweak exponential decay half-life or the inclusion threshold to match your task/session lengths.

* **Batching strategies:**
  In the `Processor`, experiment with `cluster_questions(...)` and dynamic prompt reuse to reduce per-question cost.

When you introduce a new module, also **log** what it does: the graph and the CSV/JSONL logs are designed to let you *see* the effect of your changes.

---

## Roadmap

* **Benchmark harness**: sweep over chunker/scorer hyper-parameters; unify logging for latency, token-cost, and QA quality checks.
* **Alternate vector backends**: out-of-the-box Qdrant/FAISS adapters.
* **Richer PA graph**: visualize prompt text snippets on hover; color edges by time or frequency.
* **Domain presets**: lecture, interview, debate—prebaked weights for entropy vs. redundancy.
* **Model-agnostic embeddings**: configurable embedding providers (Gemini, local, OpenAI, etc.).

(These items echo the plan described in the proposal while reflecting what’s already in code.)&#x20;

---

## License & acknowledgments

* Copyright © Aryan.
* Portions of the design and text are adapted from the EchoGem proposal included with this repository.&#x20;

---

### Final notes

* This README intentionally documents **what actually works** in the scripts you’re running, even when names or pins differ from the original write-up.
* If you change filenames (`grapher.py` vs. `chunk_graph.py`), keep the CLI flags consistent with the examples above.
* If you run into anything not covered here, drop the exact error and your `pip show` versions for `pydantic`, `pydantic-core`, `pinecone`, `spacy`, and `srsly`. That’s 90% of setup pain on Windows.
