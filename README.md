# EchoGem: Teaching Gemini to Think in Batches by Prioritizing What Matters

📋 GSoC Progress Tracker: Complete project timeline and accountability document used throughout GSoC 2025: View Progress Tracker

PyPI · Google Summer of Code · Google DeepMind
Python · MIT License
A Google DeepMind GSoC 2025 Project | Powered by Google Gemini AI | Built with Python

**Mentors:** Prof. Suguman Bansal, DeepMind Research Team
**Program:** Google Summer of Code
**Organization:** Google DeepMind
**Student:** Aryan Saboo
**Email:** [aryan.saboo@gatech.edu](mailto:aryan.saboo@gatech.edu)
**Duration:** May 2025 – September 2025

---

## 📖 GSoC 2025: Project Goals & Problem Statement

### Research Challenge & Goals

Processing **long-form transcripts** with LLMs like Gemini is computationally expensive, redundant, and error-prone. Traditional approaches suffer from:

* High API costs due to naive chunking and excessive token use
* Poor scalability for transcripts exceeding context length
* Weak context preservation leading to fragmented answers
* Lack of intelligent re-use of previously processed content

### Project Goals & Requirements

* **Design a smart chunking engine** that segments transcripts based on semantic boundaries, keywords, and named entities.
* **Create a weighted retrieval system** balancing semantic similarity, entropy (information density), and recency of use.
* **Implement a caching system** for both chunks and Q\&A pairs to eliminate redundant processing.
* **Build an interactive graph UI** showing chunk usage, prompt→answer links, and similarity relations.
* **Deliver a production-ready Python package** with CLI and PyPI distribution.

---

## 🚀 What I Built During GSoC 2025

### 1. Intelligent Chunking System (`chunker.py`)

* **LLM-guided chunking:** Gemini converts transcripts into structured JSON `{title, content, keywords, entities, timestamps}`.
* **Resilient fallback splitter:** Ensures progress even if model output is malformed.
* **Local embeddings:** Sentence-transformers support local vectorization.

### 2. Context-Aware Retrieval (`vector_store.py`)

* **Weighted scoring:** Combines semantic similarity, entropy, and recency.
* **Entropy estimator:** Uses lexical diversity, NER density, and semantic variance.
* **Recency-aware ranking:** Chunks recently used in answers get priority.
* **Overfetching & normalization:** Improves selection by broad candidate sampling.

### 3. Usage & Memory Systems (`usage_cache.py`, `prompt_answer_store.py`)

* **Usage cache:** CSV-based system with rich metadata (title, keywords, usage\_count, last\_used).
* **Prompt→Answer memory:** Stores Q\&A pairs with chunk IDs in Pinecone.
* **Automatic upgrades:** Legacy CSV schemas auto-migrate.

### 4. Graph Visualizer (`graphe.py`)

* **Interactive UI:** Built with pygame for exploring chunks and Q\&A relationships.
* **Node types:** Chunk nodes and prompt→answer nodes.
* **Edge types:** Similarity, temporal adjacency, and usage connections.
* **Layouts:** Force-directed, circular, and hierarchical modes.

### 5. Processor & CLI (`processor.py`, `cli.py`)

* **Processor orchestrator:** High-level API to `process_transcript` and `query`.
* **CLI commands:** `process`, `query`, and `graph` for quick interaction.
* **Cross-platform compatibility:** Verified on Windows, macOS, Linux.

---

## 🎯 Current State & What's Working Now

### Achievements

| Component          | Status     | Details                       |
| ------------------ | ---------- | ----------------------------- |
| Core Chunking      | ✅ Complete | LLM-guided with fallback      |
| Weighted Retrieval | ✅ Complete | Semantic + entropy + recency  |
| Usage Cache        | ✅ Complete | CSV store with upgrades       |
| Q\&A Memory        | ✅ Complete | Pinecone namespace            |
| Graph UI           | ✅ Complete | Interactive pygame visualizer |
| CLI                | ✅ Complete | Process, query, visualize     |
| PyPI Package       | ✅ Complete | Published as `echogem`        |
| Cross-Platform     | ✅ Complete | Works on Win/macOS/Linux      |

### Live Features

* Transcript ingestion & chunking
* Weighted chunk retrieval & scoring
* Q\&A memory and caching
* Interactive chunk/answer visualization
* CLI and Python API

---

## 🔗 Code Availability & Distribution

* **PyPI Package:** [https://pypi.org/project/echogem/](https://pypi.org/project/echogem/)
* **Install:** `pip install echogem`
* **Repository:** GitHub (public)
* **License:** MIT

---

## 🛠️ What’s Left to Do – Future Enhancements

* **Coherence-aware selection:** Ensure diverse, non-redundant chunk sets.
* **Context reuse:** Prioritize chunks from successful past answers.
* **Adaptive normalization:** Smarter entropy weighting.
* **Streaming integration:** Handle live transcripts.

---

## 💡 Key Challenges & Learnings

* **Thread-unsafety of Pinecone/Sylvan:** Had to design cache and async logic carefully.
* **Entropy heuristics:** NER density + lexical diversity proved more reliable than token length.
* **CLI vs Processor mismatch:** Adjusted expectations; Processor is the stable API.
* **Graph rendering:** Needed multiple layouts to handle large transcript visualizations.

---

## 📊 Performance & Results

| Metric          | Traditional     | EchoGem                        | Improvement             |
| --------------- | --------------- | ------------------------------ | ----------------------- |
| Context reuse   | None            | Cached Q\&A pairs              | Major                   |
| Chunk selection | Similarity only | Similarity + entropy + recency | Higher precision        |
| Scalability     | Linear          | Overfetch + normalization      | Efficient               |
| Visualization   | None            | Interactive graph              | Debuggable, transparent |

---

## 📈 Development Timeline & Milestones

**Phase 1: Foundation (May 2025)**
✅ Repository setup & API key config
✅ Basic chunking + embeddings

**Phase 2: Core Systems (June 2025)**
✅ Weighted retrieval + entropy estimator
✅ Usage cache + Q\&A store

**Phase 3: Visualization & CLI (July 2025)**
✅ Graph visualizer
✅ CLI commands (process, query, graph)

**Phase 4: Release (August 2025)**
✅ PyPI publication (`echogem`)
✅ Documentation & testing
✅ Final code review & GSoC submission

---

## 🏆 GSoC 2025 Final Deliverables

| Deliverable      | Description                       | Status            |
| ---------------- | --------------------------------- | ----------------- |
| Core Package     | Production-ready Python package   | ✅ `echogem/`      |
| PyPI Release     | Published worldwide               | ✅ PyPI: `echogem` |
| Documentation    | Full guides, API reference        | ✅ `docs/`         |
| Graph Visualizer | Interactive chunk/answer explorer | ✅ `graphe.py`     |
| CLI              | Command-line interface            | ✅ `cli.py`        |
| Progress Tracker | Week-by-week logs                 | ✅ Completed       |

---

## 📚 Citation & Resources

If you use EchoGem in research, please cite:

```bibtex
@software{saboo2025echogem,
  author = {Saboo, Aryan},
  title = {EchoGem: Teaching Gemini to Think in Batches by Prioritizing What Matters},
  year = {2025},
  publisher = {Google DeepMind},
  journal = {Google Summer of Code 2025},
  url = {https://github.com/aryansaboo/echogem}
}
```

---

## 💻 Installation & Usage

### Install from PyPI

```bash
pip install echogem
```

### Set API Keys

```bash
export PINECONE_API_KEY=...
export GOOGLE_API_KEY=...
```

### Quickstart

```python
from echogem.processor import Processor

p = Processor()
p.process_transcript("transcript.txt", persist=True)
ans = p.query("What did the speaker say about scaling?", k=5)
print(ans.answer)
```

### CLI

```bash
echogem process --transcript transcript.txt --persist
echogem query --question "Key takeaways?"
echogem graph
```

---

## 📄 License & Attribution

**License:** MIT
**Developed by:** Aryan Saboo during Google Summer of Code 2025 at Google DeepMind
**Repository:** [https://github.com/aryansaboo/echogem](https://github.com/aryansaboo/echogem)

Acknowledgments:

* Google Summer of Code program
* Google DeepMind for mentorship
* Gemini Team for API access
* Open Source Community for embeddings, Pinecone, and pygame

---

🌟 **GSoC 2025 Success Story** 🌟
From research proposal to **production-ready PyPI package**, EchoGem makes **long-context retrieval practical, efficient, and transparent**.
