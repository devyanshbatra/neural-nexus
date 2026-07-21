# NeuralNexus 🧠

> RAG + Knowledge Graph powered company knowledge base using LangChain, LangGraph, Chroma, and Neo4j.

## What it does

Ask questions about your company documents. NeuralNexus retrieves answers using **two sources simultaneously**:
- **Vector search** (Chroma) — semantic similarity over document chunks
- **Knowledge graph** (Neo4j) — entity relationships extracted from documents

Then detects contradictions across documents and flags them in the answer.

## Innovations

1. **Auto-Schema Discovery** — LLM invents its own entity and relationship types per domain. No hardcoded schema. Works for any company, any industry.
2. **Conflict Detection** — finds when two documents disagree on the same fact (e.g., Doc A says "Jenkins deploys to AWS", Doc B says "Jenkins deploys to GCP").
3. **Confidence-Based Retry** — if answer confidence is low, graph automatically retries retrieval with broader search (max 2 retries).

## Architecture

```
INGEST TIME (run once):
  documents/ ──→ extractor agent ──→ Neo4j knowledge graph
  documents/ ──→ chunk + embed   ──→ Chroma vector store

QUERY TIME (every question):
  user query
      │
      ▼
  retriever (Chroma + Neo4j)
      │
      ├─ both empty ──→ error node ──→ END
      ├─ graph empty ──→ combiner
      └─ has graph ──→ conflict_detector ──→ schema_reader ──→ combiner
                                                                    │
                                                                 answer
                                                                    │
                                              confidence < 40% ──→ retry (max 2x)
                                              confidence ≥ 40% ──→ END
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Ollama (llama3.2) — local, free |
| Embeddings | Ollama (nomic-embed-text) |
| Vector Store | Chroma (persistent) |
| Knowledge Graph | Neo4j |
| Orchestration | LangGraph |
| Framework | LangChain |

## Project Structure

```
neural_nexus/
├── agents/
│   ├── extractor.py      # extracts entities + relationships from docs → Neo4j
│   ├── retriever.py      # hybrid vector + graph search
│   └── answer.py         # synthesizes final answer
├── graph/
│   ├── nodes.py          # LangGraph nodes
│   ├── edges.py          # conditional routing logic
│   └── graph.py          # assembles + compiles LangGraph pipeline
├── graph_store/
│   ├── neo4j_client.py   # Neo4j driver, Cypher queries, conflict detection
│   └── builder.py        # writes extracted entities to Neo4j
├── vector_store/
│   ├── store.py          # Chroma search wrapper
│   └── setup.py          # document loading + chunking + embedding
├── prompts/
│   └── templates.py      # all LLM prompts
├── state/
│   └── schema.py         # LangGraph state definition
├── config.py             # all settings
├── ingest.py             # run once to process documents
└── main.py               # interactive Q&A loop
```

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- [Neo4j Desktop](https://neo4j.com/download/) running locally

### 1. Install models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```
LANGCHAIN_API_KEY=your_langsmith_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### 4. Add documents

Create a `documents/` folder and add your `.pdf` or `.txt` files:
```
documents/
├── company_handbook.pdf
├── tech_stack.txt
└── org_chart.pdf
```

### 5. Ingest documents

```bash
python ingest.py
```

Add `--clear` to wipe and re-ingest:
```bash
python ingest.py --clear
```

### 6. Ask questions

```bash
python main.py
```

```
You: Who manages the infrastructure team?
You: What CI/CD tools do we use?
You: sources    ← lists all ingested documents
You: quit
```

## LangGraph Flow

5 nodes with 4 conditional edges:

| Edge | Condition | Routes to |
|---|---|---|
| After retriever | both stores empty | error node |
| After retriever | graph empty | combiner (skip conflict/schema) |
| After retriever | graph has data | conflict_detector → full pipeline |
| After answer | confidence < 40% + retries left | retriever (broader search) |
| After answer | confident or max retries hit | END |

## LangSmith Tracing

Set in `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=neural-nexus
```

---

Built as part of a LangChain + LangGraph learning series.
