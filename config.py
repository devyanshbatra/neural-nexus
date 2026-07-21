import os

# ── Models ────────────────────────────────────────────────────────────────────
LLM_MODEL   = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
TEMPERATURE = 0.0          # deterministic for extraction + retrieval
ANSWER_TEMPERATURE = 0.3   # slight creativity for final answer

# ── Chroma ────────────────────────────────────────────────────────────────────
CHROMA_PATH   = "chroma_db"
DOCUMENTS_DIR = "documents"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K_VECTOR  = 5          # chunks returned from vector search

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
TOP_K_GRAPH    = 10        # relationships returned from graph traversal

# ── Innovations ───────────────────────────────────────────────────────────────
# Innovation 1: auto-schema — LLM discovers entity/relationship types from docs
AUTO_SCHEMA = True

# Innovation 2: conflict detection — flag contradictions across documents
CONFLICT_DETECTION = True
CONFLICT_THRESHOLD = 0.85  # similarity score above which two facts may conflict

# ── Graph retrieval ───────────────────────────────────────────────────────────
MAX_HOPS = 3               # how deep to traverse graph relationships
