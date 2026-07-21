import operator
from typing import Annotated, List
from typing_extensions import TypedDict


class NexusState(TypedDict):

    # ── Query ─────────────────────────────────────────────────────────────
    query: str                          # user's question

    # ── Retrieval (hybrid search) ──────────────────────────────────────────
    vector_results: str                 # semantic chunks from Chroma
    graph_results: str                  # nodes + relationships from Neo4j
    combined_context: str               # merged vector + graph for answer agent

    # ── Innovation 1: Auto-Schema ──────────────────────────────────────────
    schema_types: dict                  # entity/relationship types LLM discovered
                                        # {"entities": ["CI_TOOL", "SERVER"],
                                        #  "relationships": ["DEPLOYS_TO"]}

    # ── Innovation 2: Conflict Detection ──────────────────────────────────
    conflicts: List[dict]               # contradictions found across documents
                                        # [{"fact1": "...", "fact2": "...",
                                        #   "source1": "...", "source2": "..."}]

    # ── Answer ────────────────────────────────────────────────────────────
    answer: str                         # final synthesized answer
    sources: List[str]                  # docs/nodes answer came from
    confidence: float                   # answer confidence score (0.0 - 1.0)

    # ── Tracking ──────────────────────────────────────────────────────────
    current_agent: str
    agent_history: Annotated[list, operator.add]  # accumulates across all nodes
    error: str

    # ── Conditional routing flags ──────────────────────────────────────────
    vector_empty: bool          # True if Chroma has no results
    graph_empty: bool           # True if Neo4j has no results
    retry_count: int            # how many re-retrieval attempts so far
