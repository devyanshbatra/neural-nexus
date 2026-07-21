from state.schema import NexusState

MAX_RETRIES = 2  # max re-retrieval attempts before giving up


def route_after_retriever(state: NexusState) -> str:
    """
    Conditional edge 1 + 2: check if vector/graph stores are empty.

    - Both empty → error (no data ingested)
    - Vector empty only → skip to graph-only path (still useful)
    - Graph empty only → skip conflict_detector + schema_reader (nothing to scan)
    - Both have data → full pipeline
    """
    vector_empty = state.get("vector_empty", False)
    graph_empty  = state.get("graph_empty",  False)

    if vector_empty and graph_empty:
        print("[Router] Both stores empty → error node")
        return "error_node"

    if graph_empty:
        print("[Router] Graph empty → skip conflict/schema, go straight to combiner")
        return "combiner"

    # vector_empty or both full → run conflict detector
    # (conflict detection only needs Neo4j, not Chroma)
    return "conflict_detector"


def route_after_conflict(state: NexusState) -> str:
    """
    Conditional edge 3: skip schema_reader if no conflicts AND graph is empty.
    Graph empty already handled above — if we're here, graph has data.
    Always run schema_reader (fast, useful for answer quality).
    """
    return "schema_reader"


def route_after_answer(state: NexusState) -> str:
    """
    Conditional edge 4: confidence-based re-retrieval loop.

    - confidence < 0.4 AND retry_count < MAX_RETRIES → re-run retriever
    - otherwise → END
    """
    confidence  = state.get("confidence", 0.5)
    retry_count = state.get("retry_count", 0)

    if confidence < 0.4 and retry_count < MAX_RETRIES:
        print(f"[Router] Low confidence ({confidence:.0%}), retry {retry_count + 1}/{MAX_RETRIES}")
        return "retriever"

    if confidence < 0.4:
        print(f"[Router] Low confidence after {MAX_RETRIES} retries → END anyway")

    return END_NODE


# Sentinel used by graph.py to wire the END edge
END_NODE = "__end__"
