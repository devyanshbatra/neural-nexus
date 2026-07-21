from state.schema import NexusState
from agents.retriever import run_retriever
from agents.answer import run_answer
from graph_store.neo4j_client import find_conflicts, get_all_schema_types
from config import CONFLICT_DETECTION, AUTO_SCHEMA


def retriever_node(state: NexusState) -> dict:
    """
    Hybrid retrieval: vector search + graph traversal.
    Sets vector_empty + graph_empty flags for conditional routing.
    On retry: passes retry_count so retriever can broaden search.
    """
    retry_count = state.get("retry_count", 0)
    print(f"\n[Node: retriever] Running hybrid search (attempt {retry_count + 1})...")

    result = run_retriever(state["query"], retry_count=retry_count)

    vector_results = result["vector_results"]
    graph_results  = result["graph_results"]

    vector_empty = (
        not vector_results or
        vector_results == "No relevant documents found."
    )
    graph_empty = (
        not graph_results or
        graph_results == "No graph relationships found."
    )

    if vector_empty:
        print("[Node: retriever] Vector store empty or no results.")
    if graph_empty:
        print("[Node: retriever] Graph store empty or no results.")

    return {
        "vector_results": vector_results,
        "graph_results":  graph_results,
        "vector_empty":   vector_empty,
        "graph_empty":    graph_empty,
        "current_agent":  "retriever",
        "agent_history":  ["retriever"],
    }


def conflict_detector_node(state: NexusState) -> dict:
    """
    Innovation 2: scan Neo4j for contradictions across documents.
    Only runs when graph has data (guaranteed by route_after_retriever).
    """
    if not CONFLICT_DETECTION:
        print("\n[Node: conflict_detector] Disabled in config, skipping.")
        return {"conflicts": [], "current_agent": "conflict_detector", "agent_history": ["conflict_detector"]}

    print("\n[Node: conflict_detector] Scanning for conflicts...")
    conflicts = find_conflicts()

    if conflicts:
        print(f"[Node: conflict_detector] Found {len(conflicts)} conflict(s):")
        for c in conflicts:
            print(f"  - {c['fact1']} vs {c['fact2']}")
    else:
        print("[Node: conflict_detector] No conflicts found.")

    return {
        "conflicts":      conflicts,
        "current_agent":  "conflict_detector",
        "agent_history":  ["conflict_detector"],
    }


def schema_reader_node(state: NexusState) -> dict:
    """
    Innovation 1: read live entity/relationship types from Neo4j.
    Only runs when graph has data.
    """
    if not AUTO_SCHEMA:
        return {"schema_types": {}, "current_agent": "schema_reader", "agent_history": ["schema_reader"]}

    print("\n[Node: schema_reader] Reading graph schema...")
    schema = get_all_schema_types()
    print(f"[Node: schema_reader] Entity types: {schema.get('entities', [])}")
    print(f"[Node: schema_reader] Relationship types: {schema.get('relationships', [])}")

    return {
        "schema_types":  schema,
        "current_agent": "schema_reader",
        "agent_history": ["schema_reader"],
    }


def combiner_node(state: NexusState) -> dict:
    """
    Merge vector + graph results. Works even if one side is empty.
    """
    print("\n[Node: combiner] Merging retrieval results...")

    vector_part = state.get("vector_results", "") or "No vector results."
    graph_part  = state.get("graph_results",  "") or "No graph results."

    combined = (
        f"=== VECTOR SEARCH RESULTS ===\n{vector_part}\n\n"
        f"=== GRAPH SEARCH RESULTS ===\n{graph_part}"
    )

    return {
        "combined_context": combined,
        "current_agent":    "combiner",
        "agent_history":    ["combiner"],
    }


def answer_node(state: NexusState) -> dict:
    """
    Synthesize all context into final answer.
    Increments retry_count — used by route_after_answer to decide re-retrieval.
    """
    print("\n[Node: answer] Generating answer...")

    result = run_answer(
        query=state["query"],
        vector_results=state.get("vector_results", ""),
        graph_results=state.get("graph_results",  ""),
        conflicts=state.get("conflicts", []),
    )

    confidence = result["confidence"]
    print(f"[Node: answer] Confidence: {confidence:.0%}")
    print(f"[Node: answer] Sources: {result['sources']}")

    return {
        "answer":        result["answer"],
        "confidence":    confidence,
        "sources":       result["sources"],
        "retry_count":   state.get("retry_count", 0) + 1,
        "current_agent": "answer",
        "agent_history": ["answer"],
    }


def error_node(state: NexusState) -> dict:
    """
    Reached when both vector + graph stores are empty.
    Sets friendly error message as answer instead of crashing.
    """
    print("\n[Node: error] No data in stores.")

    return {
        "answer":        (
            "No knowledge base data found. "
            "Run `python ingest.py` first to process your documents."
        ),
        "confidence":    0.0,
        "sources":       [],
        "error":         "stores_empty",
        "current_agent": "error",
        "agent_history": ["error"],
    }
