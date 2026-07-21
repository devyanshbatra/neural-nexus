from langgraph.graph import StateGraph, END
from state.schema import NexusState
from graph.nodes import (
    retriever_node,
    conflict_detector_node,
    schema_reader_node,
    combiner_node,
    answer_node,
    error_node,
)
from graph.edges import (
    route_after_retriever,
    route_after_conflict,
    route_after_answer,
)


def build_graph():
    """
    NeuralNexus query pipeline with conditional routing:

    retriever ──┬──(both empty)──────────────────────────→ error_node → END
                ├──(graph empty)────────────────────────→ combiner ──→ answer ──┬──(low conf, retry)──→ retriever
                └──(has graph)──→ conflict_detector → schema_reader ──→ combiner ──→ answer ──┴──(confident)──→ END
    """
    workflow = StateGraph(NexusState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    workflow.add_node("retriever",         retriever_node)
    workflow.add_node("conflict_detector", conflict_detector_node)
    workflow.add_node("schema_reader",     schema_reader_node)
    workflow.add_node("combiner",          combiner_node)
    workflow.add_node("answer",            answer_node)
    workflow.add_node("error_node",        error_node)

    # ── Entry ──────────────────────────────────────────────────────────────────
    workflow.set_entry_point("retriever")

    # ── Conditional edge 1 + 2: after retriever ────────────────────────────────
    # Checks vector_empty + graph_empty flags
    workflow.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "conflict_detector": "conflict_detector",  # graph has data → full pipeline
            "combiner":          "combiner",            # graph empty → skip conflict/schema
            "error_node":        "error_node",          # both empty → error
        }
    )

    # ── Conditional edge 3: after conflict detector ────────────────────────────
    # Currently always → schema_reader, but conditional for future branching
    workflow.add_conditional_edges(
        "conflict_detector",
        route_after_conflict,
        {
            "schema_reader": "schema_reader",
        }
    )

    # ── Fixed edges: schema_reader → combiner ─────────────────────────────────
    workflow.add_edge("schema_reader", "combiner")
    workflow.add_edge("combiner",      "answer")

    # ── Conditional edge 4: after answer — confidence re-retrieval loop ────────
    workflow.add_conditional_edges(
        "answer",
        route_after_answer,
        {
            "retriever": "retriever",  # low confidence → retry with broader search
            "__end__":   END,          # confident enough → done
        }
    )

    # ── Error node always exits ────────────────────────────────────────────────
    workflow.add_edge("error_node", END)

    return workflow.compile()


# Singleton — imported by main.py
nexus_graph = build_graph()
