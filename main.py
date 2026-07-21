"""
main.py — NeuralNexus interactive Q&A over company knowledge base.

Prerequisites:
  1. Neo4j Desktop running (bolt://localhost:7687)
  2. Ollama running with llama3.2 + nomic-embed-text
  3. Documents added to documents/ folder
  4. python ingest.py  (run once to build vector store + knowledge graph)

Then: python main.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from graph.graph import nexus_graph
from graph_store.neo4j_client import close_driver


def print_header():
    print("\n" + "=" * 60)
    print("  NeuralNexus — Company Knowledge Base")
    print("  RAG + Knowledge Graph powered by Ollama + Neo4j")
    print("=" * 60)
    print("Commands: 'quit' to exit | 'sources' to list docs\n")


def print_answer(result: dict):
    """Display final answer with metadata."""
    print("\n" + "─" * 60)
    print("ANSWER:")
    print(result.get("answer", "No answer generated."))

    confidence = result.get("confidence", 0.5)
    conf_label = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.5 else "LOW"
    print(f"\nConfidence: {conf_label} ({confidence:.0%})")

    sources = result.get("sources", [])
    if sources:
        print(f"Sources: {', '.join(sources)}")

    conflicts = result.get("conflicts", [])
    if conflicts:
        print(f"\n⚠️  Conflicts detected ({len(conflicts)}):")
        for c in conflicts[:3]:  # show max 3
            print(f"  - {c.get('fact1')} vs {c.get('fact2')}")

    print("─" * 60)


def list_sources():
    """Show all documents in the vector store."""
    try:
        from vector_store.store import get_all_sources
        sources = get_all_sources()
        if sources:
            print("\nDocuments in knowledge base:")
            for s in sources:
                print(f"  • {s}")
        else:
            print("\nNo documents ingested yet. Run: python ingest.py")
    except Exception as e:
        print(f"\nCould not read sources: {e}")


def main():
    print_header()

    # Check that Chroma exists (ingestion ran)
    from config import CHROMA_PATH
    if not os.path.exists(CHROMA_PATH):
        print("Vector store not found. Run ingestion first:")
        print("  python ingest.py")
        return

    print("Knowledge base ready. Ask anything about your documents.")
    print("(First query may be slow — models loading)\n")

    try:
        while True:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            if query.lower() == "sources":
                list_sources()
                continue

            print("\n[NeuralNexus] Processing...")

            # Build initial state
            initial_state = {
                "query":           query,
                "vector_results":  "",
                "graph_results":   "",
                "combined_context": "",
                "schema_types":    {},
                "conflicts":       [],
                "answer":          "",
                "sources":         [],
                "confidence":      0.0,
                "current_agent":   "",
                "agent_history":   [],
                "error":           "",
                "vector_empty":    False,
                "graph_empty":     False,
                "retry_count":     0,
            }

            try:
                result = nexus_graph.invoke(initial_state)
                print_answer(result)

                # Show agent execution path
                history = result.get("agent_history", [])
                if history:
                    print(f"Pipeline: {' → '.join(history)}")

            except Exception as e:
                print(f"\n[Error] {e}")
                print("Check: Neo4j running? Ollama running? Documents ingested?")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    finally:
        close_driver()
        print("Neo4j connection closed.")


if __name__ == "__main__":
    main()
