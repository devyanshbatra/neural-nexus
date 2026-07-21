import json
from langchain_ollama import ChatOllama
from prompts.templates import retriever_prompt
from vector_store.store import search_vector
from graph_store.neo4j_client import search_graph
from config import LLM_MODEL, TEMPERATURE


def run_retriever(query: str, retry_count: int = 0) -> dict:
    """
    Hybrid retrieval: vector search + graph traversal.
    retry_count > 0 means previous answer had low confidence — broaden search:
      - ask LLM for more entities (up to 5 instead of default 2-3)
      - use broader/rephrased query for vector search
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
    chain = retriever_prompt | llm

    # On retry, tell LLM to find more entities + be broader
    query_for_planning = query
    if retry_count > 0:
        query_for_planning = (
            f"{query}\n\n"
            f"[Retry {retry_count}: previous answer had low confidence. "
            f"Return MORE entities (up to 5). Think broadly — synonyms, related concepts.]"
        )

    # Step 1: Ask LLM which entities to search in graph
    response = chain.invoke({"query": query_for_planning})
    raw = response.content.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        plan = json.loads(raw)
        entities_to_search = plan.get("entities_to_search", [])
    except json.JSONDecodeError:
        # Fallback: use first 3 words on retry (broader than default 2)
        n = 3 if retry_count > 0 else 2
        entities_to_search = query.split()[:n]
        print(f"[Retriever] JSON parse failed, fallback entities: {entities_to_search}")

    # Step 2: Vector search
    print(f"[Retriever] Vector search for: '{query}'")
    vector_results = search_vector(query)

    # Step 3: Graph search for each identified entity
    print(f"[Retriever] Graph search for entities: {entities_to_search}")
    graph_parts = []
    for entity in entities_to_search:
        result = search_graph(entity)
        if result != "No graph relationships found.":
            graph_parts.append(result)

    graph_results = "\n\n".join(graph_parts) if graph_parts else "No graph relationships found."

    return {
        "vector_results":    vector_results,
        "graph_results":     graph_results,
        "entities_searched": entities_to_search,
    }
