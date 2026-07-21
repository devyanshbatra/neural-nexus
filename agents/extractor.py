import json
from langchain_ollama import ChatOllama
from prompts.templates import extractor_prompt
from graph_store.builder import get_current_schema, build_graph_from_extraction
from config import LLM_MODEL, TEMPERATURE


def run_extractor(text: str, source: str) -> dict:
    """
    Extract entities + relationships from document text.
    Writes results to Neo4j. Returns extraction summary.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
    chain = extractor_prompt | llm

    # Get existing schema so LLM reuses types (Innovation 1: auto-schema consistency)
    existing_schema = get_current_schema()
    schema_str = json.dumps(existing_schema, indent=2) if existing_schema else "None yet — invent appropriate types."

    response = chain.invoke({
        "text": text,
        "source": source,
        "existing_schema": schema_str,
    })

    # Parse JSON from LLM response
    raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        extraction = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[Extractor] JSON parse failed for {source}: {e}")
        return {"entities": [], "relationships": [], "error": str(e)}

    # Write to Neo4j
    summary = build_graph_from_extraction(extraction, source)
    extraction["summary"] = summary

    print(f"[Extractor] {source}: "
          f"{len(extraction.get('entities', []))} entities, "
          f"{len(extraction.get('relationships', []))} relationships extracted.")

    return extraction
