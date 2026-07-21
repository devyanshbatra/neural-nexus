from graph_store.neo4j_client import create_entity, create_relationship, get_all_schema_types
from config import AUTO_SCHEMA


def build_graph_from_extraction(extraction: dict, source_doc: str) -> dict:
    """
    Takes structured extraction from extractor agent and writes to Neo4j.
    Returns summary of what was written.

    Expected extraction format:
    {
        "entities": [
            {"name": "Jenkins", "type": "CI_TOOL", "properties": {...}},
            ...
        ],
        "relationships": [
            {
                "from_name": "Jenkins",
                "from_type": "CI_TOOL",
                "to_name": "AWS",
                "to_type": "CLOUD_PLATFORM",
                "rel_type": "DEPLOYS_TO",
                "confidence": 0.9
            },
            ...
        ]
    }
    """
    entities_written = 0
    relationships_written = 0
    errors = []

    # ── Write entities ─────────────────────────────────────────────────────────
    for entity in extraction.get("entities", []):
        try:
            create_entity(
                name=entity["name"],
                entity_type=entity["type"],
                source=source_doc,
                properties=entity.get("properties", {}),
            )
            entities_written += 1
        except Exception as e:
            errors.append(f"Entity '{entity.get('name')}': {e}")

    # ── Write relationships ────────────────────────────────────────────────────
    for rel in extraction.get("relationships", []):
        try:
            create_relationship(
                from_name=rel["from_name"],
                from_type=rel["from_type"],
                to_name=rel["to_name"],
                to_type=rel["to_type"],
                rel_type=rel["rel_type"],
                source=source_doc,
                confidence=rel.get("confidence", 1.0),
            )
            relationships_written += 1
        except Exception as e:
            errors.append(
                f"Relationship '{rel.get('from_name')} → {rel.get('to_name')}': {e}"
            )

    summary = {
        "source": source_doc,
        "entities_written": entities_written,
        "relationships_written": relationships_written,
        "errors": errors,
    }

    print(f"[GraphBuilder] {source_doc}: {entities_written} entities, "
          f"{relationships_written} relationships written.")
    if errors:
        print(f"[GraphBuilder] Errors: {errors}")

    return summary


def get_current_schema() -> dict:
    """
    Read live schema from Neo4j. Used by extractor agent to stay consistent
    with existing entity/relationship types (AUTO_SCHEMA innovation).
    Returns empty dict if graph is empty or AUTO_SCHEMA disabled.
    """
    if not AUTO_SCHEMA:
        return {}

    try:
        return get_all_schema_types()
    except Exception as e:
        print(f"[GraphBuilder] Schema read failed: {e}")
        return {}
