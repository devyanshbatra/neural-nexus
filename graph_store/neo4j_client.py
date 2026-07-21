from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, MAX_HOPS, TOP_K_GRAPH

_driver = None


def get_driver():
    """Singleton Neo4j driver — connect once, reuse."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def run_query(cypher: str, params: dict = None) -> list:
    """Run any Cypher query. Returns list of result records as dicts."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(record) for record in result]


# ── Write operations ───────────────────────────────────────────────────────────

def create_entity(name: str, entity_type: str, source: str, properties: dict = None):
    """Create or merge entity node in Neo4j."""
    cypher = f"""
    MERGE (e:{entity_type} {{name: $name}})
    SET e.source = $source,
        e.type = $entity_type
    """
    if properties:
        for key, value in properties.items():
            cypher += f"\nSET e.{key} = ${key}"

    params = {"name": name, "source": source, "entity_type": entity_type}
    if properties:
        params.update(properties)

    run_query(cypher, params)


def create_relationship(
    from_name: str,
    from_type: str,
    to_name: str,
    to_type: str,
    rel_type: str,
    source: str,
    confidence: float = 1.0,
):
    """Create relationship between two entities. Stores source + confidence."""
    cypher = f"""
    MERGE (a:{from_type} {{name: $from_name}})
    MERGE (b:{to_type} {{name: $to_name}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r.source = $source,
        r.confidence = $confidence
    """
    run_query(cypher, {
        "from_name": from_name,
        "to_name": to_name,
        "source": source,
        "confidence": confidence,
    })


# ── Read operations ────────────────────────────────────────────────────────────

def search_graph(query_entity: str) -> str:
    """
    Multi-hop graph traversal. Finds all nodes connected to query entity
    within MAX_HOPS steps. Returns formatted string for LLM context.
    """
    cypher = f"""
    MATCH (start)
    WHERE toLower(start.name) CONTAINS toLower($query)
    MATCH path = (start)-[*1..{MAX_HOPS}]-(related)
    RETURN start.name AS source,
           [r in relationships(path) | type(r)] AS relationship_chain,
           related.name AS target,
           related.source AS document
    LIMIT {TOP_K_GRAPH}
    """
    results = run_query(cypher, {"query": query_entity})

    if not results:
        return "No graph relationships found."

    lines = ["Graph Relationships Found:"]
    for r in results:
        chain = " → ".join(r.get("relationship_chain", []))
        lines.append(
            f"  {r.get('source')} --[{chain}]--> {r.get('target')} "
            f"(from: {r.get('document', 'unknown')})"
        )
    return "\n".join(lines)


def get_all_schema_types() -> dict:
    """Return all entity types and relationship types currently in the graph."""
    entity_cypher = "MATCH (n) RETURN DISTINCT labels(n) AS types"
    rel_cypher    = "MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel_type"

    entity_results = run_query(entity_cypher)
    rel_results    = run_query(rel_cypher)

    entity_types = list({
        t for row in entity_results for t in row.get("types", [])
    })
    rel_types = [row["rel_type"] for row in rel_results if row.get("rel_type")]

    return {"entities": entity_types, "relationships": rel_types}


def find_conflicts() -> list:
    """
    Conflict detection — find entities with same name but different
    relationships to contradicting targets across different source documents.
    """
    cypher = """
    MATCH (a)-[r1]->(b)
    MATCH (a)-[r2]->(c)
    WHERE type(r1) = type(r2)
      AND b.name <> c.name
      AND r1.source <> r2.source
    RETURN a.name AS entity,
           type(r1) AS relationship,
           b.name AS fact1,
           r1.source AS source1,
           c.name AS fact2,
           r2.source AS source2
    LIMIT 20
    """
    results = run_query(cypher)

    conflicts = []
    for r in results:
        conflicts.append({
            "entity":       r.get("entity"),
            "relationship": r.get("relationship"),
            "fact1":        f"{r.get('entity')} {r.get('relationship')} {r.get('fact1')}",
            "fact2":        f"{r.get('entity')} {r.get('relationship')} {r.get('fact2')}",
            "source1":      r.get("source1"),
            "source2":      r.get("source2"),
        })
    return conflicts


def clear_graph():
    """Wipe all nodes and relationships. Use before re-ingesting."""
    run_query("MATCH (n) DETACH DELETE n")
    print("Graph cleared.")
