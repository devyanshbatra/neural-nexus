from langchain_core.prompts import ChatPromptTemplate

# ── Extractor prompt ───────────────────────────────────────────────────────────
# Reads raw document text, invents or reuses entity/relationship types,
# returns structured JSON for graph_store/builder.py

extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledge graph extraction expert.

Extract entities and relationships from the document text below.

EXISTING SCHEMA (reuse these types if they fit — don't invent duplicates):
{existing_schema}

Rules:
- Entity types: UPPERCASE_SNAKE_CASE (e.g. PERSON, CI_TOOL, SERVER, DEPARTMENT)
- Relationship types: UPPERCASE_SNAKE_CASE verbs (e.g. WORKS_IN, DEPLOYS_TO, MANAGES)
- If no existing schema yet, invent types that fit the domain
- Confidence: 0.0-1.0, how certain you are this relationship is stated in the text
- Only extract facts explicitly stated — no inference
- names should be exact as in the document

Return ONLY valid JSON, no explanation:
{{
    "entities": [
        {{"name": "...", "type": "...", "properties": {{}}}}
    ],
    "relationships": [
        {{
            "from_name": "...",
            "from_type": "...",
            "to_name": "...",
            "to_type": "...",
            "rel_type": "...",
            "confidence": 0.9
        }}
    ]
}}"""),
    ("human", "Document source: {source}\n\nDocument text:\n{text}"),
])


# ── Retriever prompt ───────────────────────────────────────────────────────────
# Decides which entities to query in graph based on user question

retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval planning expert.

Given a user question, identify the key entities to search for in a knowledge graph.

Return ONLY valid JSON:
{{
    "entities_to_search": ["entity1", "entity2"],
    "search_reasoning": "brief explanation of why these entities"
}}

Keep entities short and exact — they must match names in the graph."""),
    ("human", "Question: {query}"),
])


# ── Answer prompt ──────────────────────────────────────────────────────────────
# Synthesizes vector + graph results into final answer

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a company knowledge base assistant.

Answer the user's question using ONLY the provided context below.
Cite sources when possible. If context is insufficient, say so clearly.

VECTOR SEARCH RESULTS (semantic chunks from documents):
{vector_results}

GRAPH SEARCH RESULTS (entity relationships from knowledge graph):
{graph_results}

{conflict_warning}

Rules:
- Synthesize both vector and graph context
- Prefer graph results for factual relationships (who manages whom, what deploys where)
- Prefer vector results for explanations and detailed descriptions
- State confidence as: HIGH / MEDIUM / LOW
- If conflicts detected, flag them explicitly in your answer"""),
    ("human", "Question: {query}"),
])


# ── Conflict warning snippet injected into answer prompt when conflicts exist ──

CONFLICT_WARNING_TEMPLATE = """
⚠️ CONFLICTS DETECTED — contradictions found across source documents:
{conflicts}
Flag these discrepancies in your answer and note which sources disagree.
"""

NO_CONFLICT_TEXT = ""


def build_conflict_warning(conflicts: list) -> str:
    if not conflicts:
        return NO_CONFLICT_TEXT

    lines = []
    for c in conflicts:
        lines.append(
            f"  - {c.get('fact1')} (from: {c.get('source1')}) "
            f"vs {c.get('fact2')} (from: {c.get('source2')})"
        )
    return CONFLICT_WARNING_TEMPLATE.format(conflicts="\n".join(lines))
