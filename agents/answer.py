import re
from langchain_ollama import ChatOllama
from prompts.templates import answer_prompt, build_conflict_warning
from config import LLM_MODEL, ANSWER_TEMPERATURE


def run_answer(
    query: str,
    vector_results: str,
    graph_results: str,
    conflicts: list,
) -> dict:
    """
    Synthesize vector + graph results into final answer.
    Returns dict with answer, confidence, sources.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=ANSWER_TEMPERATURE)
    chain = answer_prompt | llm

    conflict_warning = build_conflict_warning(conflicts)

    response = chain.invoke({
        "query": query,
        "vector_results": vector_results,
        "graph_results": graph_results,
        "conflict_warning": conflict_warning,
    })

    answer_text = response.content.strip()

    # Extract confidence level from answer text
    confidence = _parse_confidence(answer_text)

    # Extract source references mentioned in the answer
    sources = _extract_sources(vector_results, graph_results)

    return {
        "answer": answer_text,
        "confidence": confidence,
        "sources": sources,
    }


def _parse_confidence(text: str) -> float:
    """Parse HIGH/MEDIUM/LOW confidence from answer text → float."""
    text_upper = text.upper()
    if "HIGH" in text_upper:
        return 0.9
    elif "MEDIUM" in text_upper:
        return 0.6
    elif "LOW" in text_upper:
        return 0.3
    return 0.5  # default if not stated


def _extract_sources(vector_results: str, graph_results: str) -> list:
    """Pull source filenames from retrieval results."""
    sources = set()

    # Sources from vector results: "[Source: filename.pdf p.3 | Score: ...]"
    for match in re.finditer(r"\[Source:\s*([^\|]+)\|", vector_results):
        sources.add(match.group(1).strip())

    # Sources from graph results: "(from: filename.pdf)"
    for match in re.finditer(r"\(from:\s*([^)]+)\)", graph_results):
        sources.add(match.group(1).strip())

    return sorted(sources)
