from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, CHROMA_PATH, TOP_K_VECTOR

_vectorstore = None


def get_vectorstore() -> Chroma:
    """Singleton Chroma instance. Connect once, reuse."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        _vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
        )
    return _vectorstore


def search_vector(query: str) -> str:
    """
    Semantic similarity search. Returns formatted string of top-K chunks
    with source metadata — ready to pass directly to LLM as context.
    """
    store = get_vectorstore()
    results = store.similarity_search_with_score(query, k=TOP_K_VECTOR)

    if not results:
        return "No relevant documents found."

    lines = ["Vector Search Results:"]
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_str = f" p.{page}" if page else ""
        lines.append(
            f"\n[Source: {source}{page_str} | Score: {score:.3f}]\n{doc.page_content}"
        )

    return "\n".join(lines)


def get_all_sources() -> list:
    """Return unique source document names stored in Chroma."""
    store = get_vectorstore()
    # Chroma stores metadata per document — collect unique sources
    collection = store._collection
    results = collection.get(include=["metadatas"])
    sources = list({
        m.get("source", "unknown")
        for m in results.get("metadatas", [])
        if m
    })
    return sources
