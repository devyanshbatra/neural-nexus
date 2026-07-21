import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR


def load_documents(doc_dir: str = DOCUMENTS_DIR) -> list:
    """Load all .pdf and .txt files from documents directory."""
    docs = []
    if not os.path.exists(doc_dir):
        print(f"[VectorSetup] Documents dir not found: {doc_dir}")
        return docs

    for filename in os.listdir(doc_dir):
        filepath = os.path.join(doc_dir, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                continue

            loaded = loader.load()
            # Tag each chunk with source filename
            for doc in loaded:
                doc.metadata["source"] = filename
            docs.extend(loaded)
            print(f"[VectorSetup] Loaded: {filename} ({len(loaded)} pages/sections)")

        except Exception as e:
            print(f"[VectorSetup] Failed to load {filename}: {e}")

    return docs


def chunk_documents(docs: list) -> list:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[VectorSetup] {len(docs)} docs → {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    """Embed chunks and persist to Chroma."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"[VectorSetup] Chroma built at: {CHROMA_PATH}")
    return store


def setup_vector_pipeline(doc_dir: str = DOCUMENTS_DIR) -> int:
    """
    Full pipeline: load → chunk → embed → persist.
    Returns number of chunks written.
    """
    docs = load_documents(doc_dir)
    if not docs:
        print("[VectorSetup] No documents found. Add .pdf or .txt files to documents/")
        return 0

    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    return len(chunks)
