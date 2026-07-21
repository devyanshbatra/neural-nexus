"""
ingest.py — run once (or re-run) to process documents into:
  1. Chroma vector store (semantic embeddings)
  2. Neo4j knowledge graph (entity/relationship extraction)

Usage:
    python ingest.py
    python ingest.py --clear   # wipe both stores before ingesting
"""

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

from vector_store.setup import load_documents, chunk_documents, build_vectorstore
from agents.extractor import run_extractor
from graph_store.neo4j_client import clear_graph
from config import DOCUMENTS_DIR


def ingest(clear_first: bool = False):
    print("=" * 60)
    print("NeuralNexus Ingestion Pipeline")
    print("=" * 60)

    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"\nCreated documents/ folder.")
        print(f"Add your .pdf or .txt files to: {os.path.abspath(DOCUMENTS_DIR)}")
        print("Then run ingest.py again.")
        return

    # ── Optional: clear existing stores ───────────────────────────────────────
    if clear_first:
        print("\n[1/4] Clearing existing stores...")
        clear_graph()
        import shutil
        from config import CHROMA_PATH
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print(f"Chroma cleared: {CHROMA_PATH}")
    else:
        print("\n[1/4] Skipping clear (use --clear to wipe stores first)")

    # ── Load raw documents ─────────────────────────────────────────────────────
    print("\n[2/4] Loading documents...")
    docs = load_documents(DOCUMENTS_DIR)
    if not docs:
        print("No documents found. Add .pdf or .txt files to documents/ folder.")
        return

    # ── Build vector store ─────────────────────────────────────────────────────
    print("\n[3/4] Building vector store...")
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)

    # ── Build knowledge graph ──────────────────────────────────────────────────
    print("\n[4/4] Extracting knowledge graph...")

    # Group chunks by source document — pass full doc text to extractor, not chunks
    # (chunks are for vector search; graph extraction needs full context)
    docs_by_source = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc.page_content)

    total_entities = 0
    total_rels = 0

    for source, texts in docs_by_source.items():
        # Join all pages/sections of this document
        full_text = "\n\n".join(texts)

        # Truncate very long docs to avoid LLM context limits
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "\n[... truncated for extraction ...]"

        print(f"\n  Extracting from: {source}")
        extraction = run_extractor(text=full_text, source=source)

        total_entities += len(extraction.get("entities", []))
        total_rels += len(extraction.get("relationships", []))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print(f"  Documents processed : {len(docs_by_source)}")
    print(f"  Vector chunks       : {len(chunks)}")
    print(f"  Entities extracted  : {total_entities}")
    print(f"  Relationships built : {total_rels}")
    print("=" * 60)
    print("\nRun: python main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralNexus document ingestion")
    parser.add_argument("--clear", action="store_true", help="Clear stores before ingesting")
    args = parser.parse_args()
    ingest(clear_first=args.clear)
