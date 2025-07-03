# app/ingestion.py
import os
from app.chunker import chunk_document
from app.embedder import build_vectorstore

RAW_DIR = "data/raw"
VDB_PATH = "data/vectordb"

def run():
    all_chunks = []
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(RAW_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_document(text)
        all_chunks.extend(chunks)

    build_vectorstore(all_chunks)
    print(f"Ingested {len(all_chunks)} chunks into {VDB_PATH}")
