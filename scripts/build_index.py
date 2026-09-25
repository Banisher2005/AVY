#!/usr/bin/env python3
"""Script to build and serialize the FAISS vector index from corpus documents."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from avy.config import AVYConfig
from avy.retrieval.embeddings import get_embedding_engine
from avy.retrieval.ingest import CorpusIngestor
from avy.retrieval.vectorstore import FAISSVectorStore


def main() -> None:
    config = AVYConfig.load()
    root_dir = Path(__file__).parent.parent
    corpus_dir = root_dir / config.corpus_dir
    index_path = root_dir / config.index_path
    meta_path = root_dir / config.metadata_path

    print(f"1. Loading corpus from: {corpus_dir}")
    ingestor = CorpusIngestor()
    chunks = ingestor.ingest_corpus(corpus_dir)
    if not chunks:
        print("Error: No chunks extracted from corpus.")
        sys.exit(1)
    print(f"Extracted {len(chunks)} chunks.")

    print(f"2. Generating embeddings using provider: {config.embedding_provider}...")
    embeddings = get_embedding_engine(config)
    texts = [c.text for c in chunks]
    emb_matrix = embeddings.embed_documents(texts)
    print(f"Embedding matrix shape: {emb_matrix.shape}")

    print(f"3. Building FAISS vector store (dimension={embeddings.dimension})...")
    vstore = FAISSVectorStore(dimension=embeddings.dimension)
    vstore.add_chunks(chunks, emb_matrix)

    print(f"4. Saving index to: {index_path}")
    print(f"   Saving metadata to: {meta_path}")
    vstore.save(index_path, meta_path)
    print("✅ Index building completed successfully!")


if __name__ == "__main__":
    main()
