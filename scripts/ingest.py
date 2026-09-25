#!/usr/bin/env python3
"""Script to inspect and chunk corpus documents for AVY."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from avy.config import AVYConfig
from avy.retrieval.ingest import CorpusIngestor


def main() -> None:
    config = AVYConfig.load()
    ingestor = CorpusIngestor()
    corpus_dir = Path(__file__).parent.parent / config.corpus_dir

    print(f"Reading corpus files from: {corpus_dir}")
    docs = ingestor.load_documents(corpus_dir)
    print(f"Found {len(docs)} documents:")
    for d in docs:
        print(f"  [{d.doc_id}] {d.title} ({d.source})")

    chunks = ingestor.ingest_corpus(corpus_dir)
    print(f"\nGenerated {len(chunks)} chunks across all documents.")
    if chunks:
        sample = chunks[0]
        print(f"\nSample Chunk ({sample.chunk_id}):")
        print(f"Title: {sample.title}")
        print(f"Text Preview: {sample.text[:150]}...")


if __name__ == "__main__":
    main()
