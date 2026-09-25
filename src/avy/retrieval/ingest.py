"""Corpus document ingestion and chunking for AVY."""

import re
from pathlib import Path

from avy.retrieval.models import Chunk, Document


class CorpusIngestor:
    """Loads documents from a corpus directory and splits them into clean chunks."""

    def __init__(self, target_chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, corpus_dir: str | Path) -> list[Document]:
        """Read all markdown and text files from corpus folder."""
        cpath = Path(corpus_dir)
        if not cpath.is_dir():
            return []

        docs: list[Document] = []
        for file_path in sorted(cpath.glob("*.*")):
            if file_path.suffix.lower() not in (".md", ".txt"):
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            # Extract title from markdown '# Title' or fallback to filename
            title = file_path.stem.replace("_", " ").title()
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("# "):
                    title = stripped[2:].strip()
                    break

            doc_id = file_path.stem
            docs.append(
                Document(
                    doc_id=doc_id,
                    title=title,
                    source=file_path.name,
                    filepath=str(file_path),
                    content=content,
                    metadata={"filesize": file_path.stat().st_size},
                )
            )
        return docs

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a document into paragraphs and overlapping passages."""
        # Split document by double newlines or section headers
        raw_sections = re.split(r"\n\s*\n|\n(?=##?\s)", doc.content)
        sections = [s.strip() for s in raw_sections if s.strip()]

        chunks: list[Chunk] = []
        chunk_idx = 0

        for section in sections:
            # If section is small enough, treat as a single chunk
            if len(section) <= self.target_chunk_size + 100:
                cid = f"{doc.doc_id}#c{chunk_idx}"
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_id=doc.doc_id,
                        source=doc.source,
                        title=doc.title,
                        chunk_index=chunk_idx,
                        text=section,
                        metadata={"section_length": len(section)},
                    )
                )
                chunk_idx += 1
            else:
                # Sub-chunk longer sections with sliding window
                words = section.split()
                start = 0
                step = 80  # words
                while start < len(words):
                    sub_words = words[start : start + 120]
                    sub_text = " ".join(sub_words)
                    cid = f"{doc.doc_id}#c{chunk_idx}"
                    chunks.append(
                        Chunk(
                            chunk_id=cid,
                            doc_id=doc.doc_id,
                            source=doc.source,
                            title=doc.title,
                            chunk_index=chunk_idx,
                            text=sub_text,
                            metadata={"section_length": len(sub_text)},
                        )
                    )
                    chunk_idx += 1
                    start += step

        return chunks

    def ingest_corpus(self, corpus_dir: str | Path) -> list[Chunk]:
        """Load all documents from directory and return combined chunks."""
        docs = self.load_documents(corpus_dir)
        all_chunks: list[Chunk] = []
        for d in docs:
            all_chunks.extend(self.chunk_document(d))
        return all_chunks
