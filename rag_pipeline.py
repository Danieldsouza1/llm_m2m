"""
rag_pipeline.py
---------------
Improved RAG pipeline with section-aware chunking.

Chunking strategy:
  1. Split document on SECTION headers
  2. Within each section, chunk by paragraph
  3. Large paragraphs split into sentence windows of 4
  4. Every chunk prefixed with its section title for better retrieval
"""

import re
import chromadb
from chromadb.utils import embedding_functions


class RAG:
    def __init__(self, collection_name: str = "energy_manual"):
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url="http://localhost:11434/api/embeddings"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )
        self._ingested = False

    # ──────────────────────────────────────────
    # SECTION-AWARE CHUNKING
    # ──────────────────────────────────────────

    def _split_sections(self, text):
        text = text.replace("\r\n", "\n")
        section_pattern = re.compile(
            r"(?:^-{10,}\n)?(SECTION\s+\d+[:\s][^\n]+)\n(?:-{10,}\n)?",
            re.MULTILINE,
        )
        sections = []
        matches = list(section_pattern.finditer(text))

        if not matches:
            return [("DOCUMENT", text.strip())]

        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("PREAMBLE", preamble))

        for i, match in enumerate(matches):
            title = match.group(1).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            sections.append((title, body))

        return sections

    def _chunk_section(self, section_title, body, max_sentences=4):
        chunks = []
        paragraphs = re.split(r"\n{2,}", body)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            sentences = re.split(r"(?<=[.!?])\s+", para)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= max_sentences:
                chunks.append(f"[{section_title}]\n{para}")
            else:
                for i in range(0, len(sentences), max_sentences - 1):
                    window = sentences[i: i + max_sentences]
                    chunks.append(f"[{section_title}]\n{' '.join(window)}")

        return chunks

    def chunk_text(self, text):
        sections = self._split_sections(text)
        all_chunks = []
        for title, body in sections:
            all_chunks.extend(self._chunk_section(title, body))
        return all_chunks

    # ──────────────────────────────────────────
    # INGEST
    # ──────────────────────────────────────────

    def ingest(self, file_path):
        if self._ingested:
            return

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = self.chunk_text(text)
        print(f"📚 Total chunks created: {len(chunks)}  (section-aware)")

        BATCH_SIZE = 10
        ids = [str(i) for i in range(len(chunks))]

        for start in range(0, len(chunks), BATCH_SIZE):
            batch_docs = chunks[start: start + BATCH_SIZE]
            batch_ids  = ids[start: start + BATCH_SIZE]
            self.collection.upsert(documents=batch_docs, ids=batch_ids)
            print(f"  Embedded chunks {start + 1}–{min(start + BATCH_SIZE, len(chunks))} of {len(chunks)}")

        print("✅ Manual ingested into vector DB")
        self._ingested = True

    # ──────────────────────────────────────────
    # QUERY
    # ──────────────────────────────────────────

    def query(self, query_text, n_results=6):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count()),
        )
        retrieved = results["documents"][0]
        context = "\n\n".join(retrieved)

        print(f"\n🔍 Retrieved {len(retrieved)} chunks:")
        for r in retrieved:
            preview = r.replace("\n", " ")[:100]
            print(f"  - {preview}")

        return context


if __name__ == "__main__":
    import os
    BASE = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE, "data", "energy_manual.txt")
    rag = RAG()
    rag.ingest(path)
    print("\n=== Query: line overload ===")
    ctx = rag.query("ERROR: Line overload in sector B")
    print(ctx[:500])
