"""Simple RAG (Retrieval-Augmented Generation) pipeline.

    Documents --> split into chunks --> OpenAI embeddings --> in-memory vector index
    Question  --> embedding --> top-k most similar chunks (cosine similarity)
              --> LLM answers using ONLY those chunks

No external vector database is needed: the index is a NumPy matrix kept in memory,
which is enough for demos and small document sets. For production, swap
`VectorIndex` for a vector DB (FAISS, Chroma, pgvector, Pinecone ...).
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer is not in the context, say \"I could not find this in the documents.\" "
    "Do not use outside knowledge. Mention the source file name for the facts you use."
)


@dataclass
class Chunk:
    text: str
    source: str  # file name
    chunk_id: int


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity, 0..1 (higher = more similar)


# ---------------- 1. Load documents ----------------
def read_file(name: str, data: bytes) -> str:
    """Extract plain text from PDF, DOCX, TXT or MD bytes."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        import docx

        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)
    return data.decode("utf-8", errors="ignore")  # .txt, .md and anything else text-like


def load_sample_docs() -> list[tuple[str, str]]:
    """Return (file name, text) for every file in sample_docs/."""
    docs = []
    for path in sorted(SAMPLE_DOCS_DIR.glob("*")):
        if path.is_file():
            docs.append((path.name, read_file(path.name, path.read_bytes())))
    return docs


# ---------------- 2. Chunk ----------------
def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split text into overlapping chunks of ~chunk_size characters.

    Splits on paragraph / sentence boundaries where possible so chunks stay readable.
    Overlap keeps facts that sit on a chunk border from being cut in half.
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+|\n\n", text)
    chunks, current = [], ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(current) + len(piece) + 1 <= chunk_size:
            current = f"{current} {piece}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {piece}".strip()
    if current:
        chunks.append(current)

    return [Chunk(text=c, source=source, chunk_id=i) for i, c in enumerate(chunks)]


# ---------------- 3. Embed + index ----------------
def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed texts with OpenAI and return L2-normalised vectors."""
    vectors = []
    for start in range(0, len(texts), 100):  # batch to stay within API limits
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[start:start + 100])
        vectors.extend(item.embedding for item in response.data)
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class VectorIndex:
    """Tiny in-memory vector store: cosine similarity over normalised embeddings."""

    def __init__(self, client: OpenAI, chunks: list[Chunk]):
        self.client = client
        self.chunks = chunks
        self.vectors = embed(client, [c.text for c in chunks]) if chunks else np.zeros((0, 1))

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        query_vector = embed(self.client, [query])[0]
        scores = self.vectors @ query_vector  # cosine similarity (vectors are normalised)
        best = np.argsort(scores)[::-1][:top_k]
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in best]


def build_index(client: OpenAI, docs: list[tuple[str, str]],
                chunk_size: int = 800, overlap: int = 150) -> VectorIndex:
    chunks = []
    for name, text in docs:
        chunks.extend(chunk_text(text, name, chunk_size, overlap))
    return VectorIndex(client, chunks)


# ---------------- 4. Generate ----------------
def format_context(retrieved: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[Source: {r.chunk.source}, chunk {r.chunk.chunk_id}]\n{r.chunk.text}" for r in retrieved
    )


def answer_with_rag(client: OpenAI, model: str, question: str,
                    retrieved: list[RetrievedChunk], system_prompt: str = RAG_SYSTEM_PROMPT) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{format_context(retrieved)}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()