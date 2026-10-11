"""
rag_pipeline.py
----------------
Retrieval-Augmented Generation (RAG) pipeline for lab-report context.

Responsibilities:
  - Load a plain-text medical knowledge base
  - Split it into overlapping chunks
  - Embed the chunks and index them in a local FAISS vector store
  - Retrieve the most relevant chunks for abnormal lab parameters

Embedding providers (see ``EMBEDDING_PROVIDER`` in ``.env``):
  - ``fastembed`` (default): real semantic embeddings via a small ONNX model.
  - ``hashing``: lightweight offline bag-of-words fallback, handy for CI.

Usage:
    from app.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    context = rag.retrieve_for_abnormal_params(report["parameters"])
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------
class HashingEmbeddings(Embeddings):
    """
    Deterministic bag-of-words embeddings using the hashing trick.

    Not semantic, but it matches on shared words (e.g. "LDL", "cholesterol"),
    needs no model download, and is fully reproducible, which makes it a good
    fit for tests and offline runs.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FastEmbedEmbeddings(Embeddings):
    """Thin LangChain wrapper around ``fastembed.TextEmbedding``."""

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError(
                "fastembed is not installed. Run `pip install -r requirements.txt` "
                "or set EMBEDDING_PROVIDER=hashing."
            ) from exc

        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception as exc:  # network / cache problems
            raise RuntimeError(
                f"Could not load embedding model '{model_name}': {exc}. "
                "Check your internet connection for the first-run download, "
                "or set EMBEDDING_PROVIDER=hashing for an offline fallback."
            ) from exc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.embed([text]))).tolist()


def build_embeddings(provider: str | None = None, model: str | None = None) -> Embeddings:
    """Create the embeddings object selected by configuration."""
    settings = get_settings()
    provider = (provider or settings.embedding_provider).lower()
    model = model or settings.embedding_model

    if provider == "hashing":
        logger.info("Using hashing embeddings (offline fallback).")
        return HashingEmbeddings()
    if provider == "fastembed":
        logger.info("Using fastembed embeddings: %s", model)
        return FastEmbedEmbeddings(model)
    raise RuntimeError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Use 'fastembed' or 'hashing'."
    )


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------
class RAGPipeline:
    """
    Encapsulates the RAG pipeline:
      1. Load the knowledge base text
      2. Split into chunks
      3. Embed and store in FAISS
      4. Retrieve relevant context for a query or a set of abnormal parameters

    Parameters
    ----------
    knowledge_file : str | Path | None
        Path to the plain-text medical knowledge file (default: bundled file).
    chunk_size, chunk_overlap : int
        Chunking parameters in characters.
    top_k : int
        Number of chunks returned by ``retrieve``.
    embeddings : Embeddings | None
        Inject a custom embeddings object (used by tests). Defaults to the
        provider selected by configuration.
    """

    def __init__(
        self,
        knowledge_file: str | Path | None = None,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        top_k: Optional[int] = None,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        settings = get_settings()
        self.top_k = top_k if top_k is not None else settings.rag_top_k
        self._knowledge_file = Path(knowledge_file or settings.knowledge_file)
        self._embeddings = embeddings or build_embeddings()
        self._vector_store: FAISS = self._build_index(chunk_size, chunk_overlap)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_text(self) -> str:
        return self._knowledge_file.read_text(encoding="utf-8")

    def _build_index(self, chunk_size: int, chunk_overlap: int) -> FAISS:
        """Chunk the knowledge base and index the embeddings in FAISS."""
        raw_text = self._load_text()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.create_documents([raw_text])
        logger.info(
            "RAG index built: %d chunks from '%s' (%s)",
            len(chunks),
            self._knowledge_file.name,
            type(self._embeddings).__name__,
        )
        return FAISS.from_documents(chunks, self._embeddings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> str:
        """
        Return the ``top_k`` most relevant chunks for *query*, joined by blank
        lines, e.g. ``"high LDL cholesterol"``.
        """
        docs = self._vector_store.similarity_search(query, k=self.top_k)
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_for_abnormal_params(
        self,
        parameters: list,
        per_param_k: int = 2,
        max_chunks: Optional[int] = None,
    ) -> str:
        """
        Retrieve context for every abnormal parameter in a report.

        One query per abnormal parameter (e.g. ``"High LDL Cholesterol"``)
        keeps retrieval focused; results are de-duplicated and capped at
        ``max_chunks`` so the LLM prompt stays small.
        """
        abnormal = [p for p in parameters if p.get("status") in ("High", "Low")]
        if not abnormal:
            return "All parameters are within normal range."

        limit = max_chunks if max_chunks is not None else get_settings().rag_max_chunks
        seen: set[str] = set()
        chunks: list[str] = []

        for param in abnormal:
            query = f"{param['status']} {param['name']}"
            for doc in self._vector_store.similarity_search(query, k=per_param_k):
                content = doc.page_content.strip()
                if content not in seen:
                    seen.add(content)
                    chunks.append(content)
                    logger.debug("RAG [%s] -> %s", query, content[:100].replace("\n", " "))
                if len(chunks) >= limit:
                    return "\n\n".join(chunks)

        return "\n\n".join(chunks)
