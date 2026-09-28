from __future__ import annotations

import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from clinical_assistant.models import GuidelineEvidence

DEFAULT_GUIDELINES_DIR = Path("data/guidelines")
DEFAULT_CHROMA_DIR = Path("data/runtime/chroma")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "guidelines"

CHUNK_MAX_CHARS = 800          # trigger recursive-split fallback above this size
CHUNK_OVERLAP_CHARS = 90
CANDIDATE_POOL_MULTIPLIER = 5  # each retriever pulls top_k * this before fusion
FUSION_WEIGHTS = [0.5, 0.5]    # [dense, lexical] — provisional, expected to be
                               # tuned down for lexical once more eval data exists

_HEADING_RE = re.compile(r"^#+\s+(.*)$", re.MULTILINE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _split_into_chunks(source: str, text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text on headings; returns (heading, chunk_text) pairs."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("(full document)", text.strip())] if text.strip() else []

    chunks: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chunks.append((heading, body))
    return chunks


def _split_oversized_chunk(
    source: str, heading: str, text: str
) -> list[tuple[str, str, str]]:
    """If text fits within CHUNK_MAX_CHARS, return it unchanged as a single chunk.
    Otherwise, recursively split it and return multiple chunks, all tagged with the
    same (source, heading) — heading identity is preserved even when a section gets
    subdivided.
    """
    if len(text) <= CHUNK_MAX_CHARS:
        return [(source, heading, text)]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_MAX_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [(source, heading, piece) for piece in splitter.split_text(text)]


def _load_guideline_chunks(
    guidelines_dir: Path,
) -> list[tuple[str, str, str]]:
    """Returns (source_filename, heading, chunk_text) tuples."""
    chunks: list[tuple[str, str, str]] = []
    for path in sorted(guidelines_dir.glob("*")):
        if path.suffix.lower() in (".md", ".txt"):
            text = _read_text_file(path)
        elif path.suffix.lower() == ".pdf":
            text = _read_pdf_file(path)
        else:
            continue

        for heading, body in _split_into_chunks(path.name, text):
            chunks.extend(_split_oversized_chunk(path.name, heading, body))
    return chunks


def _hash_guideline_files(guidelines_dir: Path) -> str:
    """Stable hash over every guideline file's name and content, so any edit,
    addition, or removal changes the hash. Embedding model and chunking params
    are included too, so tuning them invalidates the cached collection and keeps
    the Chroma and BM25 chunk lists identical."""
    hasher = hashlib.sha256()
    hasher.update(
        f"{EMBEDDING_MODEL_NAME}|{CHUNK_MAX_CHARS}|{CHUNK_OVERLAP_CHARS}".encode("utf-8")
    )
    for path in sorted(guidelines_dir.glob("*")):
        if path.suffix.lower() in (".md", ".txt", ".pdf"):
            hasher.update(path.name.encode("utf-8"))
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


class _ChromaAdapter(BaseRetriever):
    """Wraps the existing raw chromadb collection as a LangChain retriever,
    so it can be combined via EnsembleRetriever without migrating away from
    the raw chromadb client (which the hash-caching logic depends on)."""

    collection: object
    k: int = 15
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=[query], n_results=min(self.k, count)
        )
        return [
            Document(page_content=doc_text, metadata=meta)
            for doc_text, meta in zip(
                results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]
            )
        ]


class _BM25Adapter(BaseRetriever):
    chunks: list  # list[tuple[str, str, str]] — (source, heading, text)
    bm25: object
    k: int = 15
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self.k]
        return [
            Document(
                page_content=self.chunks[i][2],
                metadata={"source": self.chunks[i][0], "section": self.chunks[i][1]},
            )
            for i in top_indices
        ]


class GuidelineRetriever:
    def __init__(
        self,
        guidelines_dir: Path | str = DEFAULT_GUIDELINES_DIR,
        chroma_dir: Path | str = DEFAULT_CHROMA_DIR,
    ) -> None:
        self.guidelines_dir = Path(guidelines_dir)
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self._client = chromadb.PersistentClient(path=str(self.chroma_dir))

        current_hash = _hash_guideline_files(self.guidelines_dir)
        try:
            collection = self._client.get_collection(
                COLLECTION_NAME, embedding_function=self._embedding_fn
            )
            stored_hash = (
                collection.metadata.get("corpus_hash") if collection.metadata else None
            )
        except Exception:
            collection = None
            stored_hash = None

        if collection is not None and stored_hash == current_hash:
            print(f"GuidelineRetriever: reusing cached embeddings (hash {current_hash[:8]})")
            self._collection = collection
            self._chunks = _load_guideline_chunks(self.guidelines_dir)  # still needed for BM25
        else:
            print(f"GuidelineRetriever: rebuilding embeddings (hash {current_hash[:8]})")
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._chunks = _load_guideline_chunks(self.guidelines_dir)
            self._collection = self._client.create_collection(
                COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"corpus_hash": current_hash},
            )
            if self._chunks:
                self._collection.add(
                    ids=[f"{source}::{i}" for i, (source, _, _) in enumerate(self._chunks)],
                    documents=[body for _, _, body in self._chunks],
                    metadatas=[
                        {"source": source, "section": heading}
                        for source, heading, _ in self._chunks
                    ],
                )

        # BM25 is cheap, so it is rebuilt on every construction rather than cached.
        tokenized = [_tokenize(body) for _, _, body in self._chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

        # TODO: pool_size assumes the default top_k=3. If search() is called with a
        # larger top_k, the candidate pool does not widen to match.
        pool_size = 3 * CANDIDATE_POOL_MULTIPLIER
        chroma_adapter = _ChromaAdapter(collection=self._collection, k=pool_size)
        bm25_adapter = _BM25Adapter(chunks=self._chunks, bm25=self._bm25, k=pool_size)
        self._ensemble = EnsembleRetriever(
            retrievers=[chroma_adapter, bm25_adapter], weights=FUSION_WEIGHTS
        )

    def search(self, query: str, top_k: int = 3) -> list[GuidelineEvidence]:
        if not self._chunks:
            return []
        fused_docs = self._ensemble.invoke(query)
        return [
            GuidelineEvidence(
                source=doc.metadata.get("source", "unknown"),
                section=doc.metadata.get("section", ""),
                text=doc.page_content,
                # Positional proxy for fused rank order — EnsembleRetriever does not
                # expose the raw RRF sum, only the final ranked list.
                score=1.0 / (rank + 1),
            )
            for rank, doc in enumerate(fused_docs[:top_k])
        ]
