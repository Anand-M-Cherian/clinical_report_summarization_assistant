from __future__ import annotations

import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

from clinical_assistant.models import GuidelineEvidence

DEFAULT_GUIDELINES_DIR = Path("data/guidelines")
DEFAULT_CHROMA_DIR = Path("data/runtime/chroma")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "guidelines"

_HEADING_RE = re.compile(r"^#+\s+(.*)$", re.MULTILINE)


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
            chunks.append((path.name, heading, body))
    return chunks


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
        self._collection = self._build_collection()

    def _build_collection(self):
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = self._client.create_collection(
            COLLECTION_NAME, embedding_function=self._embedding_fn
        )

        chunks = _load_guideline_chunks(self.guidelines_dir)
        if chunks:
            collection.add(
                ids=[f"{source}::{i}" for i, (source, _, _) in enumerate(chunks)],
                documents=[body for _, _, body in chunks],
                metadatas=[
                    {"source": source, "section": heading}
                    for source, heading, _ in chunks
                ],
            )
        return collection

    def search(self, query: str, top_k: int = 3) -> list[GuidelineEvidence]:
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
        )

        evidence: list[GuidelineEvidence] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            score = 1.0 / (1.0 + distance)
            evidence.append(
                GuidelineEvidence(
                    source=meta.get("source", "unknown"),
                    section=meta.get("section", ""),
                    text=doc,
                    score=score,
                )
            )
        return evidence
