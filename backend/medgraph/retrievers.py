"""Real FAISS retrieval over the medical RAG dataset.

The Colab pipeline used sentence-transformers/all-MiniLM-L6-v2 embeddings and
FAISS similarity search. This module implements that same flow and persists the
index locally so the 13,807 chunk dataset does not need to be embedded on every
server start.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings
from .corpus import MedicalChunk


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    source: str
    category: str
    content: str
    similarity: float
    entities: list[str]


class FaissMedicalRetriever:
    """SentenceTransformer embedding + FAISS inner-product search."""

    def __init__(
        self,
        chunks: list[MedicalChunk],
        dataset_path: Path | None = None,
        model_name: str | None = None,
        index_dir: Path | None = None,
    ) -> None:
        self.chunks = chunks
        self.dataset_path = dataset_path or settings.dataset_path
        self.model_name = model_name or settings.embedding_model
        self.index_dir = index_dir or settings.index_dir
        self.index = None
        self.model = None
        self.dimension = 0
        self._lock = threading.Lock()
        self._ready = False

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        query = query.strip()
        if not query:
            return []
        self._ensure_ready()

        import faiss  # type: ignore
        import numpy as np

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector = np.asarray(vector, dtype="float32")
        faiss.normalize_L2(vector)
        count = min(k, len(self.chunks))
        scores, indices = self.index.search(vector, count)

        results: list[RetrievalResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    category=chunk.category,
                    content=chunk.content,
                    similarity=round(max(0.0, min(float(score), 1.0)), 4),
                    entities=chunk.entities,
                )
            )
        return results

    def rank_texts(
        self,
        query: str,
        items: list[dict[str, Any]],
        *,
        text_key: str = "content",
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Rank arbitrary retrieved text payloads by query embedding similarity."""
        query = query.strip()
        if not query or not items or limit <= 0:
            return []
        self._ensure_ready()

        import numpy as np

        texts = [str(item.get(text_key) or "") for item in items]
        valid = [(index, text) for index, text in enumerate(texts) if text.strip()]
        if not valid:
            return []

        vectors = self.model.encode(
            [query] + [text for _, text in valid],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype="float32")
        query_vector = vectors[0]
        scored: list[tuple[float, int]] = []
        for offset, (item_index, _) in enumerate(valid, start=1):
            score = float(np.dot(query_vector, vectors[offset]))
            scored.append((max(0.0, min(score, 1.0)), item_index))

        scored.sort(key=lambda row: row[0], reverse=True)
        ranked: list[dict[str, Any]] = []
        seen_content = set()
        for score, item_index in scored:
            item = dict(items[item_index])
            content_key = " ".join(str(item.get(text_key) or "").split()).lower()
            if not content_key or content_key in seen_content:
                continue
            seen_content.add(content_key)
            item["similarity"] = round(score, 4)
            ranked.append(item)
            if len(ranked) >= limit:
                break
        return ranked

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._load_dependencies()
            self.index_dir.mkdir(parents=True, exist_ok=True)
            cache = self._cache_paths()
            if self._cache_is_valid(cache):
                self.index = self._read_index(cache["index"])
            else:
                self.index = self._build_index()
                self._write_index(cache["index"], self.index)
                cache["meta"].write_text(
                    json.dumps(self._metadata(), indent=2), encoding="utf-8"
                )
            self._ready = True

    def _load_dependencies(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Basic RAG requires sentence-transformers. Install it and restart the backend."
            ) from exc
        try:
            import faiss  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Basic RAG requires faiss-cpu. Install it and restart.") from exc

        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(self.model_name)

    def _build_index(self) -> Any:
        import faiss  # type: ignore
        import numpy as np

        texts = [chunk.content for chunk in self.chunks]
        if not texts:
            raise RuntimeError("The medical RAG dataset is empty; cannot build FAISS index.")

        embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        self.dimension = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)
        return index

    @staticmethod
    def _read_index(path: Path) -> Any:
        import faiss  # type: ignore

        return faiss.read_index(str(path))

    @staticmethod
    def _write_index(path: Path, index: Any) -> None:
        import faiss  # type: ignore

        faiss.write_index(index, str(path))

    def _cache_paths(self) -> dict[str, Path]:
        digest = hashlib.sha256(
            json.dumps(self._metadata(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:18]
        stem = f"medical_{digest}"
        return {
            "index": self.index_dir / f"{stem}.faiss",
            "meta": self.index_dir / f"{stem}.json",
        }

    def _metadata(self) -> dict[str, Any]:
        stat = self.dataset_path.stat() if self.dataset_path.exists() else None
        return {
            "dataset_path": str(self.dataset_path.resolve()),
            "dataset_size": stat.st_size if stat else None,
            "dataset_mtime_ns": stat.st_mtime_ns if stat else None,
            "chunk_count": len(self.chunks),
            "embedding_model": self.model_name,
            "index_type": "faiss.IndexFlatIP",
            "normalized": True,
        }

    def _cache_is_valid(self, cache: dict[str, Path]) -> bool:
        if not cache["index"].exists() or not cache["meta"].exists():
            return False
        try:
            cached = json.loads(cache["meta"].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return cached == self._metadata()


# Preserve the old import contract while making the implementation real FAISS.
HybridRetriever = FaissMedicalRetriever
