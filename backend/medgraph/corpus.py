"""Medical corpus loading for the real notebook-exported RAG dataset."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import settings
from .metrics import estimate_tokens


@dataclass(frozen=True)
class MedicalChunk:
    chunk_id: str
    content: str
    source: str
    category: str = "medical"
    entities: list[str] = field(default_factory=list)


class MedicalCorpus:
    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = dataset_path or settings.dataset_path
        self.chunks = self._load_chunks()
        self.loaded_from = str(self.dataset_path)
        self.sample_mode = False

    def _load_chunks(self) -> list[MedicalChunk]:
        if self.dataset_path.exists():
            return self._load_csv(self.dataset_path)
        raise FileNotFoundError(
            f"Medical RAG dataset not found at {self.dataset_path}. "
            "Set MEDGRAPH_DATASET_PATH to final_medical_rag_dataset.csv."
        )

    @staticmethod
    def _load_csv(path: Path) -> list[MedicalChunk]:
        _raise_csv_field_limit()
        chunks: list[MedicalChunk] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                content = (row.get("content") or row.get("text") or "").strip()
                if not content:
                    continue
                source = row.get("source") or row.get("category") or "Medical Corpus"
                chunk_id = row.get("chunk_id") or row.get("id") or f"chunk_{index}"
                entities = [
                    item.strip()
                    for item in (row.get("entities") or "").split("|")
                    if item.strip()
                ]
                chunks.append(
                    MedicalChunk(
                        chunk_id=chunk_id,
                        content=content,
                        source=source,
                        category=row.get("category") or source,
                        entities=entities,
                    )
                )
        return chunks

    def stats(self, graph_stats: dict[str, int] | None = None) -> dict[str, object]:
        graph_stats = graph_stats or {}
        loaded_tokens = sum(estimate_tokens(chunk.content) for chunk in self.chunks)
        return {
            "full_dataset_chunks": settings.full_dataset_chunks,
            "loaded_chunks": len(self.chunks),
            "sample_mode": False,
            "loaded_from": self.loaded_from,
            "sources": self.source_counts(),
            "full_dataset_tokens": settings.full_dataset_tokens,
            "loaded_tokens": loaded_tokens,
            "entities": graph_stats.get("entities", settings.tg_uploaded_vertices),
            "relationships": graph_stats.get("relationships", settings.tg_uploaded_edges),
            "graph_source": graph_stats.get("source", "configured_upload_counts"),
            "graph_configured": graph_stats.get("configured", False),
        }

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        return dict(sorted(counts.items()))

    def __iter__(self) -> Iterable[MedicalChunk]:
        return iter(self.chunks)


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)
