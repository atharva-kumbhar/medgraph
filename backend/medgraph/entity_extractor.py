"""Entity extraction for GraphRAG seed selection."""

from __future__ import annotations

import re
from typing import Iterable

from .retrievers import RetrievalResult


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "or",
    "patient",
    "patients",
    "the",
    "to",
    "with",
}


def extract_entities(query: str, retrievals: list[RetrievalResult] | None = None) -> list[str]:
    """Rule-based entity extraction from the query and retrieved chunk metadata."""
    text = query.lower()
    if retrievals:
        text = f"{text} {' '.join(result.content[:400] for result in retrievals[:3])}"
        for result in retrievals[:3]:
            text = f"{text} {' '.join(result.entities)}"

    candidates: list[str] = []
    tokens = [token for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]
    for size in range(5, 0, -1):
        for index in range(0, len(tokens) - size + 1):
            phrase = " ".join(tokens[index : index + size]).strip()
            if len(phrase) >= 4:
                candidates.append(phrase.title())

    aliases = {
        "kidney issues": "Kidney Disease",
        "kidney problem": "Kidney Disease",
        "heart attack": "Myocardial Infarction",
        "blood thinner": "Anticoagulation",
        "shortness breath": "Shortness Of Breath",
    }
    for trigger, entity in aliases.items():
        if trigger in query.lower():
            candidates.append(entity)

    return _dedupe(candidates)


def _dedupe(items: Iterable[str], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(item.strip())
        if len(ordered) >= limit:
            break
    return ordered
