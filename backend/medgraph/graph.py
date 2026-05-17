"""TigerGraph retrieval and multi-hop traversal for GraphRAG."""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import requests
import urllib3

from .config import settings
from .entity_extractor import extract_entities
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

DOMAIN_BLOCKLIST = {
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "radiomics",
    "glioblastoma",
    "gbm",
    "glioma",
    "oncology",
    "tumor",
    "cancer",
    "car-t",
    "car t",
}

PAIN_MEDICATION_TERMS = {
    "acetaminophen",
    "analgesic",
    "bupivacaine",
    "codeine",
    "diclofenac",
    "hydrocodone",
    "ibuprofen",
    "lidocaine",
    "morphine",
    "naproxen",
    "nsaid",
    "opioid",
    "pain medication",
    "tramadol",
}

CARDIOLOGY_INTENT_TERMS = {
    "acute coronary syndrome",
    "angiography",
    "atherosclerosis",
    "cardiac",
    "cardiology",
    "cardiovascular",
    "chest pain",
    "coronary",
    "coronary artery disease",
    "coronary ct angiography",
    "cta",
    "ct angiography",
    "ecg",
    "heart",
    "ischemia",
    "myocardial infarction",
    "stress test",
    "troponin",
}

RENAL_INTENT_TERMS = {
    "contrast",
    "egfr",
    "kidney",
    "kidney disease",
    "nephrotoxic",
    "renal",
    "renal function",
}

DIABETES_INTENT_TERMS = {
    "diabetes",
    "diabetic",
    "glycemic",
}


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str

    @property
    def graph_weight(self) -> float:
        lowered = self.relation.lower()
        if "usually appropriate" in lowered and "not appropriate" not in lowered:
            return 1.0
        if "may be appropriate" in lowered:
            return 0.45
        if "usually not appropriate" in lowered or "not appropriate" in lowered:
            return 0.0
        return 0.35


class GraphReasoner:
    """Facade that requires live TigerGraph for GraphRAG."""

    def __init__(self) -> None:
        self.tigergraph = TigerGraphClient()

    def stats(self) -> dict[str, Any]:
        return self.tigergraph.stats()

    def reason(self, query: str, retrievals: list[RetrievalResult]) -> dict[str, Any]:
        if not self.tigergraph.configured:
            raise RuntimeError(
                "TigerGraph is not configured. Set TG_HOST, TG_GRAPH_NAME, and "
                "TG_API_TOKEN or TG_SECRET."
            )
        extracted = extract_entities(query, retrievals)
        intent = QueryIntent.from_query(query)
        seeds = self.tigergraph.match_entities(query, retrievals, intent=intent)
        traversal = self.tigergraph.traverse(seeds=seeds, max_hops=1, intent=intent)

        chunk_entities = self._chunk_entities(
            query=query,
            extracted=extracted,
            seeds=seeds,
            connected_entities=traversal.get("entities") or [],
            intent=intent,
        )
        tg_chunks = self.tigergraph.fetch_chunks_for_entities(chunk_entities, intent=intent)
                
        traversal["backend"] = "tigergraph"
        traversal["query_intent"] = intent.as_dict()
        traversal["seed_entities"] = seeds
        traversal["extracted_entities"] = extracted
        traversal["chunk_entities"] = chunk_entities
        traversal["tg_chunks"] = tg_chunks
        return traversal

    @staticmethod
    def _chunk_entities(
        *,
        query: str,
        extracted: list[str],
        seeds: list[str],
        connected_entities: list[str],
        intent: "QueryIntent",
    ) -> list[str]:
        ordered: list[str] = []
        ordered.extend(intent.priority_entities())
        if len(query.split()) <= 6:
            ordered.append(query.strip())
            ordered.extend(GraphReasoner._query_terms(query))
        ordered.extend(extracted)
        ordered.extend(seeds)
        ordered.extend(connected_entities)
        selected: list[str] = []
        seen = set()
        for entity in ordered:
            if not intent.is_relevant(entity):
                continue
            key = entity.lower()
            if key in seen:
                continue
            selected.append(entity)
            seen.add(key)
            if len(selected) >= settings.tg_chunk_entity_limit:
                break
        return selected

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        return [
            token
            for token in TOKEN_RE.findall(query)
            if len(token) >= 4 and token.lower() not in STOPWORDS
        ][:4]


@dataclass(frozen=True)
class QueryIntent:
    """Small domain filter that keeps graph traversal aligned with the user query."""

    raw_query: str
    tokens: frozenset[str]
    allowed_terms: frozenset[str]
    blocked_terms: frozenset[str]
    allow_pain_medications: bool
    min_overlap: int

    @classmethod
    def from_query(cls, query: str) -> "QueryIntent":
        lowered = query.lower()
        tokens = frozenset(TigerGraphClient._tokens(lowered))
        allowed = set(tokens)
        blocked = set()

        is_chest_pain = "chest pain" in lowered or (
            "chest" in tokens and "pain" in tokens
        )
        if is_chest_pain:
            allowed.update(CARDIOLOGY_INTENT_TERMS)
            blocked.update(DOMAIN_BLOCKLIST)

        if any(term in lowered for term in RENAL_INTENT_TERMS) or "kidney" in tokens:
            allowed.update(RENAL_INTENT_TERMS)
        if any(term in lowered for term in DIABETES_INTENT_TERMS):
            allowed.update(DIABETES_INTENT_TERMS)

        oncology_requested = cls._contains_any(lowered, DOMAIN_BLOCKLIST)
        if not oncology_requested:
            blocked.update(DOMAIN_BLOCKLIST)

        pain_med_requested = cls._contains_any(lowered, PAIN_MEDICATION_TERMS) or (
            "medication" in tokens and "pain" in tokens
        )
        if not pain_med_requested:
            blocked.update(PAIN_MEDICATION_TERMS)

        if not allowed:
            allowed.update(tokens)

        return cls(
            raw_query=query,
            tokens=tokens,
            allowed_terms=frozenset(allowed),
            blocked_terms=frozenset(blocked),
            allow_pain_medications=pain_med_requested,
            min_overlap=1 if len(tokens) >= 3 else 0,
        )

    def priority_entities(self) -> list[str]:
        lowered = self.raw_query.lower()
        if "chest pain" not in lowered and not (
            "chest" in self.tokens and "pain" in self.tokens
        ):
            return []
        entities = [
            "coronary artery disease",
        ]
        if "kidney" in self.tokens or "renal" in self.tokens:
            entities.extend(["renal function", "contrast imaging", "eGFR"])
        if "diabetes" in self.tokens or "diabetic" in self.tokens:
            entities.append("diabetes")
        entities.extend(
            [
                "acute coronary syndrome",
                "myocardial infarction",
                "ECG",
                "troponin",
                "coronary CT angiography",
                "cardiology",
            ]
        )
        return entities

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed_terms": sorted(self.allowed_terms),
            "blocked_terms": sorted(self.blocked_terms),
            "allow_pain_medications": self.allow_pain_medications,
        }

    def is_relevant(self, text: str, *, strict: bool = False) -> bool:
        lowered = (text or "").lower()
        if not lowered.strip():
            return False
        if self._contains_any(lowered, self.blocked_terms):
            return False

        text_tokens = set(TigerGraphClient._tokens(lowered))
        if text_tokens & self.allowed_terms:
            return True
        if self._contains_any(lowered, self.allowed_terms):
            return True
        if strict:
            return False
        return self.min_overlap == 0

    def relevance_score(self, text: str) -> float:
        lowered = (text or "").lower()
        if not lowered:
            return 0.0
        if self._contains_any(lowered, self.blocked_terms):
            return -100.0
        text_tokens = set(TigerGraphClient._tokens(lowered))
        score = float(len(text_tokens & self.allowed_terms))
        for term in self.allowed_terms:
            if self._contains_term(lowered, term):
                score += 3.0
        return score

    @staticmethod
    def _contains_any(text: str, terms: set[str] | frozenset[str]) -> bool:
        return any(QueryIntent._contains_term(text, term) for term in terms)

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        term = term.lower().strip()
        if not term:
            return False
        if " " in term or "-" in term:
            return term in text
        return term in set(TigerGraphClient._tokens(text))


class TigerGraphClient:
    def __init__(self) -> None:
        self.host = settings.tg_host.rstrip("/")
        self.graph_name = settings.tg_graph_name
        self.vertex_type = settings.tg_vertex_type
        self.edge_type = "RELATED_TO"
        self.relationship_edge_type = settings.tg_relationship_edge_type or "RELATED_TO"
        self.mentioned_edge_type = settings.tg_mentioned_edge_type or "MENTIONED_IN"
        self.chunk_vertex_type = settings.tg_chunk_vertex_type or "Chunk"
        self.verify_ssl = settings.tg_verify_ssl
        self.api_token = settings.tg_api_token
        self.secret = settings.tg_secret
        self.configured = bool(self.host and self.graph_name and (self.api_token or self.secret))
        self._vertices: list[str] | None = None
        self._stats: dict[str, Any] | None = None
        self._edge_cache: dict[tuple[str, str], list[GraphEdge]] = {}
        self._chunk_cache: dict[str, list[dict[str, Any]]] = {}

    def stats(self) -> dict[str, Any]:
        base = {
            "entities": settings.tg_uploaded_vertices,
            "relationships": settings.tg_uploaded_edges,
            "configured": self.configured,
            "source": "configured_upload_counts",
        }
        if not self.configured:
            return base
        if self._stats is not None:
            return self._stats

        live = dict(base)
        live["source"] = "tigergraph_configured"
        try:
            vertices = self.fetch_vertices()
            if vertices:
                live["entities"] = len(vertices)
                live["source"] = "tigergraph_vertices_live_edges_configured"
        except Exception as exc:  # pragma: no cover - external integration boundary
            live["status"] = str(exc)
        self._stats = live
        return live

    def match_entities(
        self,
        query: str,
        retrievals: list[RetrievalResult],
        limit: int = 3,
        intent: QueryIntent | None = None,
    ) -> list[str]:
        intent = intent or QueryIntent.from_query(query)
        candidates = self._candidate_phrases(query, retrievals)
        vertices = self.fetch_vertices()
        if not vertices:
            raise RuntimeError("TigerGraph returned no vertices. Verify workspace is active.")

        text = self._matching_text(query, retrievals)
        text_tokens = set(self._tokens(text))
        scored: list[tuple[float, str]] = []
        candidate_set = {candidate.lower() for candidate in candidates}

        for vertex in vertices:
            if not intent.is_relevant(vertex):
                continue
            normalized = vertex.lower()
            vertex_tokens = set(self._tokens(vertex))
            if not vertex_tokens:
                continue
            score = 0.0
            score += intent.relevance_score(vertex) * 1.8
            if normalized in text:
                score += 8.0 + min(len(normalized) / 18, 3.0)
            if normalized in candidate_set:
                score += 7.0
            overlap = vertex_tokens & text_tokens
            if overlap:
                score += len(overlap) / len(vertex_tokens) * 5.0
            for candidate in candidate_set:
                if candidate and (candidate in normalized or normalized in candidate):
                    score += 2.0
                elif candidate:
                    ratio = SequenceMatcher(None, candidate, normalized).ratio()
                    if ratio >= 0.82:
                        score += ratio * 2.5
            if score >= 2.0:
                scored.append((score, vertex))

        scored.sort(key=lambda row: row[0], reverse=True)
        seeds: list[str] = []
        seen = set()
        for _, vertex in scored:
            key = vertex.lower()
            if key not in seen:
                seeds.append(vertex)
                seen.add(key)
            if len(seeds) >= limit:
                break
        fallback = [
            candidate
            for candidate in intent.priority_entities() + candidates
            if intent.is_relevant(candidate)
        ]
        return seeds or fallback[:limit]

    def traverse(
        self,
        seeds: list[str],
        max_hops: int = 2,
        intent: QueryIntent | None = None,
    ) -> dict[str, Any]:
        intent = intent or QueryIntent.from_query(" ".join(seeds))
        nodes: dict[str, dict[str, Any]] = {}
        graph_edges: list[GraphEdge] = []
        queue = deque((seed, 0) for seed in seeds)
        visited = {seed.lower() for seed in seeds}
        errors: list[str] = []

        for seed in seeds:
            nodes[seed.lower()] = self._node(seed)

        while queue and len(graph_edges) < settings.tg_max_edges:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            try:
                current_edges = self.fetch_related_edges(current)
            except Exception as exc:
                errors.append(f"{current}: {exc}")
                continue
            relevant_edges = self._filter_edges(current_edges, intent)
            for edge in relevant_edges[:5]:
                graph_edges.append(edge)
                nodes[edge.source.lower()] = self._node(edge.source)
                nodes[edge.target.lower()] = self._node(edge.target)
                if edge.target.lower() not in visited and depth + 1 <= max_hops:
                    visited.add(edge.target.lower())
                    queue.append((edge.target, depth + 1))
                if len(graph_edges) >= settings.tg_max_edges:
                    break

        deduped_edges = self._dedupe(graph_edges)
        return {
            "nodes": list(nodes.values()),
            "edges": [
                {"source": edge.source, "target": edge.target, "relation": edge.relation}
                for edge in deduped_edges
            ],
            "paths": self._paths_from_edges(deduped_edges),
            "neighbors": self._neighbors(deduped_edges),
            "entities": [node["id"] for node in nodes.values()],
            "status": self._traversal_status(deduped_edges, errors),
            "errors": errors[:5],
        }

    def fetch_vertices(self) -> list[str]:
        if self._vertices is not None:
            return self._vertices
        encoded_graph = quote(self.graph_name, safe="")
        encoded_vertex_type = quote(self.vertex_type, safe="")
        path = (
            f"/restpp/graph/{encoded_graph}/vertices/{encoded_vertex_type}"
            f"?limit={settings.tg_vertex_limit}"
        )
        payload = self._get_json(path)
        vertices = []
        for item in payload.get("results") or []:
            vertex_id = item.get("v_id") or item.get("id")
            if vertex_id:
                vertices.append(str(vertex_id))
        self._vertices = vertices
        return vertices

    def fetch_related_edges(self, vertex_id: str) -> list[GraphEdge]:
        return self.fetch_edges_by_type(vertex_id, self.relationship_edge_type)

    def fetch_edges_by_type(self, vertex_id: str, edge_type: str) -> list[GraphEdge]:
        cache_key = (edge_type, vertex_id)
        if cache_key in self._edge_cache:
            return self._edge_cache[cache_key]
        encoded_graph = quote(self.graph_name, safe="")
        encoded_vertex_type = quote(self.vertex_type, safe="")
        encoded_vertex = quote(vertex_id, safe="")
        encoded_edge = quote(edge_type, safe="")
        url = (
            f"{self.host}/restpp/graph/{encoded_graph}/edges/"
            f"{encoded_vertex_type}/{encoded_vertex}/{encoded_edge}"
        )
        payload = self._get_json_url(url)

        edges: list[GraphEdge] = []
        for item in payload.get("results") or []:
            edge = self._edge_from_payload(vertex_id, item, edge_type)
            if edge:
                edges.append(edge)
        self._edge_cache[cache_key] = edges
        return edges

    def fetch_chunks_for_entities(
        self, entity_names: list[str], intent: QueryIntent | None = None
    ) -> list[dict[str, Any]]:
        intent = intent or QueryIntent.from_query(" ".join(entity_names))
        chunks: list[dict[str, Any]] = []
        seen = set()
        errors: list[str] = []
        for entity_name in entity_names[: settings.tg_chunk_entity_limit]:
            try:
                entity_chunks = self.fetch_chunks(entity_name)
            except Exception as exc:
                errors.append(f"{entity_name}: {exc}")
                continue
            for chunk in entity_chunks[: settings.tg_chunk_fetch_limit]:
                normalized = self._normalize_chunk_payload(chunk, source_entity=entity_name)
                relevance_text = (
                    f"{normalized.get('source_entity', '')} "
                    f"{normalized.get('chunk_id', '')} "
                    f"{normalized.get('category', '')} "
                    f"{normalized.get('content', '')}"
                )
                if not intent.is_relevant(relevance_text, strict=True):
                    continue
                dedupe_key = (
                    normalized.get("chunk_id")
                    or normalized.get("content")
                    or json.dumps(normalized, sort_keys=True)
                )
                dedupe_key = str(dedupe_key).lower()
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                chunks.append(normalized)
        if errors and not chunks:
            raise RuntimeError(
                "TigerGraph get_chunks retrieval failed: "
                + "; ".join(errors[:3])
            )
        return chunks

    @staticmethod
    def _filter_edges(edges: list[GraphEdge], intent: QueryIntent) -> list[GraphEdge]:
        relevant = [
            edge
            for edge in edges
            if TigerGraphClient._is_usable_recommendation_edge(edge)
            and intent.is_relevant(
                f"{edge.source} {edge.relation} {edge.target}",
                strict=True,
            )
        ]
        relevant.sort(
            key=lambda edge: intent.relevance_score(
                f"{edge.source} {edge.relation} {edge.target}"
            ),
            reverse=True,
        )
        return relevant

    @staticmethod
    def _is_usable_recommendation_edge(edge: GraphEdge) -> bool:
        relation = edge.relation.lower()
        if "appropriate" not in relation:
            return False
        if "usually not appropriate" in relation or "not appropriate" in relation:
            return False
        if "may be appropriate" in relation:
            return False
        return "usually appropriate" in relation

    def fetch_chunks(self, entity_name: str) -> list[dict[str, Any]]:
        cache_key = entity_name.lower()
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]
        encoded_graph = quote(self.graph_name, safe="")
        url = f"{self.host}/restpp/query/{encoded_graph}/get_chunks"
        headers = {"Accept": "application/json"}
        token = self._token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(
            url,
            headers=headers,
            params={"entity_name": entity_name},
            timeout=settings.request_timeout_seconds,
            verify=self.verify_ssl,
        )
        if not response.ok:
            raise RuntimeError(
                f"HTTP {response.status_code} from TigerGraph get_chunks: "
                f"{response.text[:600]}"
            )
        payload = response.json()
        if payload.get("error"):
            raise ValueError(str(payload.get("message") or "TigerGraph get_chunks error"))
        chunks = self._chunks_from_query_payload(payload)
        self._chunk_cache[cache_key] = chunks
        return chunks

    def _get_json(self, path: str) -> dict[str, Any]:
        return self._get_json_url(f"{self.host}{path}")

    def _get_json_url(self, url: str) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        token = self._token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            if not self.verify_ssl:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(
                url,
                headers=headers,
                timeout=settings.request_timeout_seconds,
                verify=self.verify_ssl,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"TigerGraph request failed: {exc}") from exc
        if not response.ok:
            raise RuntimeError(
                f"HTTP {response.status_code} from TigerGraph: {response.text[:600]}"
            )
        raw = response.text
        data = json.loads(raw)
        if data.get("error"):
            raise ValueError(str(data.get("message") or "TigerGraph API error"))
        return data

    def _token(self) -> str:
        if self.api_token:
            return self.api_token
        if not self.secret:
            return ""
        encoded_secret = quote(self.secret, safe="")
        payload = self._request_token(encoded_secret)
        token = payload.get("token") or payload.get("results")
        if isinstance(token, list) and token:
            token = token[0]
        self.api_token = str(token or "")
        return self.api_token

    def _request_token(self, encoded_secret: str) -> dict[str, Any]:
        url = f"{self.host}/restpp/requesttoken?secret={encoded_secret}"
        request = Request(url, headers={"Accept": "application/json"}, method="GET")
        try:
            with urlopen(request, timeout=settings.request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:600]
            raise RuntimeError(f"Could not request TigerGraph token: {details}") from exc

    @staticmethod
    def _chunks_from_query_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        def visit(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if not isinstance(value, dict):
                return
            attrs = value.get("attributes") if isinstance(value.get("attributes"), dict) else {}
            if (
                value.get("v_type") == settings.tg_chunk_vertex_type
                or value.get("chunk_id")
                or value.get("content")
                or value.get("text")
                or attrs.get("chunk_id")
                or attrs.get("content")
                or attrs.get("text")
            ):
                chunks.append(value)
                return
            for item in value.values():
                visit(item)

        visit(payload.get("results") or [])
        return chunks

    @staticmethod
    def _normalize_chunk_payload(chunk: dict[str, Any], *, source_entity: str) -> dict[str, Any]:
        attrs = chunk.get("attributes") if isinstance(chunk.get("attributes"), dict) else {}
        content = (
            attrs.get("content")
            or attrs.get("text")
            or chunk.get("content")
            or chunk.get("text")
            or ""
        )
        chunk_id = (
            attrs.get("chunk_id")
            or chunk.get("chunk_id")
            or chunk.get("v_id")
            or f"{source_entity}:chunk"
        )
        return {
            **chunk,
            "source_entity": source_entity,
            "attributes": attrs,
            "chunk_id": str(chunk_id),
            "source": attrs.get("source") or chunk.get("source") or "TigerGraph",
            "category": attrs.get("category") or chunk.get("category") or "graph",
            "content": str(content),
        }

    @staticmethod
    def _edge_from_payload(vertex_id: str, item: dict[str, Any], edge_type: str) -> GraphEdge | None:
        attrs = item.get("attributes") or {}
        relation = (
            attrs.get("relation")
            or attrs.get("relation STRING")
            or attrs.get("relationship")
            or attrs.get("relationship STRING")
            or edge_type
        )
        source = str(item.get("from_id") or item.get("source_id") or vertex_id)
        target = str(item.get("to_id") or item.get("target_id") or "")
        if not target and item.get("v_id"):
            target = str(item["v_id"])
        if not source or not target:
            return None
        return GraphEdge(source=source, target=target, relation=str(relation))

    @staticmethod
    def _candidate_phrases(query: str, retrievals: list[RetrievalResult]) -> list[str]:
        text = TigerGraphClient._matching_text(query, retrievals)
        words = TigerGraphClient._tokens(text)
        candidates: list[str] = []
        for size in range(5, 0, -1):
            for index in range(0, len(words) - size + 1):
                phrase = " ".join(words[index : index + size]).strip()
                if phrase:
                    candidates.extend([phrase, phrase.title()])

        aliases = {
            "kidney issues": "kidney disease",
            "kidney problem": "kidney disease",
            "heart attack": "myocardial infarction",
            "chest pain": "chest pain",
            "chest pain diabetes": "acute coronary syndrome",
            "chest pain kidney": "coronary artery disease",
            "coronary disease": "coronary artery disease",
            "shortness breath": "shortness of breath",
            "blood thinner": "anticoagulation",
        }
        lowered = text.lower()
        for trigger, entity in aliases.items():
            if trigger in lowered:
                candidates.extend([entity, entity.title()])

        deduped: list[str] = []
        seen = set()
        for candidate in candidates:
            key = candidate.lower()
            if key not in seen:
                deduped.append(candidate)
                seen.add(key)
        return deduped

    @staticmethod
    def _matching_text(query: str, retrievals: list[RetrievalResult]) -> str:
        snippets = " ".join(result.content[:500] for result in retrievals[:4])
        entities = " ".join(" ".join(result.entities) for result in retrievals[:4])
        return f"{query} {entities} {snippets}".lower()

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            match.group(0).lower()
            for match in TOKEN_RE.finditer(text)
            if match.group(0).lower() not in STOPWORDS
        ]

    @staticmethod
    def _node(entity: str) -> dict[str, Any]:
        return {
            "id": entity,
            "label": entity,
            "type": TigerGraphClient._node_type(entity),
        }

    @staticmethod
    def _node_type(entity: str) -> str:
        lowered = entity.lower()
        if any(term in lowered for term in ["pain", "breath", "symptom"]):
            return "Symptom"
        if any(term in lowered for term in ["syndrome", "embolism", "disease", "failure", "cancer"]):
            return "Disease"
        if any(term in lowered for term in ["ecg", "mri", "ct", "test", "troponin", "imaging"]):
            return "Test"
        if any(term in lowered for term in ["aspirin", "drug", "medication", "anticoag"]):
            return "Drug"
        return "Entity"

    @staticmethod
    def _paths_from_edges(edges: list[GraphEdge]) -> list[str]:
        return [f"{edge.source} -> {edge.relation} -> {edge.target}" for edge in edges[:12]]

    @staticmethod
    def _neighbors(edges: list[GraphEdge]) -> dict[str, list[str]]:
        neighbors: dict[str, list[str]] = {}
        for edge in edges:
            neighbors.setdefault(edge.source, []).append(f"{edge.relation}: {edge.target}")
        return neighbors

    @staticmethod
    def _traversal_status(edges: list[GraphEdge], errors: list[str]) -> str:
        if edges and errors:
            return "TigerGraph REST RELATED_TO traversal completed with skipped vertices."
        if edges:
            return "TigerGraph REST RELATED_TO traversal"
        if errors:
            return "TigerGraph RELATED_TO traversal returned no usable edges; chunks were retrieved with get_chunks where available."
        return "TigerGraph returned no RELATED_TO edges for the matched entities."

    @staticmethod
    def _dedupe(edges: list[GraphEdge]) -> list[GraphEdge]:
        deduped = []
        seen = set()
        for edge in edges:
            key = (edge.source.lower(), edge.relation.lower(), edge.target.lower())
            if key not in seen:
                deduped.append(edge)
                seen.add(key)
        return deduped
