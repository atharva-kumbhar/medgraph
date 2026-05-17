"""Three real benchmark pipelines: LLM-only, Basic RAG, and TigerGraph GraphRAG."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any, Callable

from .config import settings
from .corpus import MedicalCorpus
from .evaluator import AccuracyEvaluator, AccuracyResult
from .graph import GraphReasoner
from .llm import GenerationResult, LLMClient
from .metrics import percent_reduction, timed
from .retrievers import HybridRetriever, RetrievalResult


class PipelineService:
    def __init__(self) -> None:
        self.corpus = MedicalCorpus()
        self.retriever = HybridRetriever(self.corpus.chunks, dataset_path=self.corpus.dataset_path)
        self.graph = GraphReasoner()
        self.llm = LLMClient()
        self.evaluator = AccuracyEvaluator(self.llm)

    def stats(self) -> dict[str, Any]:
        payload = self.corpus.stats(self.graph.stats())
        payload["llm_providers"] = {
            "production": settings.production_provider,
            "llm_only": settings.llm_only_provider,
            "basic_rag": settings.rag_provider,
            "graphrag": settings.graphrag_provider,
            "judge": "gemini",
            "test": settings.test_provider,
            "gemini_model": settings.gemini_model.removeprefix("models/"),
            "nvidia_model": settings.llm_model,
        }
        return payload

    def run_llm_only_api(
        self, query: str, reference_answer: str | None = None, *, llm_provider: str | None = None
    ) -> dict[str, Any]:
        query = self._clean_query(query)
        provider = llm_provider
        result = timed(
            lambda: self._run_llm_only(query, reference_answer, provider=provider)
        )
        pipeline = self._attach_latency(result.value, result.latency_ms)
        return self._public_response(pipeline)

    def run_basic_rag_api(
        self, query: str, reference_answer: str | None = None, *, llm_provider: str | None = None
    ) -> dict[str, Any]:
        query = self._clean_query(query)
        provider = llm_provider
        result = timed(
            lambda: self._run_basic_rag(query, reference_answer, provider=provider)
        )
        pipeline = self._attach_latency(result.value, result.latency_ms)
        return self._public_response(pipeline)

    def run_graphrag_api(
        self, query: str, reference_answer: str | None = None, *, llm_provider: str | None = None
    ) -> dict[str, Any]:
        query = self._clean_query(query)
        provider = llm_provider
        result = timed(
            lambda: self._run_graphrag(query, reference_answer, provider=provider)
        )
        pipeline = self._attach_latency(result.value, result.latency_ms)
        return self._public_response(pipeline)

    def run_metrics_api(self, query: str, reference_answer: str | None = None) -> dict[str, Any]:
        """Return dashboard comparison metrics without duplicating full pipeline payloads."""
        result = self.run_query(query, reference_answer)
        pipelines = result["pipelines"]
        return {
            "query": result["query"],
            "expected_answer": result.get("expected_answer"),
            "expected_answer_source": result.get("expected_answer_source"),
            "pipelines": {
                key: {
                    "label": pipelines[key]["label"],
                    "answer": pipelines[key].get("answer", ""),
                    "error": pipelines[key].get("error"),
                    "metrics": pipelines[key]["metrics"],
                    "graph_paths": (pipelines[key].get("graph") or {}).get("paths", [])
                    if key == "graphrag"
                    else [],
                    "retrieved_chunks": len(pipelines[key].get("evidence") or []),
                }
                for key in ("llm_only", "basic_rag", "graphrag")
            },
            "comparison": result["comparison"],
            "graph": {
                "paths": result.get("graph", {}).get("paths", []),
                "nodes": len(result.get("graph", {}).get("nodes", [])),
                "edges": len(result.get("graph", {}).get("edges", [])),
                "backend": result.get("graph", {}).get("backend"),
                "status": result.get("graph", {}).get("status"),
            },
            "metric_sources": result["metric_sources"],
            "stats": result["stats"],
        }

    def run_query(
        self,
        query: str,
        reference_answer: str | None = None,
        *,
        llm_provider: str | None = None,
    ) -> dict[str, Any]:
        import logging
        logger = logging.getLogger(__name__)
        query = self._clean_query(query)
        provider = llm_provider
        reference_answer = self._clean_optional_text(reference_answer)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                "llm_only": executor.submit(
                    self._safe_timed,
                    lambda: self._run_llm_only(query, reference_answer, provider=provider),
                    "llm_only",
                    "LLM Only",
                    "red",
                ),
                "basic_rag": executor.submit(
                    self._safe_timed,
                    lambda: self._run_basic_rag(query, reference_answer, provider=provider),
                    "basic_rag",
                    "Basic RAG",
                    "yellow",
                ),
                "graphrag": executor.submit(
                    self._safe_timed,
                    lambda: self._run_graphrag(query, reference_answer, provider=provider),
                    "graphrag",
                    "TigerGraph GraphRAG",
                    "cyan",
                ),
            }
            timed_results = {}
            for key, future in futures.items():
                try:
                    timed_results[key] = future.result(timeout=180)
                except Exception as exc:
                    logger.exception(f"Pipeline {key} failed: {exc}")
                    timed_results[key] = self._error_result(key, str(exc))

        pipelines = {
            key: self._attach_latency(result.value, result.latency_ms)
            for key, result in timed_results.items()
        }

        expected_answer = reference_answer
        expected_answer_source = "provided"
        if not expected_answer:
            expected_answer, expected_answer_source = self._generate_expected_answer(query, pipelines)
            if expected_answer:
                self._score_pipelines_against_expected(query, pipelines, expected_answer)

        self._attach_comparison_metrics(pipelines)

        basic_evidence = pipelines["basic_rag"].get("evidence") or []
        graph_payload = pipelines["graphrag"].get("graph") or self._empty_graph()
        return {
            "query": query,
            "expected_answer": expected_answer,
            "expected_answer_source": expected_answer_source,
            "pipelines": pipelines,
            "retrieval": {
                "top_chunks": basic_evidence,
                "entities": sorted({entity for chunk in basic_evidence for entity in chunk.get("entities", [])}),
            },
            "graph": graph_payload,
            "comparison": self._comparison(pipelines),
            "llm_provider": provider,
            "metric_sources": {
                "tokens": "Hosted LLM API usage when returned; otherwise estimated from prompt and answer text.",
                "cost": "Calculated from tokens and MEDGRAPH_PRICE_PER_1K_TOKENS.",
                "latency": "Measured wall-clock time for the real backend pipeline, including retrieval, graph traversal, hosted generation, and judging.",
                "accuracy": "Gemini LLM-as-a-judge PASS/FAIL against the generated expected_answer.",
                "bertscore": "BERTScore F1 between the generated answer and expected_answer when bert-score is installed.",
                "hallucination_risk": "LLM-as-a-judge assessment against retrieved chunks and/or TigerGraph context.",
                "graph": "TigerGraph REST traversal only. No local graph fallback is used.",
                "llm": "LLM-only, Basic RAG, and GraphRAG default to NVIDIA. Gemini generates expected_answer and performs LLM-as-a-judge scoring.",
            },
            "stats": self.stats(),
        }

    def run_benchmark(self, items: list[dict[str, str]]) -> dict[str, Any]:
        rows = []
        aggregate = {
            "llm_only": self._empty_benchmark_bucket(),
            "basic_rag": self._empty_benchmark_bucket(),
            "graphrag": self._empty_benchmark_bucket(),
        }
        for item in items:
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            result = self.run_query(
                question,
                item.get("correct_answer"),
            )
            rows.append(result)
            for key, bucket in aggregate.items():
                metrics = result["pipelines"][key]["metrics"]
                bucket["tokens"] += metrics["tokens"]
                bucket["latency_ms"] += metrics["latency_ms"]
                bucket["cost_usd"] += metrics["cost_usd"]
                if metrics["accuracy"] is not None:
                    bucket["judge_pass_percent"] += metrics["accuracy"]
                    bucket["judge_evaluated"] += 1
                    if metrics["accuracy"] >= 100:
                        bucket["judge_passes"] += 1
                if metrics.get("bertscore_f1") is not None:
                    bucket["bertscore_f1"] += metrics["bertscore_f1"]
                    bucket["bertscore_evaluated"] += 1

        count = max(len(rows), 1)
        for key, bucket in aggregate.items():
            bucket["avg_tokens"] = int(bucket.pop("tokens") / count)
            bucket["avg_latency_ms"] = int(bucket.pop("latency_ms") / count)
            bucket["avg_cost_usd"] = round(bucket.pop("cost_usd") / count, 6)
            bucket["judge_pass_percent"] = (
                round(bucket["judge_pass_percent"] / bucket["judge_evaluated"], 1)
                if bucket["judge_evaluated"]
                else None
            )
            bucket["avg_bertscore_f1"] = (
                round(bucket.pop("bertscore_f1") / bucket["bertscore_evaluated"], 4)
                if bucket["bertscore_evaluated"]
                else None
            )

            # Backward-compatible aliases for older scripts.
            bucket["tokens"] = bucket["avg_tokens"]
            bucket["latency_ms"] = bucket["avg_latency_ms"]
            bucket["cost_usd"] = bucket["avg_cost_usd"]
            bucket["accuracy"] = bucket["judge_pass_percent"]
            bucket["bertscore_f1"] = bucket["avg_bertscore_f1"]

        improvements = self._benchmark_improvements(aggregate)

        return {
            "items_evaluated": len(rows),
            "llm_provider": "per-pipeline defaults",
            "generation_providers": {
                "llm_only": settings.llm_only_provider,
                "basic_rag": settings.rag_provider,
                "graphrag": settings.graphrag_provider,
            },
            "judge_provider": "gemini",
            "aggregate": aggregate,
            "improvements": improvements,
            "token_reduction_percent": improvements["graph_token_reduction_vs_rag_percent"],
            "cost_reduction_percent": improvements["graph_cost_reduction_vs_rag_percent"],
            "accuracy_improvement_percent": improvements["graph_accuracy_improvement_vs_rag_percent"],
            "rows": rows,
        }

    @staticmethod
    def _empty_benchmark_bucket() -> dict[str, Any]:
        return {
            "tokens": 0,
            "latency_ms": 0,
            "cost_usd": 0.0,
            "judge_pass_percent": 0.0,
            "judge_passes": 0,
            "judge_evaluated": 0,
            "bertscore_f1": 0.0,
            "bertscore_evaluated": 0,
        }

    @staticmethod
    def _benchmark_improvements(aggregate: dict[str, dict[str, Any]]) -> dict[str, Any]:
        rag = aggregate["basic_rag"]
        graph = aggregate["graphrag"]
        rag_accuracy = rag.get("judge_pass_percent")
        graph_accuracy = graph.get("judge_pass_percent")
        rag_bert = rag.get("avg_bertscore_f1")
        graph_bert = graph.get("avg_bertscore_f1")
        accuracy_delta = (
            round(graph_accuracy - rag_accuracy, 1)
            if graph_accuracy is not None and rag_accuracy is not None
            else None
        )
        bert_delta = (
            round(graph_bert - rag_bert, 4)
            if graph_bert is not None and rag_bert is not None
            else None
        )
        accuracy_improvement = None
        if accuracy_delta is not None and rag_accuracy not in (None, 0):
            accuracy_improvement = round((accuracy_delta / rag_accuracy) * 100, 1)
        latency_improvement = percent_reduction(
            rag["avg_latency_ms"], graph["avg_latency_ms"]
        )
        bert_improvement = None
        if bert_delta is not None and rag_bert not in (None, 0):
            bert_improvement = round((bert_delta / rag_bert) * 100, 1)
        return {
            "graph_token_reduction_vs_rag_percent": percent_reduction(
                rag["avg_tokens"], graph["avg_tokens"]
            ),
            "graph_cost_reduction_vs_rag_percent": percent_reduction(
                rag["avg_cost_usd"], graph["avg_cost_usd"]
            ),
            "graph_latency_improvement_vs_rag_percent": latency_improvement,
            "graph_accuracy_improvement_vs_rag_percent": accuracy_improvement,
            "graph_accuracy_point_delta_vs_rag": accuracy_delta,
            "graph_bertscore_improvement_vs_rag_percent": bert_improvement,
            "graph_bertscore_delta_vs_rag": bert_delta,
        }

    def _run_llm_only(
        self,
        query: str,
        reference_answer: str | None,
        *,
        provider: str | None = None,
    ) -> dict[str, Any]:
        prompt = f"""
Answer this medical question directly, without using retrieval, vector search, or graph context.

Question:
{query}
"""
        llm_provider = provider or settings.llm_only_provider
        generation = self.llm.generate(
            prompt,
            system=self._medical_system_prompt("No retrieved evidence is available."),
            provider=llm_provider,
            temperature=0.2,
            max_tokens=850,
        )
        return self._build_pipeline_result(
            key="llm_only",
            label="LLM Only",
            accent="red",
            generation=generation,
            query=query,
            context=prompt,
            evidence="",
            reference_answer=reference_answer,
            warning=f"Direct {llm_provider} generation with no retrieval or TigerGraph grounding.",
            extra={"evidence": []},
        )

    def _run_basic_rag(
        self,
        query: str,
        reference_answer: str | None,
        *,
        provider: str | None = None,
    ) -> dict[str, Any]:
        return self._run_basic_rag_with_k(
            query,
            reference_answer,
            provider=provider,
            top_k=settings.rag_retrieval_top_k,
        )

    def _run_basic_rag_with_k(
        self,
        query: str,
        reference_answer: str | None,
        *,
        provider: str | None = None,
        top_k: int,
    ) -> dict[str, Any]:
        retrievals = self.retriever.search(query, k=top_k)
        if not retrievals:
            raise RuntimeError("FAISS retrieval returned no chunks for the query.")
        context = self._retrieval_context(retrievals)
        prompt = f"""
Use only the retrieved medical chunks below to answer the question.
When evidence is incomplete, say what is missing instead of inventing facts.

Question:
{query}

Retrieved chunks ({len(retrievals)} vector matches, top-{top_k}):
{context}
"""
        evidence = "\n\n".join(result.content for result in retrievals)
        retrieved_payload = [self._retrieval_to_dict(result) for result in retrievals]
        llm_provider = provider or settings.rag_provider
        try:
            generation = self.llm.generate(
                prompt,
                system=self._medical_system_prompt("Ground the answer in the retrieved chunks."),
                provider=llm_provider,
                temperature=0.2,
                max_tokens=min(8192, 1200 + len(retrievals) * 120),
            )
        except Exception as exc:
            result = self._pipeline_error("basic_rag", "Basic RAG", "yellow", exc)
            result["warning"] = "FAISS retrieval succeeded, but hosted LLM generation failed."
            result["evidence"] = retrieved_payload
            result["retrieved_chunks"] = retrieved_payload
            return result
        return self._build_pipeline_result(
            key="basic_rag",
            label="Basic RAG",
            accent="yellow",
            generation=generation,
            query=query,
            context=prompt,
            evidence=evidence,
            reference_answer=reference_answer,
            warning=(
                f"SentenceTransformer + FAISS top-{top_k} retrieval + {llm_provider} generation."
            ),
            extra={
                "evidence": retrieved_payload,
                "retrieved_chunks": retrieved_payload,
            },
        )

    def _run_graphrag(
        self,
        query: str,
        reference_answer: str | None,
        *,
        provider: str | None = None,
    ) -> dict[str, Any]:
        if self._is_simple_query(query):
            result = self._run_basic_rag_with_k(
                query,
                reference_answer,
                provider=provider,
                top_k=settings.graphrag_retrieval_top_k,
            )
            result["label"] = "TigerGraph GraphRAG"
            result["accent"] = "cyan"
            result["warning"] = (
                "Simple factual query routed to compact FAISS RAG; TigerGraph traversal skipped."
            )
            result["graph"] = {
                **self._empty_graph(),
                "status": "Simple factual query routed to compact Basic RAG; graph traversal skipped.",
                "routing": "basic_rag_for_simple_query",
            }
            result["graph_context"] = "Simple factual query; graph traversal skipped."
            return result

        retrievals = self.retriever.search(query, k=settings.graphrag_retrieval_top_k)
        retrieved_payload = [self._retrieval_to_dict(result) for result in retrievals]
        try:
            graph = self.graph.reason(query, retrievals)
        except Exception as exc:
            result = self._pipeline_error("graphrag", "TigerGraph GraphRAG", "cyan", exc)
            result["warning"] = "FAISS retrieval ran, but TigerGraph traversal failed."
            result["evidence"] = retrieved_payload
            result["retrieved_chunks"] = retrieved_payload
            result["graph"]["status"] = str(exc)
            return result
        ranked_paths = self._rank_graph_paths(query, graph.get("edges") or [], retrievals)
        graph["ranked_paths"] = ranked_paths
        graph["paths"] = [path["path"] for path in ranked_paths]
        graph["edges"] = [
            {
                "source": path["source"],
                "target": path["target"],
                "relation": path["relation"],
                "final_score": path["final_score"],
            }
            for path in ranked_paths
        ]
        ranked_graph_chunks = self._rank_graph_chunks(query, graph.get("tg_chunks") or [])
        graph["tg_chunks_ranked"] = ranked_graph_chunks
        graph_context = self._graph_context(graph)
        prompt = f"""
Answer the medical question using only the compressed TigerGraph recommendations below.
If graph evidence is missing, state that limitation clearly.

Question:
{query}

Compressed GraphRAG context:
{graph_context}
"""
        evidence = graph_context
        llm_provider = provider or settings.graphrag_provider
        try:
            generation = self.llm.generate(
                prompt,
                system=self._medical_system_prompt("Ground the answer in TigerGraph paths and retrieved evidence."),
                provider=llm_provider,
                temperature=0.2,
                max_tokens=800,
            )
        except Exception as exc:
            result = self._pipeline_error("graphrag", "TigerGraph GraphRAG", "cyan", exc)
            result["warning"] = "TigerGraph traversal succeeded, but hosted LLM generation failed."
            result["graph"] = graph
            result["graph_context"] = graph_context
            result["evidence"] = retrieved_payload
            result["retrieved_chunks"] = retrieved_payload
            return result
        return self._build_pipeline_result(
            key="graphrag",
            label="TigerGraph GraphRAG",
            accent="cyan",
            generation=generation,
            query=query,
            context=prompt,
            evidence=evidence,
            reference_answer=reference_answer,
            warning=(
                f"TigerGraph direct-neighbor traversal + top "
                f"{settings.graphrag_graph_chunk_limit} ranked graph chunks + "
                f"{llm_provider} generation."
            ),
            extra={
                "graph": graph,
                "graph_context": graph_context,
                "evidence": ranked_paths or [self._graph_chunk_to_dict(chunk) for chunk in ranked_graph_chunks],
                "retrieved_chunks": ranked_paths or [self._graph_chunk_to_dict(chunk) for chunk in ranked_graph_chunks],
            },
        )

    def _build_pipeline_result(
        self,
        key: str,
        label: str,
        accent: str,
        generation: GenerationResult,
        query: str,
        context: str,
        evidence: str,
        reference_answer: str | None,
        warning: str,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        accuracy = self.evaluator.score(
            query=query,
            answer=generation.answer,
            evidence=evidence,
            pipeline=key,
            reference_answer=reference_answer,
        )
        result = {
            "label": label,
            "accent": accent,
            "answer": generation.answer,
            "warning": warning,
            "provider": generation.provider,
            "model": generation.model,
            "error": None,
            "metrics": self._metrics(generation, accuracy),
        }
        result.update(extra)
        return result

    def _generate_expected_answer(
        self, query: str, pipelines: dict[str, dict[str, Any]]
    ) -> tuple[str | None, str]:
        if not self.llm.has_provider("gemini"):
            fallback = self._extractive_expected_answer(query, pipelines)
            return fallback, "retrieved_context_fallback"

        context = self._reference_generation_context(query, pipelines)
        prompt = f"""
You are generating a reference medical answer for evaluation purposes only.
Given this medical query, return the ideal concise answer using the supplied
retrieved ACR/recommendation evidence as the primary source.

Medical query:
{query}

Retrieved ACR guideline chunks, medical recommendation dataset excerpts, graph evidence,
and medication-safety signals available to the benchmark:
{context}

Instructions:
- Return only the expected answer, not JSON.
- Be concise but medically complete enough for evaluation.
- Prefer specific clinical recommendations from the supplied evidence over generic medical knowledge.
- Include relevant tests, imaging procedures, and medication/contrast precautions when supported by evidence.
- Use cautious educational language.
- If the evidence is incomplete, say so briefly instead of filling gaps with broad generic advice.
"""
        try:
            generation = self.llm.generate(
                prompt,
                system=(
                    "You write concise reference medical answers for QA evaluation. "
                    "Ground the answer primarily in retrieved ACR guideline and medical "
                    "recommendation dataset context. Use established knowledge only to connect "
                    "evidence that is already present."
                ),
                provider="gemini",
                temperature=0.0,
                max_tokens=650,
            )
        except Exception as exc:  # pragma: no cover - external integration boundary
            fallback = self._extractive_expected_answer(query, pipelines)
            if fallback:
                return fallback, "retrieved_context_fallback"
            return None, self._friendly_external_error("expected_answer generation failed", exc)

        answer = generation.answer.strip()
        if not answer:
            fallback = self._extractive_expected_answer(query, pipelines)
            if fallback:
                return fallback, "retrieved_context_fallback"
            return None, "expected_answer generation returned an empty response."
        return answer, "gemini_auto_reference"

    def _extractive_expected_answer(
        self, query: str, pipelines: dict[str, dict[str, Any]]
    ) -> str | None:
        context = self._reference_generation_context(query, pipelines)
        context = context.split("TigerGraph evidence:", 1)[0]
        if not context.strip():
            return None
        salient = self._salient_reference_sentences(query, context)
        if not salient:
            return None
        return " ".join(salient[:5])

    @staticmethod
    def _salient_reference_sentences(query: str, context: str) -> list[str]:
        import re

        query_terms = set(PipelineService._query_tokens(query))
        priority_terms = {
            "acr",
            "appropriate",
            "recommended",
            "procedure",
            "imaging",
            "coronary",
            "angiography",
            "ecg",
            "troponin",
            "contrast",
            "renal",
            "kidney",
            "diabetes",
            "treatment",
            "precaution",
        }
        parts = [
            " ".join(part.split())
            for part in re.split(r"(?<=[.!?])\s+|\n+", context)
            if len(part.split()) >= 5 and "->" not in part and "usually not appropriate" not in part.lower()
        ]
        blocked_terms = {
            "bupivacaine",
            "hydrocodone",
            "lidocaine",
            "morphine",
            "opioid",
            "pain medication",
        }

        def score(sentence: str) -> tuple[int, int, int]:
            lowered = sentence.lower()
            tokens = set(PipelineService._query_tokens(lowered))
            appropriateness_bonus = 8 if "usually appropriate" in lowered else 0
            appropriateness_bonus += 3 if "may be appropriate" in lowered else 0
            return (
                len(tokens & query_terms) * 2
                + sum(1 for term in priority_terms if term in lowered),
                appropriateness_bonus,
                len(sentence),
            )

        ranked = sorted(parts, key=score, reverse=True)
        selected = []
        seen = set()
        for sentence in ranked:
            key = sentence.lower()
            if key in seen or any(term in key for term in blocked_terms):
                continue
            selected.append(sentence)
            seen.add(key)
            if len(selected) >= 5:
                break
        return selected

    @staticmethod
    def _friendly_external_error(prefix: str, exc: Exception) -> str:
        message = str(exc)
        if "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
            return f"{prefix}: Gemini quota/rate limit reached"
        return f"{prefix}: external LLM request failed"

    def _score_pipelines_against_expected(
        self,
        query: str,
        pipelines: dict[str, dict[str, Any]],
        expected_answer: str,
    ) -> None:
        for key, pipeline in pipelines.items():
            accuracy = self.evaluator.score(
                query=query,
                answer=pipeline.get("answer", ""),
                evidence=self._pipeline_evidence_text(key, pipeline),
                pipeline=key,
                reference_answer=expected_answer,
            )
            self._apply_accuracy(pipeline, accuracy)

    @staticmethod
    def _apply_accuracy(pipeline: dict[str, Any], accuracy: AccuracyResult) -> None:
        metrics = pipeline["metrics"]
        metrics["accuracy"] = accuracy.pass_rate
        metrics["bertscore_f1"] = accuracy.bertscore_f1
        metrics["judge"] = accuracy.judge
        metrics["judge_rationale"] = accuracy.rationale
        metrics["hallucination_risk"] = accuracy.hallucination_risk
        metrics["accuracy_method"] = accuracy.method

    def _reference_generation_context(
        self, query: str, pipelines: dict[str, dict[str, Any]]
    ) -> str:
        sections = []
        reference_retrievals = self.retriever.search(
            query, k=max(settings.rag_retrieval_top_k, 12)
        )
        reference_chunks = [
            self._retrieval_to_dict(result)
            for result in self._rank_reference_retrievals(query, reference_retrievals)
        ]
        if reference_chunks:
            sections.append("Priority ACR/recommendation chunks:")
            for chunk in reference_chunks[:8]:
                sections.append(
                    f"[{chunk.get('chunk_id', 'chunk')} | {chunk.get('source', '')} | "
                    f"{chunk.get('category', '')}]\n"
                    f"{self._compress(str(chunk.get('content') or chunk.get('preview') or ''), 1000)}"
                )

        basic_chunks = pipelines["basic_rag"].get("retrieved_chunks") or []
        if basic_chunks:
            sections.append("Basic RAG top chunks:")
            for chunk in basic_chunks[: settings.rag_retrieval_top_k]:
                sections.append(
                    f"[{chunk.get('chunk_id', 'chunk')}] "
                    f"{self._compress(str(chunk.get('content') or chunk.get('preview') or ''), 700)}"
                )

        graph = pipelines["graphrag"].get("graph") or {}
        graph_context = pipelines["graphrag"].get("graph_context") or ""
        paths = graph.get("paths") or []
        if paths or graph_context:
            sections.append("TigerGraph evidence:")
            sections.extend(str(path) for path in paths[:8])
            if graph_context:
                sections.append(self._compress(str(graph_context), 1600))

        if not sections:
            sections.append("No retrieved context was available; rely on established medical knowledge.")
        return "\n".join(sections)

    @staticmethod
    def _rank_reference_retrievals(
        query: str, retrievals: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        query_terms = set(PipelineService._query_tokens(query))
        recommendation_terms = {
            "acr",
            "appropriateness",
            "recommended",
            "recommendation",
            "procedure",
            "imaging",
            "contrast",
            "coronary",
            "troponin",
            "ecg",
            "egfr",
            "renal",
            "kidney",
            "diabetes",
        }

        def score(result: RetrievalResult) -> tuple[float, float]:
            text = f"{result.source} {result.category} {result.content}".lower()
            tokens = set(PipelineService._query_tokens(text))
            value = result.similarity * 8.0
            if str(result.source).lower() == "acr":
                value += 4.0
            value += len(tokens & query_terms) * 0.8
            value += sum(1.0 for term in recommendation_terms if term in text)
            return value, result.similarity

        return sorted(retrievals, key=score, reverse=True)

    @staticmethod
    def _query_tokens(text: str) -> list[str]:
        import re

        return [
            token.lower()
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text or "")
            if len(token) >= 3
        ]

    @staticmethod
    def _pipeline_evidence_text(key: str, pipeline: dict[str, Any]) -> str:
        if key == "graphrag":
            graph_context = pipeline.get("graph_context")
            if graph_context:
                return str(graph_context)
            graph = pipeline.get("graph") or {}
            return "\n".join(str(path) for path in graph.get("paths") or [])

        chunks = pipeline.get("retrieved_chunks") or pipeline.get("evidence") or []
        if isinstance(chunks, list):
            return "\n\n".join(
                str(chunk.get("content") or chunk.get("preview") or chunk)
                if isinstance(chunk, dict)
                else str(chunk)
                for chunk in chunks
            )
        return str(chunks or "")

    def _safe_timed(
        self,
        fn: Callable[[], dict[str, Any]],
        key: str,
        label: str,
        accent: str,
    ):
        try:
            return timed(fn)
        except Exception as exc:
            return timed(lambda: self._pipeline_error(key, label, accent, exc))

    def _error_result(self, key: str, error_msg: str) -> dict[str, Any]:
        """Create a timed error result for when a pipeline thread fails."""
        label_map = {
            "llm_only": "LLM Only",
            "basic_rag": "Basic RAG",
            "graphrag": "TigerGraph GraphRAG"
        }
        accent_map = {
            "llm_only": "red",
            "basic_rag": "yellow",
            "graphrag": "cyan"
        }
        exc = RuntimeError(error_msg)
        return timed(lambda: self._pipeline_error(
            key, 
            label_map.get(key, key),
            accent_map.get(key, "gray"),
            exc
        ))

    def _pipeline_error(
        self, key: str, label: str, accent: str, exc: Exception
    ) -> dict[str, Any]:
        graph = self._empty_graph() if key == "graphrag" else None
        result: dict[str, Any] = {
            "label": label,
            "accent": accent,
            "answer": "",
            "warning": "Pipeline did not run successfully. No answer was generated.",
            "error": str(exc),
            "provider": None,
            "model": None,
            "evidence": [],
            "retrieved_chunks": [],
            "metrics": {
                "tokens": 0,
                "latency_ms": 0,
                "cost_usd": 0.0,
                "accuracy": None,
                "bertscore_f1": None,
                "judge": "NOT_EVALUATED",
                "judge_rationale": str(exc),
                "hallucination_risk": "UNKNOWN",
                "accuracy_method": "pipeline failed",
                "token_source": "none",
                "token_reduction_vs_basic": None,
                "cost_reduction_vs_basic": None,
            },
        }
        if graph is not None:
            result["graph"] = graph
            result["graph_context"] = ""
        return result

    @staticmethod
    def _metrics(generation: GenerationResult, accuracy: AccuracyResult) -> dict[str, Any]:
        return {
            "tokens": generation.tokens_used,
            "latency_ms": 0,
            "cost_usd": generation.cost_usd,
            "accuracy": accuracy.pass_rate,
            "bertscore_f1": accuracy.bertscore_f1,
            "judge": accuracy.judge,
            "judge_rationale": accuracy.rationale,
            "hallucination_risk": accuracy.hallucination_risk,
            "accuracy_method": accuracy.method,
            "token_source": generation.token_source,
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "token_reduction_vs_basic": None,
            "cost_reduction_vs_basic": None,
        }

    @staticmethod
    def _attach_latency(result: dict[str, Any], measured_latency_ms: int) -> dict[str, Any]:
        result["metrics"]["latency_ms"] = measured_latency_ms
        return result

    @staticmethod
    def _attach_comparison_metrics(pipelines: dict[str, dict[str, Any]]) -> None:
        basic_tokens = pipelines["basic_rag"]["metrics"]["tokens"]
        graph_tokens = pipelines["graphrag"]["metrics"]["tokens"]
        basic_cost = pipelines["basic_rag"]["metrics"]["cost_usd"]
        graph_cost = pipelines["graphrag"]["metrics"]["cost_usd"]
        pipelines["graphrag"]["metrics"]["token_reduction_vs_basic"] = percent_reduction(
            basic_tokens, graph_tokens
        )
        pipelines["graphrag"]["metrics"]["cost_reduction_vs_basic"] = percent_reduction(
            basic_cost, graph_cost
        )

    @staticmethod
    def _comparison(pipelines: dict[str, dict[str, Any]]) -> dict[str, Any]:
        successful = {
            key: value
            for key, value in pipelines.items()
            if not value.get("error") and value["metrics"]["tokens"] > 0
        }
        winner = None
        if successful:
            winner = min(successful.items(), key=lambda row: row[1]["metrics"]["tokens"])[0]
        return {
            "token_reduction_percent": pipelines["graphrag"]["metrics"][
                "token_reduction_vs_basic"
            ],
            "cost_reduction_percent": pipelines["graphrag"]["metrics"][
                "cost_reduction_vs_basic"
            ],
            "winner": winner,
        }

    @staticmethod
    def _public_response(pipeline: dict[str, Any]) -> dict[str, Any]:
        metrics = pipeline["metrics"]
        payload = {
            "answer": pipeline["answer"],
            "tokens_used": metrics["tokens"],
            "latency": metrics["latency_ms"],
            "cost": metrics["cost_usd"],
            "error": pipeline.get("error"),
            "provider": pipeline.get("provider"),
            "model": pipeline.get("model"),
            "accuracy_score": metrics.get("accuracy"),
            "hallucination_risk": metrics.get("hallucination_risk"),
        }
        if pipeline.get("retrieved_chunks") is not None:
            payload["retrieved_chunks"] = pipeline.get("retrieved_chunks", [])
        if pipeline.get("graph") is not None:
            graph = pipeline["graph"]
            payload.update(
                {
                    "graph_nodes": graph.get("nodes", []),
                    "graph_edges": graph.get("edges", []),
                    "graph_context": pipeline.get("graph_context", ""),
                }
            )
        return payload

    @staticmethod
    def _retrieval_context(retrievals: list[RetrievalResult]) -> str:
        return "\n\n".join(
            f"[{result.chunk_id} | {result.source} | score={result.similarity}]\n{result.content}"
            for result in retrievals
        )

    def _graph_context(self, graph: dict[str, Any]) -> str:
        ranked_paths = graph.get("ranked_paths") or []
        if ranked_paths:
            lines = ["Top medical recommendations:"]
            for index, item in enumerate(ranked_paths[: settings.graphrag_graph_chunk_limit], start=1):
                procedure = self._short_procedure(str(item.get("target") or "recommended procedure"))
                relation = str(item.get("relation") or "usually appropriate")
                lines.append(f"{index}. {procedure} ({relation})")
            return "\n".join(lines)

        paths = graph.get("paths") or []
        tg_chunks = graph.get("tg_chunks_ranked") or []
        chunk_lines = []
        for chunk in tg_chunks[: settings.graphrag_graph_chunk_limit]:
            attrs = chunk.get("attributes", {})
            chunk_id = attrs.get("chunk_id") or chunk.get("chunk_id") or "graph_chunk"
            content = attrs.get("content") or chunk.get("content") or ""
            similarity = chunk.get("similarity")
            score = f" | similarity={similarity}" if similarity is not None else ""
            chunk_lines.append(f"[{chunk_id}{score}] {self._compress(content, 320)}")
            
        result_lines = []
        if paths:
            result_lines.extend(paths)
        if chunk_lines:
            result_lines.append("\nTigerGraph Chunks:")
            result_lines.extend(chunk_lines)
            
        if result_lines:
            return "\n".join(result_lines)
            
        entities = ", ".join(graph.get("entities") or [])
        status = graph.get("status") or "No graph path returned."
        return f"{status}\nMatched entities: {entities}"

    def _rank_graph_paths(
        self,
        query: str,
        edges: list[dict[str, Any]],
        retrievals: list[RetrievalResult],
    ) -> list[dict[str, Any]]:
        candidates = []
        retrieval_similarity = max((result.similarity for result in retrievals), default=0.0)
        for edge in edges:
            relation = str(edge.get("relation") or "")
            if not self._is_usually_appropriate(relation):
                continue
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if not target:
                continue
            content = f"{source} {relation} {target}"
            candidates.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "path": f"{source} -> {relation} -> {target}",
                    "content": content,
                    "graph_edge_weight": self._graph_edge_weight(relation),
                    "retrieval_similarity": retrieval_similarity,
                }
            )
        if not candidates:
            return []

        ranked_by_vector = self.retriever.rank_texts(
            query,
            candidates,
            text_key="content",
            limit=max(len(candidates), settings.graphrag_graph_chunk_limit),
        )
        ranked = []
        for item in ranked_by_vector:
            vector_score = float(item.get("similarity") or item.get("retrieval_similarity") or 0.0)
            graph_weight = float(item.get("graph_edge_weight") or 0.0)
            item["vector_similarity_score"] = round(vector_score, 4)
            item["final_score"] = round((0.6 * vector_score) + (0.4 * graph_weight), 4)
            ranked.append(item)
        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        return self._dedupe_graph_procedures(ranked, limit=settings.graphrag_graph_chunk_limit)

    @classmethod
    def _dedupe_graph_procedures(cls, ranked: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_names: list[str] = []
        for item in ranked:
            normalized = cls._procedure_key(str(item.get("target") or ""))
            if any(SequenceMatcher(None, normalized, seen).ratio() >= 0.72 for seen in seen_names):
                continue
            selected.append(item)
            seen_names.append(normalized)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _is_usually_appropriate(relation: str) -> bool:
        lowered = relation.lower()
        return (
            "usually appropriate" in lowered
            and "usually not appropriate" not in lowered
            and "not appropriate" not in lowered
            and "may be appropriate" not in lowered
        )

    @staticmethod
    def _graph_edge_weight(relation: str) -> float:
        return 1.0 if PipelineService._is_usually_appropriate(relation) else 0.0

    @staticmethod
    def _procedure_key(value: str) -> str:
        lowered = value.lower()
        if (
            "magnetic resonance" in lowered or "mri" in lowered
        ) and any(term in lowered for term in ["heart", "cardiac", "stress", "morphology", "function"]):
            return "mri cardiac"
        if (
            "computed tomography angiography" in lowered
            or "cta" in lowered
        ) and "coronary" in lowered:
            return "cta coronary"
        lowered = lowered.replace("computed tomography angiography", "cta")
        lowered = lowered.replace("computed tomography", "ct")
        lowered = lowered.replace("magnetic resonance imaging", "mri")
        lowered = lowered.replace("magnetic resonance angiography", "mra")
        lowered = lowered.replace("intravenous", "iv")
        tokens = [
            token
            for token in PipelineService._query_tokens(lowered)
            if token not in {"without", "with", "contrast", "function", "morphology", "heart"}
        ]
        return " ".join(tokens)

    @staticmethod
    def _short_procedure(value: str) -> str:
        clean = " ".join(value.split())
        replacements = {
            "Computed Tomography Angiography": "CTA",
            "Computed Tomography": "CT",
            "Magnetic Resonance Imaging": "MRI",
            "Magnetic Resonance Angiography": "MRA",
            "Intravenous": "IV",
        }
        for source, target in replacements.items():
            clean = clean.replace(source, target)
        return PipelineService._compress(clean, 96)

    @staticmethod
    def _is_simple_query(query: str) -> bool:
        lowered = query.lower().strip()
        multi_hop_terms = {
            "interaction",
            "contraindication",
            "because",
            "compare",
            "complex",
            "diabetes",
            "kidney",
            "renal",
            "multiple",
            "history",
            "precaution",
            "risk",
            "treatment",
            "workup",
            "pathway",
            "relationship",
        }
        if any(term in lowered for term in multi_hop_terms):
            return False
        if lowered.count("?") > 1:
            return False
        if len([token for token in lowered.split() if token]) > 12:
            return False
        simple_starts = ("what is", "define", "who is", "when is", "which test", "what test")
        return lowered.startswith(simple_starts)

    @staticmethod
    def _medical_system_prompt(grounding_instruction: str) -> str:
        return (
            "You are MedGraph AI, a medical research benchmark assistant. "
            "Give concise, clinically cautious educational answers. "
            "Do not diagnose or prescribe. Recommend urgent care for emergency symptoms. "
            f"{grounding_instruction}"
        )

    @staticmethod
    def _clean_query(query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        return query

    @staticmethod
    def _clean_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _compress(text: str, limit: int = 260) -> str:
        clean = " ".join(text.split())
        if len(clean) <= limit:
            return clean
        return clean[:limit].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _retrieval_to_dict(result: RetrievalResult) -> dict[str, Any]:
        return {
            "chunk_id": result.chunk_id,
            "source": result.source,
            "category": result.category,
            "content": result.content,
            "preview": PipelineService._compress(result.content, 260),
            "similarity": result.similarity,
            "entities": result.entities,
        }

    def _rank_graph_chunks(self, query: str, tg_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = [self._normalize_graph_chunk(chunk) for chunk in tg_chunks]
        return self.retriever.rank_texts(
            query,
            normalized,
            limit=settings.graphrag_graph_chunk_limit,
        )

    @staticmethod
    def _normalize_graph_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
        attrs = chunk.get("attributes") or {}
        content = attrs.get("content") or chunk.get("content") or attrs.get("text") or ""
        chunk_id = attrs.get("chunk_id") or chunk.get("chunk_id") or chunk.get("v_id") or "graph_chunk"
        source = attrs.get("source") or chunk.get("source") or "TigerGraph"
        category = attrs.get("category") or chunk.get("category") or "graph"
        return {
            **chunk,
            "attributes": attrs,
            "chunk_id": str(chunk_id),
            "source": str(source),
            "category": str(category),
            "content": str(content),
        }

    @staticmethod
    def _graph_chunk_to_dict(chunk: dict[str, Any]) -> dict[str, Any]:
        content = str(chunk.get("content") or "")
        return {
            "chunk_id": chunk.get("chunk_id") or "graph_chunk",
            "source": chunk.get("source") or "TigerGraph",
            "category": chunk.get("category") or "graph",
            "content": content,
            "preview": PipelineService._compress(content, 260),
            "similarity": chunk.get("similarity"),
            "entities": [],
        }

    @staticmethod
    def _empty_graph() -> dict[str, Any]:
        return {
            "nodes": [],
            "edges": [],
            "paths": [],
            "neighbors": {},
            "entities": [],
            "extracted_entities": [],
            "backend": "tigergraph",
            "status": "No TigerGraph traversal was completed.",
            "errors": [],
        }
