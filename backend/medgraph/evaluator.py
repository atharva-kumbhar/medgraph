"""BERTScore and LLM-as-judge evaluation hooks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .config import settings
from .llm import LLMClient


@dataclass(frozen=True)
class AccuracyResult:
    judge: str
    pass_rate: float | None
    bertscore_f1: float | None
    hallucination_risk: str
    method: str
    rationale: str


class AccuracyEvaluator:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def score(
        self,
        query: str,
        answer: str,
        evidence: str,
        pipeline: str,
        reference_answer: str | None = None,
    ) -> AccuracyResult:
        if not answer:
            return AccuracyResult(
                judge="NOT_EVALUATED",
                pass_rate=None,
                bertscore_f1=None,
                hallucination_risk="UNKNOWN",
                method="no answer generated",
                rationale="The pipeline did not produce an answer.",
            )

        bertscore_f1 = self._bertscore(answer, reference_answer) if reference_answer else None
        if not reference_answer:
            return AccuracyResult(
                judge="NOT_EVALUATED",
                pass_rate=None,
                bertscore_f1=bertscore_f1,
                hallucination_risk="UNKNOWN",
                method="requires expected answer",
                rationale="Generate or provide expected_answer/reference_answer to compute LLM judge accuracy and BERTScore.",
            )
        if not settings.enable_llm_judge:
            return AccuracyResult(
                judge="NOT_EVALUATED",
                pass_rate=None,
                bertscore_f1=bertscore_f1,
                hallucination_risk="UNKNOWN",
                method="LLM-as-judge disabled",
                rationale="Set MEDGRAPH_ENABLE_LLM_JUDGE=true to evaluate generated answers.",
            )
        if not self.llm_client or not self.llm_client.has_provider("gemini"):
            return AccuracyResult(
                judge="NOT_EVALUATED",
                pass_rate=None,
                bertscore_f1=bertscore_f1,
                hallucination_risk="UNKNOWN",
                method="requires hosted LLM judge",
                rationale="Set GEMINI_API_KEY for Gemini LLM-as-a-judge scoring.",
            )

        prompt = self._judge_prompt(
            query=query,
            answer=answer,
            evidence=evidence,
            pipeline=pipeline,
            reference_answer=reference_answer,
            bertscore_f1=bertscore_f1,
        )
        try:
            generation = self.llm_client.generate(
                prompt,
                system=(
                    "You are a strict medical QA evaluator. Return exactly PASS or FAIL "
                    "and no other text."
                ),
                provider="gemini",
                temperature=0.0,
                max_tokens=32,
            )
        except Exception as exc:  # pragma: no cover - external integration boundary
            fallback = self._fallback_judge(bertscore_f1)
            return AccuracyResult(
                judge=fallback or "NOT_EVALUATED",
                pass_rate=100.0 if fallback == "PASS" else 0.0 if fallback == "FAIL" else None,
                bertscore_f1=bertscore_f1,
                hallucination_risk="UNKNOWN",
                method="LLM-as-judge failed" + ("; semantic fallback" if fallback else ""),
                rationale=self._friendly_external_error(exc),
            )

        judge = self._parse_pass_fail(generation.answer)
        if not judge:
            judge = self._fallback_judge(bertscore_f1)
        accuracy = 100.0 if judge == "PASS" else 0.0 if judge == "FAIL" else None
        return AccuracyResult(
            judge=judge or "NOT_EVALUATED",
            pass_rate=accuracy,
            bertscore_f1=bertscore_f1,
            hallucination_risk="UNKNOWN",
            method=(
                "LLM-as-a-judge"
                + (" + BERTScore" if bertscore_f1 is not None else "")
                + ("; semantic fallback" if judge and self._parse_pass_fail(generation.answer) is None else "")
            ),
            rationale=(
                f"Gemini judge result: {judge}"
                if self._parse_pass_fail(generation.answer)
                else f"Gemini judge raw response did not contain PASS/FAIL: {generation.answer[:160]}"
            ),
        )

    @staticmethod
    def _judge_prompt(
        *,
        query: str,
        answer: str,
        evidence: str,
        pipeline: str,
        reference_answer: str | None,
        bertscore_f1: float | None,
    ) -> str:
        reference_block = reference_answer or ""
        bert_block = (
            f"BERTScore F1 against reference: {bertscore_f1}"
            if bertscore_f1 is not None
            else "BERTScore F1 unavailable."
        )
        evidence_block = evidence or "No retrieved evidence was provided by this pipeline."
        return f"""
Evaluate the generated answer for a medical QA system.

Pipeline: {pipeline}
User query:
{query}

Expected medical answer:
{reference_block}

Generated answer to evaluate:
{answer}

{bert_block}

Retrieved or graph evidence:
{evidence_block}

Instructions:
Compare expected answer vs generated answer.
Return PASS if the generated answer is medically consistent with the expected answer.
Return FAIL if it misses or contradicts important clinical recommendations.
Your entire response must be exactly one uppercase word: PASS or FAIL.
"""

    @staticmethod
    def _bertscore(answer: str, reference_answer: str | None) -> float | None:
        if not reference_answer:
            return None
        try:
            from bert_score import score as bert_score
        except ImportError:
            return AccuracyEvaluator._semantic_f1(answer, reference_answer)
        _, _, f1 = bert_score([answer], [reference_answer], lang="en", verbose=False)
        return round(float(f1.mean().item()), 4)

    @staticmethod
    def _semantic_f1(answer: str, reference_answer: str | None) -> float | None:
        """Token-level semantic fallback used only when bert-score is not installed."""
        if not reference_answer:
            return None
        answer_terms = AccuracyEvaluator._content_terms(answer)
        reference_terms = AccuracyEvaluator._content_terms(reference_answer)
        if not answer_terms or not reference_terms:
            return None
        overlap = answer_terms & reference_terms
        precision = len(overlap) / len(answer_terms)
        recall = len(overlap) / len(reference_terms)
        if math.isclose(precision + recall, 0.0):
            return 0.0
        return round((2 * precision * recall) / (precision + recall), 4)

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        stopwords = {
            "about",
            "after",
            "and",
            "are",
            "because",
            "before",
            "consider",
            "for",
            "from",
            "has",
            "have",
            "into",
            "not",
            "patient",
            "patients",
            "should",
            "that",
            "the",
            "their",
            "this",
            "with",
            "would",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", (text or "").lower())
            if len(token) >= 3 and token not in stopwords
        }

    @staticmethod
    def _number_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return round(float(value), 1)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_score_0_to_100(text: str) -> float | None:
        import re

        match = re.search(r"\d+(?:\.\d+)?", text or "")
        if not match:
            return None
        value = float(match.group(0))
        return round(max(0.0, min(value, 100.0)), 1)

    @staticmethod
    def _parse_pass_fail(text: str) -> str | None:
        normalized = (text or "").strip().upper()
        if re.search(r"\bFAIL\b", normalized):
            return "FAIL"
        if re.search(r"\bPASS\b", normalized):
            return "PASS"
        return None

    @staticmethod
    def _fallback_judge(bertscore_f1: float | None) -> str | None:
        if bertscore_f1 is None:
            return None
        return "PASS" if bertscore_f1 >= 0.42 else "FAIL"

    @staticmethod
    def _friendly_external_error(exc: Exception) -> str:
        message = str(exc)
        if "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
            return "Gemini quota/rate limit reached; used semantic fallback where possible."
        return "Hosted judge request failed; used semantic fallback where possible."
