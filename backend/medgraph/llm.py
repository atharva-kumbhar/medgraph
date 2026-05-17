"""Hosted LLM clients used by the benchmark pipelines."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

from .config import settings
from .metrics import estimate_cost, estimate_tokens


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    tokens_used: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float
    provider: str
    model: str
    token_source: str
    raw_usage: dict[str, Any]


class LLMClient:
    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        provider: str = "auto",
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> GenerationResult:
        resolved = self._resolve_provider(provider)
        if resolved == "gemini":
            return self._generate_gemini(
                prompt,
                system=system,
                model=model or settings.gemini_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if resolved == "nvidia":
            return self._generate_nvidia(
                prompt,
                system=system,
                model=model or settings.llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raise RuntimeError(f"Unsupported LLM provider: {resolved}")

    def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        provider: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> tuple[dict[str, Any], GenerationResult]:
        result = self.generate(
            prompt,
            system=system,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse_json_object(result.answer), result

    def has_provider(self, provider: str = "auto") -> bool:
        try:
            self._resolve_provider(provider)
        except RuntimeError:
            return False
        return True

    def _resolve_provider(self, provider: str) -> str:
        normalized = (provider or "auto").lower()
        if normalized == "auto":
            if settings.gemini_api_key:
                return "gemini"
            if settings.nvidia_api_key:
                return "nvidia"
            raise RuntimeError(
                "No hosted LLM is configured. Set GEMINI_API_KEY or NVIDIA_API_KEY."
            )
        if normalized == "gemini":
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required for this pipeline.")
            return "gemini"
        if normalized == "nvidia":
            if not settings.nvidia_api_key:
                raise RuntimeError("NVIDIA_API_KEY is required for this pipeline.")
            return "nvidia"
        raise RuntimeError(f"Unknown LLM provider '{provider}'.")

    def _generate_gemini(
        self,
        prompt: str,
        *,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        model_name = (model or settings.gemini_model).removeprefix("models/")
        endpoint = (
            f"{settings.gemini_base_url.rstrip('/')}/models/"
            f"{quote(model_name, safe='')}:generateContent"
        )
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        response = requests.post(
            endpoint,
            params={"key": settings.gemini_api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=settings.request_timeout_seconds,
        )
        self._raise_for_status(response)
        data = response.json()
        answer = self._gemini_text(data)
        usage = data.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")
        token_source = "api_usage" if total_tokens else "estimated"
        tokens = int(total_tokens or (estimate_tokens(prompt) + estimate_tokens(answer)))
        return GenerationResult(
            answer=answer.strip(),
            tokens_used=tokens,
            prompt_tokens=int(prompt_tokens) if prompt_tokens else None,
            completion_tokens=int(completion_tokens) if completion_tokens else None,
            cost_usd=estimate_cost(tokens, settings.price_per_1k_tokens_usd),
            provider="gemini",
            model=model_name,
            token_source=token_source,
            raw_usage=usage,
        )

    def _generate_nvidia(
        self,
        prompt: str,
        *,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        endpoint = f"{settings.nvidia_base_url.rstrip('/')}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {settings.nvidia_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.nvidia_request_timeout_seconds,
        )
        self._raise_for_status(response)
        data = response.json()
        choices = data.get("choices") or []
        answer = ""
        if choices:
            message = choices[0].get("message") or {}
            answer = message.get("content") or choices[0].get("text") or ""
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        token_source = "api_usage" if total_tokens else "estimated"
        tokens = int(total_tokens or (estimate_tokens(prompt) + estimate_tokens(answer)))
        return GenerationResult(
            answer=answer.strip(),
            tokens_used=tokens,
            prompt_tokens=int(prompt_tokens) if prompt_tokens else None,
            completion_tokens=int(completion_tokens) if completion_tokens else None,
            cost_usd=estimate_cost(tokens, settings.price_per_1k_tokens_usd),
            provider="nvidia",
            model=model,
            token_source=token_source,
            raw_usage=usage,
        )

    @staticmethod
    def _gemini_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        return "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("LLM judge did not return a JSON object.")
            return json.loads(match.group(0))

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.ok:
            return
        message = response.text[:800]
        try:
            payload = response.json()
            message = json.dumps(payload.get("error") or payload)[:800]
        except ValueError:
            pass
        raise RuntimeError(f"LLM API request failed with HTTP {response.status_code}: {message}")
