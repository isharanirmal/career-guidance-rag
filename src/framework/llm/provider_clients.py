import os
import time
from typing import Any

from src.framework.interfaces.interfaces import ILLMClient


class ProviderClientError(RuntimeError):
    pass


class OpenAIClient(ILLMClient):
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.provider = "openai"
        self.web_search_enabled = False
        self.last_sources: list[dict] = []

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction or "You are a helpful career and academic guidance assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.35,
                max_tokens=1800,
            )
            text = response.choices[0].message.content if response.choices else None
            if not text:
                raise ProviderClientError("OpenAI returned an empty response.")
            return text.strip()
        except Exception as exc:
            raise ProviderClientError(f"OpenAI request failed: {type(exc).__name__}") from exc


class GroqClient(ILLMClient):
    def __init__(self, api_key: str):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.provider = "groq"
        self.web_search_enabled = False
        self.last_sources: list[dict] = []

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction or "You are a helpful career and academic guidance assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.35,
                max_tokens=1800,
            )
            text = response.choices[0].message.content if response.choices else None
            if not text:
                raise ProviderClientError("Groq returned an empty response.")
            return text.strip()
        except Exception as exc:
            raise ProviderClientError(f"Groq request failed: {type(exc).__name__}") from exc


class AnthropicClient(ILLMClient):
    def __init__(self, api_key: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.provider = "anthropic"
        self.web_search_enabled = False
        self.last_sources: list[dict] = []

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1800,
                temperature=0.35,
                system=system_instruction or "You are a helpful career and academic guidance assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            parts = getattr(response, "content", [])
            text = "".join(getattr(part, "text", "") for part in parts if getattr(part, "text", None))
            if not text:
                raise ProviderClientError("Anthropic returned an empty response.")
            return text.strip()
        except Exception as exc:
            raise ProviderClientError(f"Anthropic request failed: {type(exc).__name__}") from exc


class GeminiClient(ILLMClient):
    """Gemini with optional Google Search grounding."""

    def __init__(self, api_key: str | None = None):
        from google import genai
        from google.genai import types
        self._types = types
        key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not key:
            raise ValueError("GEMINI_API_KEY is required when Gemini is enabled.")
        self.client = genai.Client(api_key=key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.web_search_enabled = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.provider = "gemini"
        self.last_sources: list[dict] = []

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        tools = [self._types.Tool(google_search=self._types.GoogleSearch())] if self.web_search_enabled else []
        config = self._types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            tools=tools or None,
            temperature=0.35,
            max_output_tokens=1800,
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                text = getattr(response, "text", None)
                if not text:
                    raise ProviderClientError("Gemini returned an empty response.")
                self.last_sources = self._extract_sources(response)
                return text.strip()
            except Exception as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                if code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise ProviderClientError(f"Gemini request failed: {type(exc).__name__}") from exc
        raise ProviderClientError("Gemini request failed after retries.") from last_error

    @staticmethod
    def _extract_sources(response: Any) -> list[dict]:
        sources: list[dict] = []
        seen: set[str] = set()
        try:
            metadata = response.candidates[0].grounding_metadata
            for chunk in (getattr(metadata, "grounding_chunks", None) or []):
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None) if web else None
                title = getattr(web, "title", None) if web else None
                if uri and uri not in seen:
                    sources.append({"title": title or uri, "url": uri})
                    seen.add(uri)
        except (AttributeError, IndexError, TypeError):
            pass
        return sources
