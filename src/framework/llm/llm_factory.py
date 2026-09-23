import os
from collections.abc import Callable

from src.framework.interfaces.interfaces import ILLMClient
from src.framework.llm.provider_clients import AnthropicClient, GeminiClient, GroqClient, OpenAIClient, ProviderClientError


def _key(name: str) -> str:
    return os.getenv(name, "").strip()


def build_llm_client() -> ILLMClient:
    """Build a multi-provider client from configured API keys.

    No provider is required to be configured at import time; the app reports a
    clear setup error until at least one key is supplied. Providers are tried
    in the configured fallback order, skipping providers without keys.
    """
    factories: dict[str, Callable[[], ILLMClient]] = {
        "openai": lambda: OpenAIClient(_key("OPENAI_API_KEY")),
        "anthropic": lambda: AnthropicClient(_key("ANTHROPIC_API_KEY")),
        "claude": lambda: AnthropicClient(_key("CLAUDE_API_KEY")),
        "gemini": lambda: GeminiClient(_key("GEMINI_API_KEY")),
        "groq": lambda: GroqClient(_key("GROQ_API_KEY")),
    }

    requested = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    order = [x.strip().lower() for x in os.getenv(
        "LLM_FALLBACK_ORDER", "openai,anthropic,gemini,groq"
    ).split(",") if x.strip()]
    if requested != "auto":
        order = [requested] + [x for x in order if x != requested]

    available: list[tuple[str, Callable[[], ILLMClient]]] = []
    env_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "claude": "CLAUDE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    for provider in order:
        if provider in factories and _key(env_keys[provider]):
            available.append((provider, factories[provider]))

    # Support a sensible default even if a user only set a key but forgot to
    # update LLM_FALLBACK_ORDER.
    for provider, env_name in env_keys.items():
        if _key(env_name) and provider in factories and all(provider != p for p, _ in available):
            available.append((provider, factories[provider]))

    if not available:
        raise ValueError(
            "No AI provider API key is configured. Add at least one of "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, "
            "or CLAUDE_API_KEY to .env."
        )

    return MultiProviderClient(available)


class MultiProviderClient(ILLMClient):
    def __init__(self, providers: list[tuple[str, Callable[[], ILLMClient]]]):
        self._providers = providers
        self.provider_names = [name for name, _ in providers]
        self.active_provider: str | None = None
        self.web_search_enabled = any(name == "gemini" for name, _ in providers) and os.getenv("WEB_SEARCH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.last_sources: list[dict] = []

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        errors: list[str] = []
        for name, factory in self._providers:
            try:
                client = factory()
                text = client.generate(prompt, system_instruction)
                self.active_provider = name
                self.web_search_enabled = getattr(client, "web_search_enabled", False)
                self.last_sources = getattr(client, "last_sources", []) or []
                return text
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"[AI PROVIDER ERROR] {name}: {type(exc).__name__}: {exc}")
                continue
        self.active_provider = None
        self.last_sources = []
        raise ProviderClientError("All configured AI providers failed: " + ", ".join(errors))
