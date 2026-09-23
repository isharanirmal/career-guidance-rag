import os
from dotenv import load_dotenv

load_dotenv(override=True)

providers = {
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash"),
    "groq": ("GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-120b"),
}

for name, (key_name, model_name, default_model) in providers.items():
    key = os.getenv(key_name, "").strip()
    model = os.getenv(model_name, default_model).strip()
    print(f"{name:10} key={'CONFIGURED' if key else 'EMPTY':10} model={model}")

print(f"WEB_SEARCH_ENABLED={os.getenv('WEB_SEARCH_ENABLED', 'true')}")
