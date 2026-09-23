import os
from dotenv import load_dotenv
load_dotenv(override=True)
from src.framework.llm.llm_factory import build_llm_client

llm = build_llm_client()
print("Configured providers:", ", ".join(llm.provider_names))
try:
    answer = llm.generate("Reply with exactly: AI connection OK", "You are a connectivity test.")
    print("SUCCESS:", answer)
    print("Active provider:", llm.active_provider)
except Exception as exc:
    print("FAILED:", exc)
    raise SystemExit(1)
