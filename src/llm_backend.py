"""
llm_backend.py
---------------
Keeps this project 100% free to run. Both supported backends speak the same
OpenAI-compatible /chat/completions "tools" API, so this is one thin wrapper
that just changes base_url/model depending on env vars:

  LLM_PROVIDER=groq    (default)  -> free cloud API, no credit card required
      Sign up at https://console.groq.com, create an API key, set:
        GROQ_API_KEY=...
      Free tier is generous (dozens of requests/min) and plenty for this
      project. Model defaults to openai/gpt-oss-20b, which has strong tool
      tool calling well.

  LLM_PROVIDER=ollama              -> fully local, zero signup, zero API key
      Install Ollama (https://ollama.com), then:
        ollama pull qwen2.5:7b
        ollama serve
      Nothing leaves your machine, no account needed anywhere.

Both are genuinely free -- no trial credits, no card-on-file.
"""

from __future__ import annotations
import os

DEFAULT_MODELS = {
    # Groq deprecated llama-3.1-8b-instant and llama-3.3-70b-versatile,
    # shutting them down entirely on August 16, 2026 -- if you see a
    # "model does not exist" 404 for either of those names, that's why.
    # openai/gpt-oss-20b is Groq's own recommended replacement: strong tool-
    # calling support, fast, and on the free tier. If Groq retires this one
    # too in the future, check https://console.groq.com/docs/models for the
    # current list and override via LLM_MODEL in .env rather than editing
    # this file.
    "groq": "openai/gpt-oss-20b",
    "ollama": "qwen2.5:7b",
}


def get_client_and_model():
    """Lazily imports openai so the rest of the codebase (cleaning, KPIs,
    tools, and the agent's own tool-dispatch/retry logic) can be imported
    and unit-tested without the openai package installed or any network
    access -- only actually asking a question needs it."""
    from openai import OpenAI

    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
                "and put it in your .env file, or set LLM_PROVIDER=ollama to run "
                "fully offline instead."
            )
        # max_retries=0: the openai SDK's default behavior is to silently
        # sleep-and-retry on 429/5xx errors *inside* the client.create() call
        # before ever raising anything back to our code -- which means
        # agent.py's own rate-limit handling (fail fast with a clear message)
        # never gets a chance to run; instead the whole CLI appears to hang
        # for minutes. Disabling SDK-level retries makes errors surface
        # immediately so our own explicit handling in agent.py is what
        # actually decides what to do.
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", max_retries=0)
        model = os.getenv("LLM_MODEL", DEFAULT_MODELS["groq"])
        return client, model

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        client = OpenAI(api_key="ollama", base_url=base_url, max_retries=0)  # api_key is unused but required by the SDK
        model = os.getenv("LLM_MODEL", DEFAULT_MODELS["ollama"])
        return client, model

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' or 'ollama'.")
