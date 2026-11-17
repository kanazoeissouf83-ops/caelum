"""Providers known to CAELUM, including local OpenAI-compatible servers.

Cloud providers (``openai``, ``anthropic``) are reached through their own SDK. Local
backends (Ollama, llama.cpp, vLLM, LM Studio) expose an OpenAI-compatible API on your
machine: CAELUM reaches them through the ``openai`` SDK at their default address, so a
local model can serve as a free routing target or as a last-resort fallback tier::

    client = OpenAI(fallback=["ollama:qwen3:1.7b"])   # cloud first, local if it fails

Each local server's address can be changed with an environment variable
(``CAELUM_OLLAMA_BASE_URL``, ``CAELUM_LLAMACPP_BASE_URL``, ``CAELUM_VLLM_BASE_URL``,
``CAELUM_LMSTUDIO_BASE_URL``; Ollama also honours its own ``OLLAMA_HOST``) or by passing
a ready-made client in ``provider_clients``.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

__all__ = [
    "CLOUD_PROVIDERS",
    "LOCAL_PROVIDERS",
    "SUPPORTED_PROVIDERS",
    "detect_provider",
    "dialect",
    "is_local",
    "local_base_url",
]

CLOUD_PROVIDERS = frozenset({"openai", "anthropic"})

#: Local backend -> default OpenAI-compatible base URL.
LOCAL_PROVIDERS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "llamacpp": "http://localhost:8080/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
}

SUPPORTED_PROVIDERS = CLOUD_PROVIDERS | frozenset(LOCAL_PROVIDERS)

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})  # noqa: S104
_PORT_TO_LOCAL = {
    urlsplit(url).port: name for name, url in LOCAL_PROVIDERS.items() if urlsplit(url).port
}


def is_local(provider: str) -> bool:
    """True for a local backend (no API key, no per-token cost).

    ``"local"`` is an unidentified server on this machine (see :func:`detect_provider`).
    """
    return provider in LOCAL_PROVIDERS or provider == "local"


def dialect(provider: str) -> str:
    """API format spoken by ``provider``: ``"openai"`` or ``"anthropic"``."""
    return "openai" if is_local(provider) else provider


def local_base_url(provider: str) -> str:
    """OpenAI-compatible base URL of a local backend (environment overrides first)."""
    override = os.environ.get(f"CAELUM_{provider.upper()}_BASE_URL")
    if override:
        return override
    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST")
        if host:
            # OLLAMA_HOST is "host:port" or a URL without the /v1 suffix.
            base = host if "://" in host else f"http://{host}"
            if base.startswith("http://0.0.0.0"):
                base = base.replace("0.0.0.0", "localhost", 1)  # noqa: S104 - listen-all, not a target
            return base.rstrip("/") + "/v1"
    return LOCAL_PROVIDERS[provider]


def detect_provider(base_url: object, default: str = "openai") -> str:
    """Provider actually behind an OpenAI-SDK client, guessed from its base URL.

    A client pointed at the machine itself (``localhost``) is a local backend: the port
    names it (11434 = Ollama, 8080 = llama.cpp, 8000 = vLLM, 1234 = LM Studio), otherwise
    it is reported as ``"local"``. Any other address keeps ``default``.
    """
    try:
        parts = urlsplit(str(base_url))
        host, port = parts.hostname, parts.port
    except ValueError:
        return default
    if host not in _LOOPBACK_HOSTS:
        return default
    return _PORT_TO_LOCAL.get(port, "local")
