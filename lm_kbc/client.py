from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import requests

from .config import LMStudioConfig


_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
_env_file_loaded = False

# Transient conditions worth retrying: provider rate limits, upstream capacity
# problems, and gateway errors. A hosted endpoint returns these routinely over
# the ~7,500 calls in one validation run.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}

# Comfortably above any sane generation.concurrency setting.
_CONNECTION_POOL_SIZE = 64


def load_env_file(path: str | Path = _ENV_FILE) -> None:
    """Read KEY=VALUE lines from .env without overriding the real environment."""
    global _env_file_loaded
    if _env_file_loaded:
        return
    _env_file_loaded = True
    target = Path(path)
    if not target.is_file():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_api_key(config: LMStudioConfig) -> str:
    """Return the bearer token, preferring an environment variable over YAML.

    Keeping the secret out of the dataclass means it can never reach a run
    manifest, a trace file, or the repository.
    """
    if not config.api_key_env:
        return config.api_key
    load_env_file()
    value = os.environ.get(config.api_key_env, "").strip()
    if not value:
        raise ValueError(
            f"lm_studio.api_key_env names {config.api_key_env!r} but that "
            "variable is empty or unset; add it to .env or the environment"
        )
    return value


class LMStudioClient:
    """Client for any OpenAI-compatible chat-completions endpoint.

    Despite the name this also serves hosted gateways such as OpenRouter, which
    expose the same protocol; only `base_url`, `model`, and the API key differ.
    """

    def __init__(self, config: LMStudioConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {resolve_api_key(config)}",
                "Content-Type": "application/json",
            }
        )
        # urllib3 defaults to 10 pooled connections and blocks past that, which
        # would quietly serialize concurrent samples back into one stream.
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=_CONNECTION_POOL_SIZE,
            pool_maxsize=_CONNECTION_POOL_SIZE,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = response_format
        if self.config.reasoning is not None:
            # Hidden reasoning tokens ignore max_tokens and bill at the
            # completion rate, so they must be controlled explicitly.
            payload["reasoning"] = self.config.reasoning
        if self.config.provider_routing is not None:
            # Pins the upstream provider and quantization on gateways that route
            # one model name across several backends, so a run stays comparable.
            payload["provider"] = self.config.provider_routing

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        last_error = ""
        retry_after: float | None = None
        dropped: set[str] = set()
        for attempt in range(self.config.max_retries + 1):
            if attempt:
                time.sleep(self._backoff(attempt, retry_after=retry_after))
            attempted = {
                key: value for key, value in payload.items() if key not in dropped
            }
            try:
                body = self._request(url, attempted)
            except _TransientError as exc:
                last_error = str(exc)
                retry_after = exc.retry_after
                # Backends differ in what they accept: a JAX-served provider
                # rejects `seed` outright. Drop an unsupported optional
                # parameter and continue rather than failing the whole run over
                # reproducibility we cannot get across providers anyway.
                unsupported = _unsupported_parameter(last_error)
                if unsupported and unsupported in payload:
                    dropped.add(unsupported)
                    retry_after = None
                continue
            return self._content(body), body
        raise RuntimeError(
            f"chat completion failed after {self.config.max_retries + 1} "
            f"attempts: {last_error}"
        )

    def _request(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                url, json=payload, timeout=self.config.timeout_seconds
            )
        except requests.RequestException as exc:
            raise _TransientError(f"{type(exc).__name__}: {exc}") from exc

        if response.status_code in _RETRYABLE_STATUS:
            raise _TransientError(
                f"HTTP {response.status_code}: {response.text[:200]}",
                retry_after=_parse_retry_after(response.headers.get("Retry-After")),
            )
        response.raise_for_status()

        try:
            body = response.json()
        except ValueError as exc:
            raise _TransientError(f"non-JSON response body: {exc}") from exc
        if not isinstance(body, dict):
            raise _TransientError("response body was not a JSON object")

        # OpenRouter reports upstream failures as HTTP 200 with an `error` key
        # rather than as an error status, so a bare status check misses them.
        error = body.get("error")
        if error:
            message = (
                error.get("message") if isinstance(error, dict) else str(error)
            )
            code = error.get("code") if isinstance(error, dict) else None
            if code in _RETRYABLE_STATUS or code is None:
                raise _TransientError(f"gateway error: {message}")
            raise RuntimeError(f"gateway rejected the request: {message}")
        if not body.get("choices"):
            raise _TransientError("response contained no choices")
        return body

    @staticmethod
    def _content(body: dict[str, Any]) -> str:
        message = body["choices"][0].get("message") or {}
        # Some local and hosted reasoning models place the complete response,
        # including its final JSON, in reasoning_content and leave content empty.
        content = message.get("content") or message.get("reasoning_content") or ""
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("chat completion returned no usable response text")
        return content

    def _backoff(self, attempt: int, *, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, 60.0)
        base = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
        return min(base, 60.0) * (0.5 + random.random() / 2)


class _TransientError(RuntimeError):
    """A failure worth retrying rather than surfacing to the pipeline.

    The server's requested delay travels with the exception rather than on the
    client, which is shared across threads when samples run concurrently.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


_OPTIONAL_PARAMETERS = ("seed", "response_format", "top_p")


def _unsupported_parameter(message: str) -> str | None:
    """Name the optional parameter an upstream provider says it cannot honour."""
    lowered = message.casefold()
    if "support" not in lowered:
        return None
    return next((name for name in _OPTIONAL_PARAMETERS if name in lowered), None)


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return max(float(value), 0.0) if value else None
    except (TypeError, ValueError):
        return None
