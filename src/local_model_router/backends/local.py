from __future__ import annotations

import time

import httpx

from .base import BackendMetadata, GenerateResult


class LocalBackend:
    """OpenAI-compatible client for llama-server (llama.cpp)."""

    def __init__(self, name: str, base_url: str, timeout_seconds: float = 120.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.metadata = BackendMetadata()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def discover(self) -> None:
        """Query /v1/models and /props to populate backend metadata.

        Tolerates failure — if the server isn't up yet, metadata stays empty
        and we try again the next time discover() is called.
        """
        try:
            r = await self._client.get(f"{self.base_url}/v1/models", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                models = data.get("data") or []
                if models:
                    self.metadata.model_id = models[0].get("id")
        except httpx.HTTPError:
            pass

        try:
            r = await self._client.get(f"{self.base_url}/props", timeout=5.0)
            if r.status_code == 200:
                props = r.json()
                self.metadata.model_path = props.get("model_path") or props.get("model")
                n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
                if n_ctx is None:
                    n_ctx = props.get("n_ctx")
                self.metadata.context_size = n_ctx
                self.metadata.extra = {
                    k: v for k, v in props.items()
                    if k in ("chat_template", "build_info", "total_slots")
                }
        except httpx.HTTPError:
            pass

    async def generate(
        self,
        system: str | None,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerateResult:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start = time.perf_counter()
        r = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        r.raise_for_status()
        data = r.json()

        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}

        return GenerateResult(
            text=text,
            model=data.get("model") or self.metadata.model_id or "unknown",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            backend_name=self.name,
        )
