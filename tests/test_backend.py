"""LocalBackend tests — mocked HTTP via respx."""
from __future__ import annotations

import httpx
import pytest
import respx

from local_model_router.backends.local import LocalBackend


async def test_generate_parses_openai_response():
    async with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "qwen2.5-14b-instruct-Q4_K_M",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Hi there."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                },
            )
        )
        b = LocalBackend("local", "http://localhost:8080")
        result = await b.generate(system="be brief", user="hello", max_tokens=10)
        assert result.text == "Hi there."
        assert result.model == "qwen2.5-14b-instruct-Q4_K_M"
        assert result.prompt_tokens == 12
        assert result.completion_tokens == 4
        assert result.finish_reason == "stop"
        assert result.backend_name == "local"
        assert result.latency_ms >= 0
        await b.aclose()


async def test_generate_handles_missing_usage_field():
    async with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "test",
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "stop"}
                    ],
                },
            )
        )
        b = LocalBackend("local", "http://localhost:8080")
        result = await b.generate(system=None, user="hi")
        assert result.text == "ok"
        assert result.prompt_tokens is None
        assert result.completion_tokens is None
        await b.aclose()


async def test_discover_populates_metadata_from_models_endpoint():
    async with respx.mock(base_url="http://localhost:8080") as mock:
        mock.get("/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "qwen2.5-14b-Q4_K_M"}]},
            )
        )
        mock.get("/props").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model_path": "/path/to/qwen2.5-14b-Q4_K_M.gguf",
                    "default_generation_settings": {"n_ctx": 8192},
                },
            )
        )
        b = LocalBackend("local", "http://localhost:8080")
        await b.discover()
        assert b.metadata.model_id == "qwen2.5-14b-Q4_K_M"
        assert b.metadata.model_path.endswith(".gguf")
        assert b.metadata.context_size == 8192
        await b.aclose()


async def test_discover_tolerates_server_down():
    async with respx.mock(base_url="http://localhost:8080") as mock:
        mock.get("/v1/models").mock(side_effect=httpx.ConnectError("nope"))
        mock.get("/props").mock(side_effect=httpx.ConnectError("nope"))
        b = LocalBackend("local", "http://localhost:8080")
        await b.discover()  # must not raise
        assert b.metadata.model_id is None
        assert b.metadata.context_size is None
        await b.aclose()


async def test_generate_raises_on_http_error():
    async with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "boom"}),
        )
        b = LocalBackend("local", "http://localhost:8080")
        with pytest.raises(httpx.HTTPStatusError):
            await b.generate(system=None, user="hi")
        await b.aclose()


async def test_messages_include_system_when_provided():
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "t",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    async with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=capture)
        b = LocalBackend("local", "http://localhost:8080")
        await b.generate(system="sys text", user="user text", max_tokens=5)
        msgs = captured["payload"]["messages"]
        assert msgs[0] == {"role": "system", "content": "sys text"}
        assert msgs[1] == {"role": "user", "content": "user text"}
        assert captured["payload"]["max_tokens"] == 5
        await b.aclose()
