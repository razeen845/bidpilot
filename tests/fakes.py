"""Test doubles: a scripted LLM and a fake OpenAI-compatible HTTP server."""
from __future__ import annotations

import asyncio
import json

import httpx


class FakeLLM:
    """Same interface as agent.llm.LLMClient.chat_json; each role answered by a handler(system, user) -> dict."""

    def __init__(self, handlers: dict, delay: float = 0.0):
        self.handlers = handlers
        self.delay = delay
        self.calls: list[str] = []
        self.enabled = True
        self.in_flight = 0
        self.max_in_flight = 0

    def available(self) -> bool:
        return True

    async def _wait(self):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1

    async def chat_text(self, role, system, user, model=None, max_tokens=300, temperature=0.2):
        await self._wait()
        self.calls.append(role)
        out = self.handlers[role](system, user)
        if isinstance(out, Exception):
            raise out
        if isinstance(out, dict) and "code" in out:
            return "Here you go:\n```python\n" + out["code"] + "\n```"
        return out if isinstance(out, str) else json.dumps(out)

    async def chat_json(self, role, system, user, schema, model=None, max_tokens=700, temperature=0.2):
        await self._wait()
        self.calls.append(role)
        h = self.handlers.get(role)
        if h is None:
            raise RuntimeError(f"no fake handler for {role}")
        out = h(system, user)
        if isinstance(out, Exception):
            raise out
        return schema.model_validate(out).model_dump()


def completion(content: str, model: str = "fake") -> dict:
    return {"id": "x", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}


class FakeOpenAIServer:
    """httpx transport that behaves like /v1/chat/completions."""

    def __init__(self, reply, reject_schema: bool = False, delay: float = 0.0):
        self.reply = reply              # fn(request_json) -> content string
        self.reject_schema = reject_schema
        self.delay = delay
        self.requests: list[dict] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.reject_schema and "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return httpx.Response(200, json=completion(self.reply(body), body.get("model", "fake")))
        finally:
            self.in_flight -= 1

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
