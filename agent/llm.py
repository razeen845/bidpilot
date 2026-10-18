"""Async OpenAI-compatible LLM client (vLLM on ROCm in production).

* semaphore (`llm.max_concurrency`), timeout, one retry;
* `chat_json(role, model, system, user, schema)` returns a dict validated against a pydantic model;
* JSON mode "schema" uses response_format=json_schema; if the server rejects it, the client
  falls back to JSON-only prompting for the rest of its life;
* every call is logged to out/llm_log.jsonl (role, model, tokens in/out, seconds).
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from settings import out_dir


class LLMError(RuntimeError):
    pass


def run_sync(coro):
    """Run a coroutine from sync code, even if an event loop is already running in this thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def target():
        try:
            box["v"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


def parse_json(text: str) -> Any:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


class LLMClient:
    def __init__(self, cfg: dict, out: str | Path | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg["llm"]
        self.enabled = bool(self.cfg.get("enabled", True))
        self.json_mode = self.cfg.get("json_mode", "schema")
        self.log_path = out_dir(out) / "llm_log.jsonl"
        self._transport = transport
        self._states: dict[int, tuple] = {}
        self._available: bool | None = None
        self._log_lock = threading.Lock()
        self.run_tag = ""     # free-form tag written to the log (bench uses it)

    # ---------- plumbing ----------
    def _sem(self) -> asyncio.Semaphore:
        """One concurrency limiter per event loop (each run_sync call has its own loop)."""
        key = id(asyncio.get_running_loop())
        if key not in self._states:
            self._states = {key: asyncio.Semaphore(int(self.cfg.get("max_concurrency", 32)))}
        return self._states[key]

    def _client(self):
        """A client used for one call and closed inside the same event loop."""
        from openai import AsyncOpenAI
        http_client = httpx.AsyncClient(transport=self._transport) if self._transport else None
        return AsyncOpenAI(base_url=self.cfg["base_url"], api_key=self.cfg.get("api_key") or "EMPTY",
                           timeout=float(self.cfg.get("timeout_s", 30)), max_retries=0, http_client=http_client)

    def _log(self, rec: dict) -> None:
        rec = {"ts": round(time.time(), 3), "tag": self.run_tag, **rec}
        with self._log_lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def available(self) -> bool:
        """Cheap health check (GET /models). Cached."""
        if not self.enabled:
            return False
        if self._available is None:
            if self._transport is not None:
                self._available = True
            else:
                try:
                    r = httpx.get(self.cfg["base_url"].rstrip("/") + "/models", timeout=3,
                                  headers={"Authorization": f"Bearer {self.cfg.get('api_key') or 'EMPTY'}"})
                    self._available = r.status_code == 200
                except httpx.HTTPError:
                    self._available = False
        return self._available

    def model_for(self, role: str) -> str:
        return self.cfg["model_planner"] if role in ("planner", "consultant", "day_report") else self.cfg["model_explainer"]

    # ---------- calls ----------
    async def chat_json(self, role: str, system: str, user: str, schema: type[BaseModel], model: str | None = None,
                        max_tokens: int = 700, temperature: float = 0.2) -> dict:
        model = model or self.model_for(role)
        last: Exception | None = None
        for attempt in range(2):          # one retry
            t0 = time.perf_counter()
            usage = None
            try:
                text, usage = await self._complete(model, system, user, schema, max_tokens, temperature)
                data = schema.model_validate(parse_json(text)).model_dump()
                self._log({"role": role, "model": model, "ok": True, "attempt": attempt, "json_mode": self.json_mode,
                           "tokens_in": getattr(usage, "prompt_tokens", 0) if usage else 0,
                           "tokens_out": getattr(usage, "completion_tokens", 0) if usage else 0,
                           "seconds": round(time.perf_counter() - t0, 3)})
                return data
            except Exception as e:  # noqa: BLE001
                last = e
                self._log({"role": role, "model": model, "ok": False, "attempt": attempt, "json_mode": self.json_mode,
                           "tokens_in": getattr(usage, "prompt_tokens", 0) if usage else 0,
                           "tokens_out": getattr(usage, "completion_tokens", 0) if usage else 0,
                           "seconds": round(time.perf_counter() - t0, 3), "error": f"{type(e).__name__}: {e}"[:300]})
                if self.json_mode == "schema" and _is_schema_rejection(e):
                    self.json_mode = "prompt"
        raise LLMError(f"{role}: {last}")

    async def _complete(self, model, system, user, schema, max_tokens, temperature):
        sem = self._sem()
        kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
        if self.json_mode == "schema":
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
            sys_msg = system
        else:
            sys_msg = (system + "\n\nRespond with a single JSON object only, no prose, matching this JSON schema:\n"
                       + json.dumps(schema.model_json_schema()))
        kwargs["messages"] = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
        async with sem, self._client() as client:
            resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content, resp.usage

    async def chat_text(self, role: str, system: str, user: str, model: str | None = None, max_tokens: int = 300,
                        temperature: float = 0.2) -> str:
        """Free-text call (used where JSON hurts: code writing). The caller validates the text."""
        model = model or self.model_for(role)
        sem = self._sem()
        t0 = time.perf_counter()
        async with sem, self._client() as client:
            resp = await client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=temperature,
                                                        messages=[{"role": "system", "content": system},
                                                                  {"role": "user", "content": user}])
        u = resp.usage
        self._log({"role": role, "model": model, "ok": True, "attempt": 0, "json_mode": "text",
                   "tokens_in": getattr(u, "prompt_tokens", 0), "tokens_out": getattr(u, "completion_tokens", 0),
                   "seconds": round(time.perf_counter() - t0, 3)})
        return resp.choices[0].message.content


def _is_schema_rejection(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    return status in (400, 422) and ("response_format" in str(e) or "json_schema" in str(e) or "guided" in str(e))


def make_llm(cfg: dict, out=None) -> LLMClient | None:
    """An LLM client if LLM is enabled AND reachable, else None (= --no-llm behaviour)."""
    if not cfg["llm"].get("enabled", True):
        return None
    llm = LLMClient(cfg, out)
    return llm if llm.available() else None
