"""Phase 0 check: one chat request + one JSON-schema request against the configured endpoint.

Run on/against the AMD box:
    vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000
    LLM_BASE_URL=http://<amd-box>:8000/v1 python scripts/check_llm.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import prompts as P  # noqa: E402
from agent.llm import LLMClient  # noqa: E402
from settings import load_config  # noqa: E402


async def main() -> int:
    cfg = load_config()
    llm = LLMClient(cfg)
    print(f"endpoint: {cfg['llm']['base_url']}  model: {cfg['llm']['model_planner']}")
    if not llm.available():
        print("FAIL: endpoint not reachable (GET /models). Start vLLM or set LLM_BASE_URL.")
        return 1
    t = time.perf_counter()
    text = await llm.chat_text("check", "You are concise.", "Say hello in five words.")
    print(f"chat ({time.perf_counter() - t:.2f}s): {text!r}")
    t = time.perf_counter()
    out = await llm.chat_json("check", P.EXPLAINER_SYSTEM,
                              P.explainer_user({"name": "cotton bedsheet", "entity_type": "keyword", "workflow": "bid_adjust",
                                                "action": "raise", "old_bid": 18.0, "new_bid": 21.0, "change_pct": 0.1667,
                                                "evidence": {"roas_7d": 4.1, "spend_7d": 2310.0}}), P.ExplainOut)
    print(f"json ({time.perf_counter() - t:.2f}s, mode={llm.json_mode}): {out}")
    print("OK - record the json mode above in NOTES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
