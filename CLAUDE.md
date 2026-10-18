# BidPilot — Claude Code project rules

Hackathon project with a 2.5-day deadline. The full plan is in PLAN.md. Read it before any work.

## Always
- Work on ONE phase at a time, the one the user names. Stop when its "Done when" checks pass.
- Add pytest tests for each module you write. Run `pytest` before saying a phase is done.
- Keep code simple: plain Python, pandas/numpy/scipy, Streamlit. No agent frameworks, no database, no auth.
- Seed all randomness from config.yaml. Same seed must give same results.
- Every number or fact shown to the user must carry a citation: `[D#]` for sandbox analysis results, `[W#]` for verified web sources. Run the citation check before showing anything.

## Never
- Never call real ad platform APIs (Google Ads, Meta, Daraz). Use `sim/platform.py` only.
- Never let the LLM produce bid numbers. Bids come from `engine/optimizer.py`; consultant actions are bounded and pass through guardrails.
- Never `exec`/`eval` LLM-written code in the main process. It only runs through `sandbox/exec_runner.py` after `sandbox/policy.py` approves it.
- Never import `sim/market.py` or read `out/_hidden_market.json` from `agent/`, `engine/`, `analysis/`, or `sandbox/`. It is the hidden truth.
- Never keep a web "fact" whose quote wasn't found in the retrieved page text.
- Never make the demo depend on live internet. Research must work in `cache` mode.
- Never add features that are not in PLAN.md. Write ideas in NOTES.md instead.
- Never make bidding depend on the LLM. `--no-llm` mode must always work.

## LLM
- OpenAI-compatible client; `base_url` and model names come from config.yaml (vLLM on an AMD GPU).
- All LLM outputs are JSON matching the schemas in `agent/prompts.py`.
