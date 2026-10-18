# NOTES

Environment facts, decisions and ideas that are not in PLAN.md.

## Environment (dev machine, 2026-10-15)
- Windows 11, Python 3.12 venv (`.venv`, created with `uv`). PLAN says 3.11; 3.12 works with every dependency.
- **Docker: not installed** -> `sandbox.backend: subprocess`. On Windows `resource.setrlimit` is unavailable, so the
  subprocess sandbox enforces the wall-clock timeout but not a memory cap; on Linux it also sets CPU + memory limits.
- **No AMD GPU / no LLM endpoint reachable from the dev machine** (nothing on `localhost:8000`). All LLM code is tested
  against a fake OpenAI-compatible server (httpx MockTransport) and a scripted fake client. Point `LLM_BASE_URL` at the
  vLLM-on-ROCm box to run for real. Model picked for all roles: `Qwen/Qwen2.5-32B-Instruct` (fits easily on one MI300X).
- JSON output: `llm.json_mode: schema` uses `response_format={"type":"json_schema"}` (supported by recent vLLM).
  If the server rejects it (HTTP 400) the client switches itself to JSON-only prompting + `json.loads` with one retry.
  **Still to record:** which mode the AMD box actually used (`python scripts/check_llm.py` prints it).
- **No search API key** in the environment -> research runs in `cache` mode from a hand-curated cache (see Phase 7 below).

## Decisions
- Day numbering: history days are -29..0, simulated days are 1..14 (`out/day_{n}_plan.json` uses n = sim day).
- `stock.parquet` carries `units_sold` and `restocked_units` in addition to `stock_units` so days-of-stock and
  stockout ad spend can be computed without the hidden market.
- Stockout ad spend = ad cost on entities whose product had 0 units at the start of the day.
- Paused entities are recorded with `bid = 0` in `daily_stats` (so curve fits ignore them).
- The optimizer cannot always meet the budget within one day's +-20% range; the platform still enforces the budget
  hard by scaling clicks.
- Waste cleanup ignores products that were out of stock in the last 7 days (no sales there is a stock problem,
  not a keyword problem).
- The planner may hold any change except stock-guard changes (safety first).

## Phase 6-11 notes
- Sandbox: `subprocess` backend with `python -I`, near-empty env, restricted builtins and an import hook that
  only allows pandas/numpy/math/statistics; policy check runs before and the sandbox refuses hidden-market
  columns. Docker backend implemented (`sandbox/Dockerfile`) but untested (no Docker here).
- Policy also bans `.query`, `.eval`, `.pipe` and any string containing `__` (pandas evaluates query strings).
- Without an LLM, "Ask the data" answers the 4 preset questions using reviewed reference code
  (`agent/analyst.py: PRESET_CODE`); the D source records `code_author: preset`. With an LLM, the preset code is
  used only if the LLM's code fails twice (`code_author: preset (llm code failed)`).
- Research: no search API key -> hand-collected cache (PLAN Phase 7 fallback), built 2026-10-15 by
  `scripts/build_research_cache.py`: 9/9 quotes verified against live page text. 4 other candidate pages failed
  DNS from this network (optmyzr, yaelconsulting, furnituretoday, morpheabed) and were skipped.
- Consultant check covers finding + expected_impact (as in PLAN). Bid actions are clamped to max 20% and an
  empty filter turns a bid_action into an owner to-do. Unverified recommendations can only become to-dos.
- Dashboard "Fast-forward N days" just loops run_agent_day(auto_approve=True); it is how to reach day 7 live.
- bench/latency.py `--dry-run` uses a fake in-process server so the script can be tested; it never writes
  `bench/results.json`.

## Still needs the AMD box (cannot be done from this machine)
1. `python scripts/check_llm.py` -> record JSON mode in this file.
2. `python -m sim.runner --compare --llm` -> confirm `agent` >= `rules` and one agent day < 60 s.
3. `python scripts/prepare_demo.py` -> day-7 consultant report with >= 3 verified recommendations.
4. `GPU_NAME=... python bench/latency.py` -> real speed numbers in `bench/results.json`.
5. Optional: set `SEARCH_API_KEY`, `research.mode: live`, record one live run, then switch back to `cache`.

## Phase 12 notes (practitioner upgrades, 2026-10-17)
- Waste rule now: 14-day window, newest day's clicks ignored (lag), pause only if (1 - category_cvr)^clicks < 5%.
  Result on seed 42: 6 of 18 junk keywords paused outright, the other 12 pushed to small spend by the optimizer;
  one real keyword (K005) was paused after an unlucky streak - the 95% rule allows ~5% of those by design and every
  pause needs approval.
- Contribution margin breakdown uses its own RNG stream ([seed, 3]); weak-listing choice uses [seed, 4]. The weak
  listing story (P16, P26, P38 on seed 42) changed the hidden market, so all race numbers moved; README updated.
- The agent sells ~7% less in total than "fixed" but earns the most total contribution profit: it drops
  unprofitable low-margin sales. We show this openly; the growth objective is the lever for sellers who want volume.
- Growth objective check: realised ACOS lands at the cap (~35%), spend and clicks rise; sales for expensive items
  are noisy over 14 days (few orders), so tests check clicks/spend/ACOS rather than revenue.
- Not adopted: incrementality/holdout tests, platform-specific rule sets, rollback rules (see PLAN Phase 12).

## Phase 13 notes (2026-10-17)
- Sim cannibalization: ad sales reduce that day's organic sales by Binomial(ad_units, 1 - incrementality), capped at
  organic demand. Low-margin best-sellers got strong organic demand (organic x3 + 3) so low incrementality is real.
- Switchback tests at 14 days are noisy (CI ~0.7 wide). Rule: conclusive if the 90% CI excludes 1.0 (or is narrower
  than 0.6); factor = CI upper bound (conservative). On seed 42 P13 (true 0.36) -> CI [-0.14, 0.57], factor 0.57.
  Applying it adds ~Rs 1.7k of total profit: small, because the agent was already cutting this unprofitable product.
- Rollback first fired on noise (history bids were randomized +-30%); fixed by judging only the agent's own moves and
  p < 0.01 (~160 targets tested daily). In the stationary sim it is neutral (+-Rs 2k on ~Rs 900k); it is a safety
  net, proven by a test with a deliberately bad bid.
- Subprocess sandbox now passes data as pickle files written by the parent (same interpreter, trusted) so the child
  never imports pyarrow; Docker keeps parquet (version-safe). Linux RLIMIT_AS floor 1.5 GB so numpy/pandas can import;
  BLAS threads pinned to 1. The Linux path is exercised by CI only (no WSL/Linux on the dev machine).
- Platform profiles override `guardrails.conversion_lag_days` (amazon 2, google 3); generic keeps config values.
