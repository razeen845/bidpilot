# BidPilot

An AI bid manager **and consultant** for product ads and keyword ads, built for the AMD hackathon.
It runs on a **practice ad account** (simulated); no real ad platform API is called and no real money is spent.

- **Daily:** reads yesterday's results, sets bids with a math optimizer (never the LLM), applies safety rules,
  a stock guard and rollback rules, has the LLM plan the day and explain every change with real numbers,
  sends big changes to a human, and applies the rest.
- **Weekly (days 7 and 14) or on request:** the AI writes pandas code that runs in a sandbox (every result
  becomes a citable `[D#]` source), researches the web (every fact keeps its URL and an exact, verified quote,
  `[W#]`), and writes a ranked action list. A citation check removes anything not backed by a real source.
- **Incrementality:** switchback holdout tests measure how many ad sales would have happened anyway, and the
  optimizer values clicks by the sales they actually add.

The goal is not the lowest ACOS or the least spend. It is more **total contribution profit** (every unit sold,
ads + organic, at contribution margin after fees and returns, minus ad spend) with decisions anyone can check.

## Quick start (MVP)

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m cli demo                                   # prepares demo data and opens the dashboard on :8501
```

Or with Docker: `docker compose up app` (dashboard) / `docker compose --profile amd up` (dashboard + vLLM on ROCm,
on the AMD GPU box). See [MVP.md](MVP.md) for the scope, the demo script and the acceptance checks.

### The CLI

```text
python -m cli generate | reset | status
python -m cli day [--plan | --apply | --auto --days N]     # plan, decide, apply one day on the live account
python -m cli decide --approve K001 P07 --reject K012       # or --approve-all / --reject-all
python -m cli race [--llm]                                  # 14-day race on fresh copies of the store
python -m cli ask 1..6 | "free text"                        # ask the data ([D#] source + code + table)
python -m cli consult ; python -m cli recs --accept R1      # weekly consultant (needs the LLM)
python -m cli experiment list | propose P13 | approve X1 | cancel X1
python -m cli validate-config | check-llm | bench [--dry-run] | dashboard
```

`--no-llm` (or an unreachable endpoint) is always safe: bidding uses template explanations, "Ask the data"
answers the six preset questions with reviewed code, and the consultant is skipped.

### Configuration

`config.yaml` is validated on load (clear error messages). Secrets come from the environment:
`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `SEARCH_API_KEY`; also `BIDPILOT_OUT`, `BIDPILOT_CONFIG`,
`BIDPILOT_LOG_LEVEL`. Key settings: `platform: generic | amazon | google` (rule profile), `objectives`
(per-product growth push with an ACOS cap), `rollback`, `experiments`, `guardrails`.

## The dashboard

Eight pages in plain language: **Your store, Today's bids, Results, Weekly advice, Ask a question, Experiments,
Real ads check, Speed**. A status banner always shows the day, whether the AI is connected and how many changes
wait for you, with a *Next step* hint. **Simple view** (default) keeps to plain words; **Detailed view** (sidebar)
adds IDs, code, statistics and full tables. Every technical term has a hover explanation and a glossary; risky
actions (reset, switching store) ask for confirmation; big bid changes can be approved or rejected one by one or all
at once. The look is taken from block-printed textiles (indigo, madder, marigold) to match a home-textiles shop.

## Speed

Measured on an AMD Ryzen 5 7430U laptop (Radeon iGPU, llama.cpp Vulkan, Qwen2.5-3B): daily bidding for 160 ads
takes 6.8 min with BidPilot vs about 20.9 min with one AI call per ad (3.1x). Estimated on one **AMD Instinct MI300X
with vLLM and Qwen2.5-32B** (`python -m cli estimate-speed`, a roofline estimate from the measured workload, not a
measurement): about **7-10 s vs 152-250 s (about 21-24x)**, and the weekly analysis about 3.2-3.5 s side by side vs
8.4-9.0 s one by one. Confirm on the GPU with `python -m cli bench`.

## Results in simulation (seed 42, no LLM)

| Strategy | Total contribution profit | Ad-attributed profit | Total sales | Ad spend | ACOS | Stock availability | Ad spend while out of stock |
|---|---|---|---|---|---|---|---|
| fixed | Rs 807,136.80 | Rs 128,819.83 | Rs 3,020,230.00 | Rs 278,948.62 | 21.3% | 94.3% | Rs 9,640.36 |
| rules | Rs 842,639.92 | Rs 155,987.00 | Rs 2,931,930.00 | Rs 218,190.71 | 18.4% | 94.3% | Rs 10,161.33 |
| **agent_nollm** | **Rs 918,161.44** | **Rs 220,166.80** | Rs 2,903,090.00 | Rs 142,747.85 | 13.0% | 94.8% | **Rs 0.00** |

Read honestly: the agent sells 3.9% less than "fixed" because it stops buying unprofitable sales (low-margin
best-sellers bid far above break-even, many of whose ad sales would have happened organically anyway). It earns
the most contribution profit, keeps more products in stock, and never pays for ads on products that are out of
stock. It also wins on seeds 1, 7, 123 and 2024 on both profit measures. **This is a simulated account: it shows
the method works, not a guaranteed live-account result.**

## What is real and what is simulated

| Part | Real or simulated |
|---|---|
| Agent, optimizer, guardrails, stock guard, rollback, approvals, experiments analysis | Real (this repo) |
| LLMs on an AMD GPU via vLLM (ROCm), OpenAI-compatible API | Real when `LLM_BASE_URL` points at the AMD box |
| AI-written analysis code in a sandbox | Real: AST policy check + separate restricted process (or Docker) |
| Web research and citations | Real pages, URLs and quotes verified against page text; replayed from cache in the demo |
| Ad account, store, stock, market response, organic cannibalization | **Simulated** (the agent never sees the hidden formulas) |
| Amazon / Google | **Rule profiles only** (lag, attribution window, KPI, waste action). Not connected |

## How it works

**Bidding** (`agent/workflows.py`, `engine/`): curves `clicks(b) = C(1 − e^(−kb))` are fitted per target; the
optimizer tests the current bid ×0.80…1.20 and picks the most expected profit, valuing a click at
`cvr × price × contribution margin × incrementality`. Guardrails then apply, in priority order:
pause > stock guard > approved experiment > rollback > consultant action > waste cleanup > optimizer.

- **Stock guard:** < 3 days of stock → bid ×0.5; out of stock → pause; restocked → resume; overstocked → favour higher bids.
- **Waste cleanup:** 14-day window, newest day's clicks ignored (conversion lag; 2 days on Amazon, 3 on Google),
  pause only if zero sales would be < 5% likely at the category's conversion rate.
- **Proven targets:** top-10 profit makers can be cut at most 10% a day; cuts need approval.
- **Rollback:** if a bid the agent moved ≥ 15% coincides with a significant profit drop (Welch t-test p < 0.01,
  ≥ Rs 300/day), revert and hold for 3 days. In a stable market it rarely fires; it is a safety net.
- **Approval:** changes > 10%, any pause, any change on a top-10 spender, any cut on a proven target.
- **Seller objectives:** "growth" products get the most sales whose expected ACOS stays under the chosen cap.
- Every change carries evidence (this week vs last week), role (`discovery | proven | waste_candidate | standard`),
  confidence (`high | medium | low`) and, for waste, the platform action (e.g. "add as negative exact keyword").

**Incrementality** (`agent/experiments.py`): a switchback test turns a product's ads ON/OFF on alternating days
(randomized in 2-day blocks) and compares total units sold. If the 90% CI excludes 1.0, the product gets a
factor equal to the CI's upper bound (conservative). Tests pause ads, so the owner approves them first. On the
planted best-seller P13 (true incrementality 0.36) a 14-day test returns a CI entirely below 1.0.

**Consultant** (`agent/consultant.py`): brief → LLM plan → analytics and web research **in parallel** →
ranked recommendations (priority, confidence, risk, next action) → citation check on finding, impact and risk
(one rewrite, else "unverified") → the human accepts or rejects. Accepted bid actions go into the next day through
the same guardrails; the rest become owner to-dos.

**Explanations** (`agent/explain.py`): the LLM writes ≤ 25-word reasons; every number must exist in the change's
evidence, otherwise a template reason is used and the rejection is logged.

## Architecture

```
cli.py                  one entry point for every operation
settings.py             validated config, platform profile, logging, atomic writes
data/generate.py        fake store + hidden market + 30-day history (stories planted for the agent to find)
sim/market.py           hidden response curves + organic cannibalization (only sim/ and data/ import it)
sim/platform.py         practice ad account: report, stock, set_bids, pause, resume, step
sim/runner.py           N-day races per strategy, fair outcome report
analysis/metrics.py     rolling metrics: spend, ROAS, ACOS, CVR, CPC, days of stock
engine/curves.py, optimizer.py, guardrails.py, rollback.py, platform_rules.py, baselines.py
agent/workflows.py      daily workflows + merge     agent/orchestrator.py  one agent day, approval queue
agent/llm.py            async OpenAI-compatible client (semaphore, retry, JSON schema, call log)
agent/explain.py        explanations + number check  agent/citations.py     citation registry + checks
agent/analyst.py        question → code → policy → sandbox → [D#]
agent/research.py       search → page text → exact quotes → quote check → [W#]
agent/consultant.py     weekly consultant          agent/experiments.py   switchback incrementality tests
sandbox/policy.py, exec_runner.py, Dockerfile       AST allow-list + isolated runner
app/dashboard.py        Streamlit: Store, Run a day, Race, Consultant, Ask the data, Experiments, Speed
bench/latency.py        naive vs BidPilot, sequential vs parallel consultant (on the AMD GPU)
```

Hard rules enforced in code and tests: the LLM never outputs bid numbers; LLM-written code only runs in the
sandbox after the policy check; `agent/`, `engine/`, `analysis/` and `sandbox/` never touch the hidden market
(a test greps for it); same seed = same results (common random numbers per `(seed, day, entity)`); every LLM
sentence with a number is checked against its source.

## Quality

`ruff check .` and `pytest` (136 tests, 93% coverage, 25/25 planted bugs caught - see docs/TESTING.md: simulator, engine, guardrails, LLM client against a fake
OpenAI-compatible server, sandbox escapes, research quote checks, consultant citation checks, experiments,
rollback, platform profiles, config validation, CLI end to end, dashboard via Streamlit AppTest) run in GitHub
Actions (`.github/workflows/ci.yml`).

## Honest notes and limitations

- **Not yet measured on the AMD GPU.** No LLM endpoint was reachable while building. Every LLM path is tested
  against fakes; the consultant report and the speed numbers need `LLM_BASE_URL` (see MVP.md, "On the AMD box").
- **Research sources were hand-collected** (no search API key): 9 real pages fetched on 2026-10-15, every quote
  verified against the page text by the same code the live pipeline uses. The live Tavily/Brave path is
  implemented and tested with fakes only.
- **Sandbox:** subprocess backend (Docker not installed here). Windows: timeout only; Linux: CPU + address-space
  limits. The Docker backend (`--network none --read-only`) is implemented but untested here.
- **Incrementality tests are noisy at 14 days** (7 ON / 7 OFF days); only strong effects become conclusive, and
  the factor used is the conservative CI upper bound. Longer tests give tighter answers.
- **Platform profiles change rules and wording only**; there is no Amazon or Google connection yet.

## Next steps

Connect a real Google Ads / Amazon Ads account behind the same `AdPlatform` interface (read reports, write
bids, negative keywords), starting read-only with recommendations before allowing writes.
