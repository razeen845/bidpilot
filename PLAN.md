# BidPilot — Build Plan (for Claude Code)

Read this whole file before writing any code. Then do **only the phase you are asked to do**, and stop when its "Done when" checks pass.

## 1. What we are building (short)

An AI bid manager **and consultant** for **product ads** and **keyword ads**.

**Daily (bidding):**
1. Reads yesterday's results (spend, clicks, sales, stock).
2. Calculates new bids with a math optimizer (NOT the LLM).
3. Applies safety rules and a stock guard.
4. Uses an LLM to plan the day and explain every change, using only real numbers from the data.
5. Sends big changes to a human for approval; small changes apply automatically.
6. Applies bids to a **simulated ad account** (no real ad APIs, no real money).

**Weekly or on request (consultancy):**
7. **Data analytics:** the AI writes pandas code to answer questions about the data and runs it in a **sandbox**. Every result is saved and becomes a citable data source `[D1]`, `[D2]`, …
8. **Research:** the AI searches the web for market facts (seasonality, benchmarks, best practices). Every fact keeps its URL and an exact quote, and becomes a citable web source `[W1]`, `[W2]`, …
9. **Consultant report:** up to 5 recommendations, each with citations. A **citation check** removes anything not backed by a real data result or a real retrieved quote. The human picks which recommendations to act on. Bid-related ones go into the next day's bidding; others (restock, pricing) become owner to-dos.

Demo goal:
- 14-day **cumulative profit** chart: `fixed` vs `rules` vs `agent`. The agent should win.
- A consultant report with clickable citations (data tables with the code that made them, and web quotes with links).
- A measured speed result on the AMD GPU.

Hackathon: AMD. LLMs run on an AMD GPU (MI300X class) in the cloud, served with **vLLM on ROCm** through its OpenAI-compatible API. Deadline: 2.5 days.

## 2. Hard rules (follow in every phase)

- **No real ad platform APIs.** Everything goes through `sim/` (the practice ad account).
- **The LLM never outputs bid numbers.** Bids come from `engine/optimizer.py`. Consultant bid actions are bounded instructions (for example "lower by up to 20%") that still pass through guardrails.
- **Every number or fact the AI shows must be cited and checked.** Data claims → `[D#]` results that contain the number. Web claims → `[W#]` sources whose quote was found in the retrieved page text. Unchecked claims are dropped or shown as "unverified", never shown as fact.
- **AI-written code only runs inside the sandbox** (`sandbox/`). Never `exec` LLM code in the main process.
- **Same seed = same results.** All randomness uses `numpy.random.default_rng` seeded from config. Use common random numbers: draws depend only on `(seed, day, entity_id)`, so all strategies see identical luck.
- **The product must run with no LLM** (`--no-llm`): bidding works with template explanations; consultancy is skipped.
- **The demo must not depend on live internet.** Research has `live` and `cache` modes. Record once live, demo from cache, and show the retrieval date.
- **LLM client is OpenAI-compatible** (`base_url` + `api_key` in config).
- Keep it simple: plain Python modules, no agent frameworks (no LangChain/LangGraph), no database (CSV/Parquet/JSON files), no auth.
- Every phase adds `pytest` tests for its own module. Do not break earlier tests.
- Do not add features outside this plan. If something seems missing, write it in `NOTES.md` and keep going.
- Currency is Rs. Money values are rounded to 2 decimals in outputs.

## 3. Tech stack

- Python 3.11, `numpy`, `pandas`, `scipy`, `pyyaml`, `pydantic`, `openai` (client only), `httpx`, `trafilatura` (page text), `streamlit`, `plotly`, `pytest`.
- LLM serving: `vllm serve <model> --port 8000` on the AMD GPU box (ROCm build of vLLM).
- Start with **one instruct model for all LLM roles** (an open Qwen or Llama instruct model; size chosen in Phase 0). A second, smaller model is optional on Day 2 only if time allows.
- Web search: any search API with a key (for example Tavily or Brave Search). Provider and key in config / env var `SEARCH_API_KEY`.
- Sandbox: Docker with `--network none --read-only` if available on the machine; otherwise a subprocess sandbox (see Phase 6). Record which one is used in `NOTES.md`.

## 4. Repo structure

```
bidpilot/
  config.yaml
  PLAN.md  CLAUDE.md  NOTES.md  README.md
  data/
    generate.py          # Phase 1: fake store + hidden market + 30-day history
  sim/
    market.py            # Phase 1: hidden response functions (agent must NOT import this)
    platform.py          # Phase 2: practice ad account API
    runner.py            # Phase 2/5: run N days for a strategy
  analysis/
    metrics.py           # Phase 2: per-entity rolling metrics
  engine/
    baselines.py         # Phase 2: fixed + rules strategies
    curves.py            # Phase 3: fit response curves from history
    optimizer.py         # Phase 3: choose bids
    guardrails.py        # Phase 3: limits, stock guard, waste cleanup, approval flag
  agent/
    llm.py               # Phase 4: OpenAI-compatible client, async, JSON outputs
    prompts.py           # Phase 4+: prompt templates + JSON schemas
    explain.py           # Phase 4: explanations + number check
    workflows.py         # Phase 5: the 4 bidding workflows
    orchestrator.py      # Phase 5: one agent day, approval queue
    analyst.py           # Phase 6: question -> code -> sandbox -> [D#] result
    research.py          # Phase 7: web search -> verified [W#] sources
    citations.py         # Phase 7/8: citation registry + checks
    consultant.py        # Phase 8: weekly consultant report
  sandbox/
    exec_runner.py       # Phase 6: runs untrusted code safely
    policy.py            # Phase 6: AST allow-list checks
  app/
    dashboard.py         # Phase 9: Streamlit UI
  bench/
    latency.py           # Phase 10: speed test
  out/                   # generated files (gitignored)
    analysis/  research/  consult/
  tests/
```

`agent/`, `engine/`, `analysis/`, `sandbox/` may only read data through `sim/platform.py` and files in `out/`. They must never import `sim/market.py` or read `out/_hidden_market.json`.

## 5. config.yaml (defaults)

```yaml
seed: 42
store:
  n_products: 40
  n_keywords: 120          # each keyword maps to one product
  category_theme: home textiles
  history_days: 30
  sim_days: 14
  daily_budget: 25000      # Rs, total for all ads per day
bids:
  min_bid: 3
  max_bid: 80
  max_daily_change: 0.20   # ±20% per day
guardrails:
  approval_change_pct: 0.10    # bigger than ±10% needs approval
  approval_top_spenders: 10    # any change on top-10 spenders (7d) needs approval
  min_clicks_to_act: 10        # fewer clicks in history -> hold ("still learning")
  waste_min_spend_7d: 1500     # Rs
  waste_min_clicks_7d: 20
  low_stock_days: 3
  overstock_days: 60
llm:
  enabled: true
  base_url: http://localhost:8000/v1
  api_key: EMPTY
  model_planner: <set in Phase 0>
  model_explainer: <set in Phase 0>
  max_concurrency: 32
  timeout_s: 30
sandbox:
  backend: docker          # docker | subprocess
  timeout_s: 10
  memory_mb: 512
  max_result_rows: 50
  max_fix_attempts: 1
research:
  mode: cache              # live | cache
  provider: tavily         # tavily | brave
  max_queries: 4
  results_per_query: 3
  max_quote_words: 25
consultant:
  run_on_days: [7, 14]
  max_recommendations: 5
  max_analysis_questions: 3
```

## 6. Data model

**products.csv**: `product_id, name, category, price, margin_pct, stock_units, organic_sales_per_day, restock_day, restock_units`

**entities.csv**: `entity_id, entity_type (product|keyword), name, product_id, start_bid`
- 40 product entities + 120 keyword entities = 160 things we bid on.

**daily_stats.parquet**: `day, entity_id, bid, clicks, cost, conversions, revenue, profit` (`profit = revenue × margin_pct − cost`)

**stock.parquet**: `day, product_id, stock_units`

**changes.parquet**: every applied Change record with `day` and `approved_by` (`auto|human`).

**Change record** (pydantic):
```
entity_id, entity_type, name, workflow (checkup|bid_adjust|waste|stock_guard|consultant),
action (raise|lower|pause|hold|resume), old_bid, new_bid, change_pct,
evidence: {spend_7d, clicks_7d, conversions_7d, revenue_7d, roas_7d, profit_7d, days_of_stock},
needs_approval: bool, approval_reason: str|None, reason: str|None
```

**Data source `[D#]`** (`out/analysis/D{n}.json`):
```
id, question, code, result_table (≤ 50 rows, list of dicts), result_summary (dict of key numbers), created_at, sandbox_backend
```

**Web source `[W#]`** (`out/research/W{n}.json` or inside a run file):
```
id, query, url, title, fact (one sentence), quote (exact text from the page, ≤ 25 words), retrieved_at, verified: bool
```

**Recommendation** (consultant output):
```
id, title, finding, recommendation, type (bid_action|owner_todo),
bid_action: {filter: {entity_type?, category?, keyword_contains?, product_ids?}, action: lower|raise|pause, max_pct}|None,
expected_impact: str, citations: ["D1","W2", ...], status (verified|unverified)
```

## 7. Phases and owners

Owners for a 3-person team: **A** = data/engine/analytics, **B** = AI/AMD/consultant, **C** = research/dashboard/pitch.

| Day | A | B | C |
|---|---|---|---|
| 1 | Phases 1, 2, 3 | Phases 0, 4 | Phase 7 (research), dashboard skeleton on mock JSON |
| 2 | Phase 6 (analyst sandbox) | Phases 5, 8 | Phase 9 (dashboard) |
| 3 (half) | Demo run + bug fixes | Phase 10 (speed test) | Phase 11 (polish, video, slides) |

**Day 1 gate:** `agent_nollm` beats both baselines **and** research returns at least 5 verified web sources.
**Day 2 gate:** the full demo runs from the dashboard: run a day → approve → 14-day race → consultant report with working citations → one "Ask the data" question.

---

### Phase 0 — Setup + model on AMD (Owner B, 0–3h, Day 1)

- Create repo structure, `config.yaml`, `requirements.txt`, pytest setup, `.gitignore` (`out/`, `.env`).
- On the AMD box, start vLLM with an instruct model. Write `scripts/check_llm.py`: one chat request + one JSON-schema request (`response_format={"type":"json_schema",...}`), print latency.
- If JSON schema output isn't supported by the installed vLLM version, fall back to "JSON only" prompting + `json.loads` with one retry. Record which works in `NOTES.md`.
- Check whether Docker is available on the dev machine for the sandbox; record in `NOTES.md`.

Done when: `python scripts/check_llm.py` prints valid JSON from the AMD-served model.
**If not working by hour 3:** switch to a smaller model. If AMD is unavailable, point `base_url` at any OpenAI-compatible endpoint and continue; retry AMD later.

---

### Phase 1 — Fake store + hidden market (Owner A, ~3h, Day 1)

`sim/market.py` — hidden truth per entity:
- `click_cap`, `k`, `cvr`, `cpc_ratio` (0.6–0.9).
- Expected clicks at bid `b`: `click_cap × (1 − exp(−k × b))`; actual clicks: Poisson.
- Cost = clicks × b × cpc_ratio. Conversions: Binomial(clicks, cvr), **capped by available stock** (zero stock → ads cost money but sell nothing).
- Revenue = conversions × product price.

`data/generate.py` — a home-textiles store with deliberate stories for the agent and consultant to find:
- ~15% of keywords are junk (cvr near 0). Make most junk keywords share patterns (e.g., contain "free", "cheap", "diy") so the consultant can find a pattern, not just single keywords.
- Mix of high-margin (40–60%) and low-margin (8–15%) products; a few best-sellers are low-margin and eat a large share of budget.
- 4–6 products near stockout during the sim window (some restock, some don't); 4–6 overstocked.
- Some winners under-bid, some over-bid.
- One category performs clearly better than the others (for "which category should get more budget").
- **30 days of history** with bids varying randomly ±30% around `start_bid`.
- Prices Rs 800–6,000, bids Rs 5–60.
- Write `out/products.csv`, `out/entities.csv`, `out/history.parquet`, `out/stock_history.parquet`; hidden params to `out/_hidden_market.json`.

Done when: generation takes < 10 s; tests confirm same seed gives identical files, junk keywords have near-zero conversions, ≥ 4 products hit low stock within 14 days.

---

### Phase 2 — Practice ad account + metrics + baselines (Owner A, ~3h, Day 1)

`sim/platform.py` — `AdPlatform`: `get_report(day, lookback=7)`, `get_stock(day)`, `set_bids({id: bid})`, `pause(id)`, `resume(id)`, `step()` (simulate one day; enforce `daily_budget` by scaling clicks proportionally if needed; update stock with ad + organic sales and restocks; append stats). Draws seeded by `(seed, day, entity_id)`.

`analysis/metrics.py`: rolling 7-day spend, clicks, conversions, revenue, profit, ROAS, CPC, days_of_stock, top spenders.

`engine/baselines.py`: `fixed` (never changes) and `rules` (7d ROAS < 1/margin_pct → ×0.9, else ×1.05; no stock awareness).

`sim/runner.py`: `run(strategy, days=14)` → daily totals (spend, revenue, profit, cumulative profit, stockout ad spend). Fresh store copy per run.

Done when: `python -m sim.runner --strategy fixed|rules` prints totals; tests confirm determinism, budget never exceeded, paused entities get 0 clicks.

---

### Phase 3 — Optimizer + guardrails (Owner A, ~3h, Day 1 night) ⟵ part of Day 1 gate

`engine/curves.py`: fit `clicks(b) = C(1 − exp(−k b))` per entity (`scipy.optimize.curve_fit`, bounded); shrunk conversion rate `(conv + a)/(clicks + a + b)` toward the category average; estimated cpc_ratio. Too few clicks or fit fails → `learning` (hold).

`engine/optimizer.py`: test `current × [0.80 … 1.20]` (step 0.05, clipped). Expected profit(b) = clicks(b) × (cvr × price × margin − b × cpc_ratio). Pick the max. If expected spend > budget, cut the lowest profit-per-rupee entities first. All 160 entities in < 1 s.

`engine/guardrails.py` → Change records:
- **Stock guard:** days_of_stock < `low_stock_days` → ×0.5; stock 0 → pause; restocked → resume; overstocked → favour the upper end of the range.
- **Waste cleanup:** spend_7d ≥ threshold, clicks_7d ≥ threshold, 0 conversions → pause.
- **Limits:** ±20%/day (except stock guard and pauses), min/max bid.
- **Approval flag:** |change| > `approval_change_pct`, any pause, or top spender → needs approval, with `approval_reason`.
- `evidence` filled on every change.

Add strategy `agent_nollm` to the runner (auto-approve, template reasons).

Done when: `python -m sim.runner --compare` shows `agent_nollm` beating `fixed` and `rules` on 14-day profit, with lower stockout ad spend than `rules`. If not, fix curve fits and stock guard before anything else.

---

### Phase 4 — LLM client + prompts + explanations (Owner B, ~2h, Day 1 night or Day 2 morning)

`agent/llm.py`: async OpenAI-compatible client; semaphore (`max_concurrency`); timeout; 1 retry; `chat_json(model, system, user, schema)`; logs every call (role, model, tokens in/out, seconds) to `out/llm_log.jsonl`.

`agent/prompts.py` — bidding roles (consultancy prompts are added in Phases 6–8):
1. **Planner** (once/day). In: account summary. Out: `{"focus": [...], "hold": [ids], "hold_reasons": {id: reason}, "day_note": "2-3 sentences"}`. Can only hold changes, never set bids.
2. **Explainer** (one per change, concurrent). Out: `{"reason": "≤ 25 words, uses numbers from evidence"}`.
3. **Day report** (once/day). Out: `{"report": "short summary"}`.

`agent/explain.py`: **number check**: every number in a reason must match a value in that change's evidence or bids (rounding allowed); otherwise use a template reason and log it. Template reasons for `--no-llm`.

Done when: 160 explanations run concurrently against AMD with fallbacks working; tests (fake LLM) show the number check rejects a made-up number and planner holds keep old bids.

---

### Phase 5 — Bidding workflows + orchestrator (Owner B, ~3h, Day 2 morning)

`agent/workflows.py`:
1. `daily_checkup` — flags movers (cost or conversions changed > 30% vs the previous 7 days). Flags only.
2. `bid_adjust` — optimizer output.
3. `waste_cleanup` — pauses from guardrails.
4. `stock_guard` — stock rules.
Merge priority per entity: pause > stock_guard > consultant > waste > bid_adjust.

`agent/orchestrator.py` — `run_agent_day(platform, day, auto_approve, pending_consultant_actions=None)`:
1. Pull report + stock → 2. run workflows (+ apply any approved consultant bid actions, through guardrails) → 3. planner → 4. explainers (concurrent) → 5. split auto / needs approval, save `out/day_{n}_plan.json` → 6. apply (auto-approve, or wait for `out/day_{n}_decisions.json` from the dashboard) → 7. day report → 8. `platform.step()`. If `day in consultant.run_on_days`, run the consultant (Phase 8) after the step.

Add strategy `agent` to the runner (LLM on, auto-approve everything, including consultant bid actions with status `verified`).

Done when: `--compare` shows all strategies with `agent` ≥ `rules`; one agent day with LLM finishes in < 60 s.

---

### Phase 6 — Data analytics in the sandbox: "Ask the data" (Owner A, ~4h, Day 2)

`sandbox/policy.py` — AST check before running any code:
- Allowed imports: `pandas`, `numpy`, `math`, `statistics`. Anything else → reject.
- Forbidden names/calls: `open`, `exec`, `eval`, `compile`, `__import__`, `getattr`, `setattr`, `globals`, `locals`, `input`, any dunder attribute, `os`, `sys`, `subprocess`, `socket`, `pathlib`, `shutil`, `requests`, `httpx`.
- Pandas I/O methods (`read_*` except the provided loader, `to_*` file writers) → reject.

`sandbox/exec_runner.py` — `run_code(code) -> {ok, result_table, result_summary, error}`:
- Copies the allowed data (`daily_stats`, `stock`, `products`, `entities`, `changes`) into a temp dir as parquet. **Never the hidden market file.**
- `docker` backend: `docker run --rm --network none --read-only --memory {memory_mb}m` with the temp dir mounted read-only and a tiny harness script.
- `subprocess` backend: separate Python process, empty env, `resource.setrlimit` for CPU time and memory (Linux), cwd = temp dir, timeout.
- Harness preloads DataFrames named `stats, stock, products, entities, changes`. The code must assign `result` (DataFrame, Series, or scalar) and may assign `summary` (dict of up to 10 key numbers).
- Output is JSON on stdout; truncate to `max_result_rows`.

`agent/analyst.py` — `ask(question) -> D source`:
1. LLM gets the question + table schemas (column names, types, 3 sample rows) → returns `{"code": "..."}`.
2. Policy check → sandbox run. On error, send the error back once for a fix (`max_fix_attempts`).
3. Save `out/analysis/D{n}.json` (question, code, result table, summary).
4. LLM writes a 1–3 sentence answer that must cite `[D{n}]`; the number check (from Phase 4) runs against the result table and summary.

Preset demo questions (must all work; store their D files for the demo):
- "Which keyword patterns spent the most with zero sales in the last 14 days?"
- "Which category gives the most profit per rupee of ad spend?"
- "How much ad money was spent on products while they were out of stock?"
- "Which low-margin products take the biggest share of the budget?"

Done when: all 4 preset questions return correct tables (verify against a hand-written pandas answer in tests); the policy blocks `import os`, `open(...)`, and `__import__`; a sandbox timeout is handled cleanly.

---

### Phase 7 — Research with verified citations (Owner C, ~3h, Day 1 afternoon)

`agent/research.py` — `research(queries) -> [W sources]`:
1. For each query (max `max_queries`): call the search API, take the top `results_per_query` results.
2. Get the page text (the search API's content field if provided; otherwise `httpx` + `trafilatura`). Skip pages that fail.
3. LLM extracts up to 2 facts per page, each with an **exact quote** from the text (≤ 25 words).
4. **Quote check:** normalise whitespace and quotes, then the quote must appear in the page text. Pass → `verified: true`. Fail → drop it.
5. Save to `out/research/run_{timestamp}.json`; register sources as `W1, W2, …`.
6. `cache` mode loads the latest run file instead of calling the API. `live` mode calls and saves.

`agent/citations.py` — `CitationRegistry`: holds D and W sources for one consultant run; `get(id)`; `check_text(text, cited_ids)` → every `[X#]` tag in the text exists; numbers in the text match a cited D source (reuse the Phase 4 number check); W-cited sentences only cite verified sources.

Seed queries for testing (the consultant will generate its own later):
- "home textile online sales seasonality"
- "e-commerce home goods average conversion rate"
- "shopping ads bidding low inventory best practice"
- "search ads negative keywords free cheap"

Done when: a live run returns ≥ 5 verified sources with working URLs; a test with a fake page proves an invented quote is rejected; cache mode works with no network.
**If the search API isn't working by end of Day 1:** hand-collect 6–8 sources with real URLs and exact quotes into a cache file, run them through the same quote check against saved page text, and use cache mode. Note this honestly in the README.

---

### Phase 8 — Consultant (Owner B, ~3h, Day 2 afternoon/night)

`agent/consultant.py` — `run_consultant(day) -> report`:
1. **Brief:** build a summary of the last 7 days (totals, trends, change log, approvals and rejections, stock events, waste found).
2. **Plan (LLM):** `{"questions": [≤ 3 analysis questions], "queries": [≤ 4 web search queries]}`.
3. **Fan out in parallel:** analyst questions (Phase 6) and research queries (Phase 7) run at the same time (`asyncio.gather`). This parallel step is part of the speed story.
4. **Write (LLM):** up to 5 Recommendations (schema in section 6). Each must have ≥ 1 citation. Prompt rule: "Only state numbers that appear in a cited D source. Only state market facts that appear in a cited W source."
5. **Check:** `CitationRegistry.check_text` on each recommendation's finding + impact. Pass → `verified`. Fail → one rewrite attempt with the error; still failing → `unverified` (shown greyed out; cannot be auto-applied).
6. Save `out/consult/day_{n}.json` and a markdown render.
7. **Human picks:** the dashboard lets the user accept or reject each recommendation. Accepted `bid_action`s become `pending_consultant_actions` for the next day (through guardrails and approval rules). Accepted `owner_todo`s go to `out/todos.json`.

The 3 kinds of recommendation we expect from the seeded data (don't hard-code them; they should come out of the analysis):
- a junk keyword pattern to stop targeting,
- low-margin products taking too much budget (owner to-do: price or drop),
- a stockout that cost sales (owner to-do: restock earlier), optionally backed by a web source on seasonality or low-inventory practice.

Done when: running the consultant on day 7 produces ≥ 3 verified recommendations with at least one D citation each and at least one W citation in the report; clicking through citations shows the code/table or the URL/quote; an accepted bid action changes the next day's bids.

---

### Phase 9 — Dashboard (Owner C, ~5h, Day 2; skeleton on Day 1 with mock JSON)

`app/dashboard.py` (Streamlit), 6 tabs:
1. **Store** — products, stock, days of stock (low stock highlighted), 7-day totals.
2. **Run a day** — runs `run_agent_day(auto_approve=False)`; day note; change table (name, workflow, old → new bid, reason); Approve/Reject on approval rows; "Apply decisions".
3. **14-day race** — cumulative profit chart (fixed, rules, agent) + totals table (profit, spend, stockout ad spend).
4. **Consultant** — latest report: each recommendation as a card with its citations as chips. Clicking `[D#]` shows the question, code, and table; clicking `[W#]` shows the title, link, quote, and retrieval date. Accept/Reject per recommendation. Owner to-do list.
5. **Ask the data** — preset question buttons + free-text box → shows answer with `[D#]`, the generated code, and the table.
6. **Speed** — `bench/results.json`.

Cache results in `out/` so the demo doesn't recompute unless asked.

Done when (**Day 2 gate**): the whole demo script in section 8 works from the dashboard. Plain looks are fine.

---

### Phase 10 — Speed test on AMD (Owner B, ~2h, Day 3)

`bench/latency.py`:
- **Bidding, naive:** one LLM call per entity, sequential, LLM picks bid + reason.
- **Bidding, BidPilot:** optimizer + guardrails + 1 planner call + concurrent explainers.
- **Consultant, sequential vs parallel:** same analysis questions and research (cache mode for fairness) run one after another vs `asyncio.gather`.

Record wall time, LLM calls, tokens, tokens/second (from `llm_log.jsonl`), GPU name; median of 3 runs. Write `bench/results.json` and `out/speed_chart.png`.

Done when: results contain real numbers measured on the AMD GPU.

---

### Phase 11 — Polish + demo assets (Owner C, Day 3)

- `README.md`: what it is, how to run, architecture, what's simulated vs real (AI, sandbox, web research and citations are real; ad account and store are simulated), next steps (real Google Ads API).
- Run the demo script end to end; fix only bugs.
- Switch research to `cache` mode and confirm the demo works with Wi-Fi off.
- Export the race chart, a consultant report screenshot, and the speed chart for slides.
- **No new features on Day 3.**

## 8. Demo script (4 minutes)

1. Store tab: 40 products, ~120 keywords, some low stock (20 s).
2. Run a day: plan, reasons with numbers, approve one, reject one (40 s).
3. 14-day race chart: agent vs rules vs fixed (30 s).
4. Consultant tab: day-7 report; click a `[D#]` to show the code and table, click a `[W#]` to show the source quote; accept a recommendation (60 s).
5. Ask the data: one preset question live (30 s).
6. Speed tab: naive vs BidPilot on AMD (30 s).
7. Closing: practice ad account today, real Google Ads connection next (10 s).

## 9. Using this plan with Claude Code

Start each session with:

> Read PLAN.md and CLAUDE.md. Do Phase N only. Show me the "Done when" checks passing, then stop.

- One phase per session keeps context small and mistakes contained.
- After each phase: `pytest`, then commit `phase N: <summary>`.
- If a phase's "Done when" fails twice, stop and write the problem in `NOTES.md` instead of piling on workarounds.
- Phases 6 and 7 don't depend on Phase 5, so they can be built in parallel by different people.

## 10. Cut list (if behind schedule, cut in this order)

1. Second smaller LLM model (one model for all roles).
2. Day report LLM call (template instead).
3. Free-text "Ask the data" (keep the 4 preset questions).
4. Live research in the demo (cache only; still show real URLs and dates).
5. Docker sandbox (use the subprocess sandbox and say so).
6. Store tab.
7. Planner `hold` logic (planner only writes the day note).

**Never cut:** optimizer, stock guard, approval step, number and citation checks, a consultant report with at least D citations, the 14-day chart, and the measured speed result.

---

## 11. Phase 12 — Practitioner upgrades (from the marketing strategist's review)

Source: "BidPilot — Updated Plain-English Summary, grounded in practical e-commerce & PPC experience".
Adopted on Day 2 because Phases 0–11 were already done. Same hard rules apply (LLM never sets bids,
everything cited, `--no-llm` works, same seed = same results).

**Principle:** the goal is not minimum ad spend or lowest ACOS; it is more incremental, sustainable
contribution profit with transparent decisions.

1. **Metrics done right** (`analysis/metrics.py`): add ACOS (spend / attributed sales) and conversion rate
   next to ROAS and CPC. Every Change `evidence` also carries the previous 7 days (comparison period).
2. **Contribution margin** (`data/generate.py`): products get `fee_pct` (marketplace + fulfilment) and
   `return_rate`; `margin_pct` is the contribution margin after fees and returns
   (`gross_margin_pct = margin_pct + fee_pct + return_rate × 0.5`). All profit math uses contribution margin.
3. **Waste with low-data and lag protection** (`engine/guardrails.py`): evaluate over
   `waste_window_days` (14), ignore the last `conversion_lag_days` (1) of clicks, and pause only if zero
   sales is statistically unlikely: `(1 − category_cvr)^clicks < 1 − waste_confidence` (0.95).
4. **Protect proven targets**: the top `protect_top_profit` (10) entities by 7-day profit can be cut at most
   `protected_max_cut` (10%) per day, and any cut on them needs approval.
5. **Roles + confidence**: each entity is labelled `discovery | proven | waste_candidate | standard`; each
   Change gets `confidence: high | medium | low` from data volume.
6. **Seller objectives**: a product can be set to `growth` with a `max_acos`; the optimizer then picks the
   bid with the most expected sales whose expected ACOS stays under the cap (else falls back to profit).
   Set in config (`objectives`) or from the dashboard Store tab (`out/objectives.json`).
7. **Consultant = prioritized action list**: each recommendation gets `priority`, `confidence`, `risk`,
   `next_action`; the citation check covers finding, impact and risk. Account data `[D#]` and external
   evidence `[W#]` are kept in separate sentences.
8. **New preset analyses**: "Which products get many clicks but convert below their category average?"
   (listing/conversion issues) and "Which overstocked products are profitable enough to push harder?"
   (inventory pressure).
9. **Fair outcome report** (`sim/runner.py`, dashboard race tab): total sales (ads + organic), attributed
   sales, ad spend, ACOS, ROAS, conversion rate, stock availability, attributed contribution profit and
   **total contribution profit** (all sales × contribution margin − ad spend).
10. **Wording**: README and dashboard say "in simulation"; no claims about live-account results.

Done when: `pytest` passes; `--compare` shows the agent ahead of `fixed` and `rules` on **total**
contribution profit and attributed profit, with lower stockout ad spend; junk keywords are still paused;
a growth product ends with higher ad sales than under the profit objective.

Not adopted (time), recorded as limitations: incrementality / holdout testing (the sim treats attributed
sales as incremental), platform-specific rule sets (Amazon vs Google), automatic rollback rules.

---

## 12. Phase 13 — Remaining practitioner items + production hardening (MVP)

Adopts the three items Phase 12 left out, then hardens the product into a runnable MVP.

1. **Incrementality** (attributed ≠ incremental):
   - Sim: each product has a hidden `incrementality` (share of ad sales that would NOT have happened
     organically). Ad sales cannibalize organic sales by `(1 − incrementality)`. Best-sellers with strong
     organic demand are planted with low incrementality. Platforms still report every ad sale as attributed.
   - Measurement (`agent/experiments.py`): a **switchback holdout test** per product — ads ON / OFF on
     alternating days for `experiments.days` (14). Incrementality = (units/day ON − units/day OFF) / attributed
     units/day ON, with a 90% confidence interval (Welch). Conclusive only if the CI is narrow enough.
   - Use: a conclusive result sets the product's incrementality factor; the optimizer values a click at
     `cvr × price × margin × incrementality`. Experiments need owner approval to start (they pause ads).
2. **Platform rule profiles** (`engine/platform_rules.py`): `generic | amazon | google`. A profile sets the
   conversion lag, attribution window, primary KPI (ACOS for Amazon, ROAS for Google), and how waste is
   actioned (Amazon: negative exact keyword; Google: negative phrase keyword; product ads: pause/exclude).
   No real platform API is called; the profile only changes the rules and wording.
3. **Rollback rules** (`engine/rollback.py`): if an entity's bid moved ≥ 15% over the last
   `rollback.window_days` (4) and its daily profit fell significantly vs the window before (Welch t-test,
   p < 0.05, drop ≥ Rs `rollback.min_drop_per_day`), revert to the earlier bid and hold the optimizer off it for
   `rollback.cooldown_days`.
4. **Production hardening:** validated config (pydantic, clear errors), structured logging, atomic JSON/state
   writes, one CLI (`python -m cli ...`), Dockerfile + docker-compose (dashboard + optional vLLM on ROCm),
   GitHub Actions CI (ruff + pytest), `pyproject.toml`.
5. **MVP:** `python -m cli demo` prepares data and starts the dashboard; `MVP.md` describes scope, how to run,
   and acceptance checks.

Done when: `pytest` and `ruff` pass; the agent still beats `fixed` and `rules` on total contribution profit;
a switchback test on a planted low-incrementality product returns a CI that excludes 1.0; rollback reverts a
deliberately bad bid in a test; `python -m cli --help` lists every command; `docker build` file present.
