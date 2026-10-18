# BidPilot MVP

The minimum product a small online seller (or a hackathon judge) can run end to end today.

## What the MVP does

| Capability | Where | Works without the LLM |
|---|---|---|
| Daily bidding on 160 product + keyword ads: optimizer, stock guard, waste cleanup, rollback, approvals | Run a day tab, `cli day` | Yes (template explanations) |
| LLM day plan, per-change explanations (number-checked), day report | Run a day tab | Needs LLM |
| 14-day race vs "do nothing" and simple rules, fair whole-business report | 14-day race tab, `cli race` | Yes |
| Ask the data: code written, policy-checked, run in a sandbox, cited `[D#]` | Ask the data tab, `cli ask` | 6 preset questions |
| Weekly consultant: ranked, cited recommendations; accept/reject; owner to-dos | Consultant tab, `cli consult` | Needs LLM |
| Web facts with exact verified quotes `[W#]` (offline cache) | Consultant citations | Yes (cache) |
| Incrementality switchback tests (owner-approved) | Experiments tab, `cli experiment` | Yes |
| Seller objectives (growth push with ACOS cap), platform rule profiles | Store tab, `config.yaml` | Yes |
| Speed test on the AMD GPU | Speed tab, `cli bench` | Needs LLM on AMD |

Out of scope for the MVP: real ad accounts (Google/Amazon/Meta/Daraz), ad copy, audience/location/device bids,
open-ended chat, user accounts.

## Run it

**Local (any OS, Python 3.11+):**
```bash
pip install -r requirements.txt
python -m cli demo                     # fresh store, race, 6 preset analyses, then the dashboard on :8501
```

**Docker:**
```bash
docker compose up app                  # http://localhost:8501
```

**On the AMD box (makes the LLM parts live):**
```bash
docker compose --profile amd up        # vLLM on ROCm serving Qwen/Qwen2.5-32B-Instruct + the dashboard
# or point at an existing server:
export LLM_BASE_URL=http://<amd-box>:8000/v1
python -m cli check-llm                # 1 chat + 1 JSON-schema request; prints latency and JSON mode
python scripts/prepare_demo.py         # race with the LLM agent + day-7 consultant report
GPU_NAME="AMD Instinct MI300X" python -m cli bench   # writes bench/results.json + out/speed_chart.png
```

## 4-minute demo script

1. **Store** (20 s): 40 products, 120 keywords, low-stock rows highlighted, contribution margin after fees/returns.
2. **Run a day** (40 s): "N changes, M need your approval". Read one reason (this week vs last week). Approve one,
   reject one, apply.
3. **14-day race** (30 s): total contribution profit view. Agent ahead; it sells a bit less but earns more and never
   advertises out-of-stock products. Say "simulated account".
4. **Consultant** (60 s): ranked action list. Click a `[D#]` chip (question, code, table), click a `[W#]` chip
   (link, exact quote, retrieval date). Accept the #1 recommendation.
5. **Experiments** (20 s): the best-seller's switchback test: "only about a third of its ad sales are extra".
6. **Ask the data** (20 s): "Which products get many clicks but convert below their category average?"
7. **Speed** (20 s): naive one-call-per-ad vs BidPilot on the AMD GPU.
8. Close (10 s): practice account today; read-only Google Ads connection next.

## Acceptance checks

| Check | Command | Expected |
|---|---|---|
| Quality gate | `ruff check . && pytest` | all pass |
| Config | `python -m cli validate-config` | `"ok": true` |
| Agent wins | `python -m cli race` | agent_nollm highest total and attributed profit, 0 stockout ad spend |
| Human in the loop | `python -m cli day` → `decide` → `day --apply` | only decided approval rows applied |
| No-LLM safety | `python -m cli --no-llm day --auto --days 3` | runs, template reasons |
| Sandbox | `pytest tests/test_sandbox.py` | `import os`, `open()`, `__import__` blocked; timeout handled |
| Citations | `pytest tests/test_consultant.py tests/test_research.py` | invented numbers/quotes removed |
| Incrementality | `pytest tests/test_phase13.py -k lifecycle` | P13 CI below 1.0, factor stored |
| Offline demo | Wi-Fi off, `python -m cli demo` | works (research from cache) |
| AMD (on the box) | `python -m cli check-llm`, `python -m cli bench` | valid JSON; real timings in `bench/results.json` |

## Known limitations

See "Honest notes and limitations" in [README.md](README.md). The two that matter most for judges: the speed
numbers and the consultant report must be produced on the AMD box, and all business results are from a
simulated account.
