"""Speed test on the AMD GPU (Phase 10).

Scenarios (median of N runs each):
  bidding / naive      one LLM call per entity, sequential; the LLM picks bid + reason
  bidding / bidpilot   optimizer + guardrails + 1 planner call + concurrent explainers
  consultant / sequential   analysis questions and research one after another (research in cache mode)
  consultant / parallel     the same with asyncio.gather

Records wall time, LLM calls, tokens, tokens/s (from out/llm_log.jsonl) and the GPU name.

    LLM_BASE_URL=http://<amd-box>:8000/v1 GPU_NAME="AMD Instinct MI300X" python bench/latency.py
    python bench/latency.py --dry-run     # fake in-process server, writes bench/results_dryrun.json only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel  # noqa: E402

from agent.analyst import PRESET_QUESTIONS  # noqa: E402
from agent.consultant import fan_out  # noqa: E402
from agent.llm import LLMClient  # noqa: E402
from agent.orchestrator import prepare_day_async  # noqa: E402
from analysis.metrics import entity_metrics  # noqa: E402
from settings import load_config, out_dir  # noqa: E402
from sim.platform import AdPlatform  # noqa: E402

BENCH_QUESTIONS = PRESET_QUESTIONS[:3]
BENCH_QUERIES = ["home textile online sales seasonality", "shopping ads bidding low inventory best practice"]


class NaiveBidOut(BaseModel):   # benchmark only: the product never lets the LLM pick bids
    bid: float
    reason: str


NAIVE_SYSTEM = ("You are a bid manager. Given one ad's last 7 days, its current bid, price and margin, "
                "choose tomorrow's bid in Rs and give a one-sentence reason. Return JSON {\"bid\": number, \"reason\": text}.")


def gpu_name() -> str:
    import os
    if os.environ.get("GPU_NAME"):
        return os.environ["GPU_NAME"]
    for cmd in (["rocm-smi", "--showproductname"], ["amd-smi", "static", "--asic"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                if "Card series" in line or "MARKET_NAME" in line or "Card SKU" in line:
                    return line.split(":")[-1].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return "unknown (set GPU_NAME)"


async def naive_bidding(platform, cfg, llm, sample: int | None = None):
    m = entity_metrics(platform.get_stats(), platform.day, 7, platform.get_entities())
    ents = platform.get_entities().merge(platform.get_products()[["product_id", "price", "margin_pct"]], on="product_id")
    if sample:          # every 4th ad etc.: a fixed, spread-out sample (same ads in every run)
        ents = ents.iloc[:: max(1, len(ents) // sample)].head(sample)
    bids = platform.get_bids()
    for e in ents.itertuples():
        row = m.loc[e.entity_id].to_dict()
        user = json.dumps({"name": e.name, "current_bid": bids[e.entity_id], "price": e.price,
                           "margin_pct": e.margin_pct, "last_7_days": row})
        try:
            await llm.chat_json("naive_bidder", NAIVE_SYSTEM, user, NaiveBidOut, max_tokens=120)
        except Exception:  # noqa: BLE001
            pass


async def bidpilot_bidding(platform, cfg, llm, out):
    await prepare_day_async(platform, cfg, llm, None, out, save=False)


def log_stats(log: Path, tag: str) -> dict:
    recs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    recs = [r for r in recs if r.get("tag") == tag]
    return {"llm_calls": len(recs), "failed_calls": sum(not r["ok"] for r in recs),
            "tokens_in": sum(r.get("tokens_in", 0) for r in recs), "tokens_out": sum(r.get("tokens_out", 0) for r in recs),
            "llm_seconds_sum": round(sum(r.get("seconds", 0) for r in recs), 3)}


def measure(name, variant, fn, llm, log, runs):
    samples = []
    for i in range(runs):
        tag = f"{name}/{variant}/{i}"
        llm.run_tag = tag
        t = time.perf_counter()
        asyncio.run(fn())
        wall = time.perf_counter() - t
        s = log_stats(log, tag)
        s["wall_s"] = round(wall, 3)
        s["tokens_per_s"] = round(s["tokens_out"] / wall, 1) if wall > 0 else 0.0
        samples.append(s)
        print(f"  {name:<10} {variant:<10} run {i + 1}: {wall:.2f}s, {s['llm_calls']} calls, {s['tokens_out']} tokens out")
    med = {k: statistics.median(x[k] for x in samples) for k in samples[0]}
    return {"scenario": name, "variant": variant, **med, "samples": samples}


def fake_transport(delay: float):
    """Dry-run only: an in-process fake OpenAI server with fixed latency."""
    import httpx

    from agent.analyst import PRESET_CODE

    async def handler(request):
        body = json.loads(request.content)
        await asyncio.sleep(delay)
        sysmsg, user = body["messages"][0]["content"], body["messages"][1]["content"]
        if "bid manager. Given one ad" in sysmsg:
            content = {"bid": 10.0, "reason": "ok"}
        elif "daily planner" in sysmsg:
            content = {"focus": [], "hold": [], "hold_reasons": {}, "day_note": "ok"}
        elif "explain one bid change" in sysmsg:
            content = {"reason": "Following the 7 day numbers."}
        elif "data analyst" in sysmsg:
            q = user.split("Question: ", 1)[1].split("\n", 1)[0]
            content = {"code": PRESET_CODE.get(q, "result = 1")}
        elif "1-3 sentences" in sysmsg:
            content = {"answer": "See the table."}
        else:
            content = {}
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "dry-run",
                                         "choices": [{"index": 0, "finish_reason": "stop", "message": {
                                             "role": "assistant", "content": json.dumps(content)}}],
                                         "usage": {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340}})
    return httpx.MockTransport(handler)


def chart(results: list[dict], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, scen in zip(axes, ["bidding", "consultant"]):
        rows = [r for r in results if r["scenario"] == scen]
        bars = ax.bar([r["variant"] for r in rows], [r["wall_s"] for r in rows], color=["#9aa5b1", "#2b6cb0"])
        ax.bar_label(bars, labels=[f"{r['wall_s']:.1f}s" for r in rows])
        ax.set_title(scen)
        ax.set_ylabel("wall time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="fake server with fixed latency; never writes results.json")
    ap.add_argument("--fake-latency", type=float, default=0.05)
    ap.add_argument("--naive-sample", type=int, default=0,
                    help="time the naive one-call-per-ad approach on N ads and scale to all ads (calls are strictly "
                         "sequential, so time grows linearly); 0 = all ads")
    ap.add_argument("--label", default="", help="free text stored with the results, e.g. the machine")
    args = ap.parse_args()

    cfg = load_config(overrides={"research": {"mode": "cache"}})
    main_out = out_dir()
    work = Path(tempfile.mkdtemp(prefix="bidpilot_bench_"))
    for f in ["products.csv", "entities.csv", "history.parquet", "stock_history.parquet", "_hidden_market.json"]:
        if not (main_out / f).exists():
            from data.generate import generate
            generate(cfg, main_out)
        shutil.copy(main_out / f, work / f)

    if args.dry_run:
        llm = LLMClient(cfg, work, transport=fake_transport(args.fake_latency))
        model, gpu = "dry-run fake server", f"none (fake latency {args.fake_latency}s per call)"
    else:
        llm = LLMClient(cfg, work)
        if not llm.available():
            print(f"LLM endpoint {cfg['llm']['base_url']} not reachable. Start vLLM on the AMD box or use --dry-run.")
            return 1
        model, gpu = cfg["llm"]["model_planner"], gpu_name()
    log = work / "llm_log.jsonl"

    def plat():
        return AdPlatform(cfg, work)

    data = plat().public_data()
    results = [
        measure("bidding", "naive", lambda: naive_bidding(plat(), cfg, llm, args.naive_sample or None), llm, log,
                args.runs),
        measure("bidding", "bidpilot", lambda: bidpilot_bidding(plat(), cfg, llm, work), llm, log, args.runs),
        measure("consultant", "sequential",
                lambda: fan_out(BENCH_QUESTIONS, BENCH_QUERIES, data, cfg, llm, work, parallel=False), llm, log, args.runs),
        measure("consultant", "parallel",
                lambda: fan_out(BENCH_QUESTIONS, BENCH_QUERIES, data, cfg, llm, work, parallel=True), llm, log, args.runs),
    ]
    n_ads = len(plat().get_entities())
    if args.naive_sample and args.naive_sample < n_ads:
        naive = results[0]
        factor = n_ads / args.naive_sample
        naive["measured_ads"] = args.naive_sample
        naive["measured_wall_s"] = naive["wall_s"]
        for k in ("wall_s", "llm_calls", "tokens_in", "tokens_out", "llm_seconds_sum"):
            naive[k] = round(naive[k] * factor, 3)
        naive["note"] = f"measured on {args.naive_sample} of {n_ads} ads, scaled x{factor:g} (sequential calls)"
    by = {(r["scenario"], r["variant"]): r for r in results}
    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": args.dry_run,
        "gpu": gpu,
        "model": model,
        "base_url": cfg["llm"]["base_url"],
        "runs": args.runs,
        "label": args.label,
        "ads": n_ads,
        "speedup_bidding": round(by[("bidding", "naive")]["wall_s"] / by[("bidding", "bidpilot")]["wall_s"], 2),
        "speedup_consultant": round(by[("consultant", "sequential")]["wall_s"]
                                    / by[("consultant", "parallel")]["wall_s"], 2),
        "results": [{k: v for k, v in r.items() if k != "samples"} for r in results],
        "samples": {f"{r['scenario']}/{r['variant']}": r["samples"] for r in results},
    }
    name = "results_dryrun.json" if args.dry_run else "results.json"
    (ROOT / "bench" / name).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    chart(payload["results"], main_out / ("speed_chart_dryrun.png" if args.dry_run else "speed_chart.png"),
          f"{model} on {gpu}" + (" (DRY RUN - not real)" if args.dry_run else ""))
    print(json.dumps({k: payload[k] for k in ("gpu", "model", "speedup_bidding", "speedup_consultant")}, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
