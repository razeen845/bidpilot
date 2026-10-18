"""Prepare out/ for the demo: fresh store, 14-day race (+ chart PNG for slides), the 4 preset
"Ask the data" sources, and - when the LLM is reachable - 7 agent days so the day-7 consultant report exists.

    python scripts/prepare_demo.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from agent.analyst import PRESET_QUESTIONS, REAL_ADS_QUESTIONS, ask  # noqa: E402
from agent.llm import make_llm  # noqa: E402
from agent.orchestrator import run_agent_day  # noqa: E402
from analysis.real_ads import with_real_ads  # noqa: E402
from data.generate import generate  # noqa: E402
from settings import load_config, out_dir  # noqa: E402
from sim.platform import AdPlatform  # noqa: E402
from sim.runner import compare, summarize  # noqa: E402


def race_chart(race: pd.DataFrame, path: Path, col: str = "cum_total_profit",
               title: str = "14-day total contribution profit (simulation)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"fixed": "#9aa5b1", "rules": "#d69e2e", "agent_nollm": "#2b6cb0", "agent": "#2b6cb0"}
    for s, g in race.groupby("strategy", sort=False):
        ax.plot(g.day, g[col], marker="o", label=s, color=colors.get(s), linewidth=2.5 if "agent" in s else 1.8)
        ax.annotate(f"Rs {g[col].iloc[-1]:,.0f}", (g.day.iloc[-1], g[col].iloc[-1]),
                    textcoords="offset points", xytext=(6, 0), va="center", fontsize=9)
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("Cumulative profit (Rs)")
    ax.set_title(f"{title}: fixed vs rules vs BidPilot agent")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "uci_online_retail"], default=None,
                    help="store data (default: config.yaml)")
    ap.add_argument("--ai-days", type=int, default=7,
                    help="how many of the 7 live days use the LLM (earlier ones run without it; bids are identical)")
    ap.add_argument("--race-llm", action="store_true", help="run the 14-day race with the LLM agent (slow on a laptop)")
    args = ap.parse_args()
    cfg = load_config()
    if args.source:
        cfg = {**cfg, "store": {**cfg["store"], "source": args.source}}
    out = out_dir()
    for pattern in ["objectives.json", "day_*_*.json", "analysis/D*.json", "consult/day_*", "race*", "todos.json",
                    "pending_consultant_actions.json", "experiments.json", "incrementality.json"]:
        for f in out.glob(pattern):
            f.unlink()
    generate(cfg, out)
    llm = make_llm(cfg, out)
    print("LLM:", "connected" if llm else "not reachable -> no-llm mode (consultant skipped)")

    strategies = ["fixed", "rules", "agent" if (llm and args.race_llm) else "agent_nollm"]
    race = compare(strategies, cfg=cfg, out=out, llm=llm if args.race_llm else None)
    for s in strategies:
        print(summarize(race[race.strategy == s]))
    race_chart(race, out / "race_chart.png")
    race_chart(race, out / "race_chart_attributed.png", "cum_profit", "14-day ad-attributed contribution profit")

    data = with_real_ads(AdPlatform(cfg, out).public_data())
    for q in PRESET_QUESTIONS + (REAL_ADS_QUESTIONS if "real_ads" in data else []):
        src = ask(q, data, cfg, llm=llm, out=out)
        print(f"[{src.get('id')}] {q}\n    {src.get('answer', src.get('error'))}  (code by {src.get('code_author')})")

    if llm:
        p = AdPlatform(cfg, out, persist=True)
        for d in range(1, 8):
            use = llm if d > 7 - args.ai_days else None
            res = run_agent_day(p, auto_approve=True, cfg=cfg, llm=use, out=out)
            n_llm = sum(c["reason_source"] == "llm" for c in res["plan"]["changes"])
            print(f"day {d}: {len(res['plan']['changes'])} changes, {n_llm} explained by the LLM; {res['report'][:120]}")
        print("consultant report:", out / "consult" / "day_7.md")
    print("race chart:", out / "race_chart.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
