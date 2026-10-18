"""Run N simulated days for a strategy on a fresh copy of the store.

python -m sim.runner --strategy fixed|rules|agent_nollm|agent
python -m sim.runner --compare
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from settings import load_config, out_dir, write_json
from sim.platform import AdPlatform


def make_strategy(name: str, cfg: dict, out: str | Path | None = None, llm=None):
    if name == "fixed":
        from engine.baselines import FixedStrategy
        return FixedStrategy(cfg)
    if name == "rules":
        from engine.baselines import RulesStrategy
        return RulesStrategy(cfg)
    if name == "agent_nollm":
        from agent.orchestrator import AgentStrategy
        return AgentStrategy(cfg, out=out, llm=None)
    if name == "agent":
        from agent.orchestrator import AgentStrategy
        return AgentStrategy(cfg, out=out, llm=llm, use_llm=True)
    raise ValueError(f"unknown strategy {name}")


def run(strategy: str, days: int | None = None, cfg: dict | None = None, out: str | Path | None = None,
        llm=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    days = days or cfg["store"]["sim_days"]
    platform = AdPlatform(cfg, out, persist=False)
    strat = make_strategy(strategy, cfg, out, llm)
    rows = []
    for d in range(1, days + 1):
        strat.act(platform, d)
        rows.append(platform.step())
        if hasattr(strat, "after_step"):
            strat.after_step(platform, d)
    df = pd.DataFrame(rows)
    df["cum_profit"] = df.profit.cumsum().round(2)
    df["cum_total_profit"] = df.total_contribution_profit.cumsum().round(2)
    df.insert(0, "strategy", strategy)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Fair outcome report: whole-business results, not just ad-attributed ones."""
    spend, rev = float(df.spend.sum()), float(df.revenue.sum())
    return {
        "strategy": df.strategy.iloc[0],
        "profit": round(float(df.profit.sum()), 2),                       # attributed contribution - ad spend
        "total_contribution_profit": round(float(df.total_contribution_profit.sum()), 2),
        "total_sales": round(float(df.total_sales.sum()), 2),            # ads + organic
        "attributed_sales": round(rev, 2),
        "spend": round(spend, 2),
        "acos": round(spend / rev, 4) if rev else 0.0,
        "roas": round(rev / spend, 2) if spend else 0.0,
        "cvr": round(float(df.conversions.sum()) / float(df.clicks.sum()), 4) if df.clicks.sum() else 0.0,
        "stock_availability": round(float(df.products_in_stock.sum()) / float(df.products.sum()), 4),
        "stockout_ad_spend": round(float(df.stockout_ad_spend.sum()), 2),
        "revenue": round(rev, 2),
    }


def compare(strategies: list[str], cfg: dict | None = None, out: str | Path | None = None, llm=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    frames = [run(s, cfg=cfg, out=out, llm=llm) for s in strategies]
    race = pd.concat(frames, ignore_index=True)
    o = out_dir(out)
    race.to_csv(o / "race.csv", index=False)
    write_json(o / "race_summary.json", [summarize(f) for f in frames])
    return race


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="fixed")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--llm", action="store_true", help="include the LLM 'agent' strategy in --compare")
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()
    pd.set_option("display.width", 160)
    if args.compare:
        strategies = ["fixed", "rules", "agent_nollm"] + (["agent"] if args.llm else [])
        race = compare(strategies)
        summary = pd.DataFrame([summarize(race[race.strategy == s]) for s in strategies])
        print(summary.to_string(index=False))
    else:
        df = run(args.strategy, days=args.days)
        print(df.to_string(index=False))
        print(summarize(df))


if __name__ == "__main__":
    main()
