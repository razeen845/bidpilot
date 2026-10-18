"""Choose bids with math, never with the LLM.

For each entity test current x [0.80 ... 1.20] (step 0.05, clipped to min/max bid) and
pick the bid with the highest expected profit:
    profit(b) = clicks(b) * (cvr * price * margin - b * cpc_ratio)
If total expected spend exceeds the budget, step down the entities with the lowest
expected profit per rupee first.

Cautious raises: bids ABOVE the current one are valued with the pessimistic conversion rate (cvr_low), so a raise
only happens when it pays even if conversion is at the low end of what the data allows. Holding or lowering uses
the normal estimate.

Seller objective "growth" (per product, explicit choice): pick the bid with the most expected sales whose
expected ACOS stays under the product's max_acos; if none qualifies, fall back to the profit objective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MULTS = np.round(np.arange(0.80, 1.2001, 0.05), 2)
FAVOUR_HIGH_TOLERANCE = 0.95   # overstocked: highest bid within 95% of the best profit


def expected(curve: pd.Series | dict, bid: np.ndarray | float):
    b = np.asarray(bid, float)
    clicks = curve["C"] * (1 - np.exp(-curve["k"] * b))
    spend = clicks * b * curve["cpc_ratio"]
    profit = clicks * curve["value_per_click"] - spend
    return clicks, spend, profit


def optimize(curves: pd.DataFrame, bids: dict, cfg: dict, paused: set | None = None,
             favour_high: set | None = None, budget: float | None = None,
             growth: dict | None = None) -> pd.DataFrame:
    lo, hi = cfg["bids"]["min_bid"], cfg["bids"]["max_bid"]
    budget = cfg["store"]["daily_budget"] if budget is None else budget
    paused = paused or set()
    favour_high = favour_high or set()
    growth = growth or {}

    rows, ladders = [], {}
    for eid, cur in bids.items():
        if eid in paused:
            continue
        c = curves.loc[eid]
        if c.status != "ok":
            rows.append({"entity_id": eid, "status": "learning", "old_bid": cur, "new_bid": cur,
                         "exp_clicks": 0.0, "exp_spend": 0.0, "exp_profit": 0.0})
            continue
        cand = np.unique(np.clip(np.round(cur * MULTS, 2), lo, hi))
        clicks, spend, profit = expected(c, cand)
        if "value_per_click_low" in c.index:
            _, _, profit_low = expected({**c.to_dict(), "value_per_click": c["value_per_click_low"]}, cand)
            profit = np.where(cand > cur + 1e-9, profit_low, profit)
        best = int(np.argmax(profit))
        if eid in favour_high:
            ok = np.where(profit >= profit[best] - abs(profit[best]) * (1 - FAVOUR_HIGH_TOLERANCE))[0]
            best = int(ok.max())
        if eid in growth:
            acos = cand * c["cpc_ratio"] / max(c["cvr"] * c["price"], 1e-9)   # expected spend / expected sales
            ok = np.where(acos <= float(growth[eid]))[0]
            if len(ok):
                best = int(ok[np.argmax(clicks[ok])])
        ladders[eid] = (cand, clicks, spend, profit)
        rows.append({"entity_id": eid, "status": "ok", "old_bid": cur, "new_bid": float(cand[best]),
                     "exp_clicks": float(clicks[best]), "exp_spend": float(spend[best]),
                     "exp_profit": float(profit[best]), "_i": best})
    df = pd.DataFrame(rows).set_index("entity_id")

    # budget: step down the worst profit-per-rupee entities until expected spend fits
    total = df.exp_spend.sum()
    guard = 0
    while total > budget and guard < 10_000:
        guard += 1
        movable = df[(df.status == "ok") & (df["_i"] > 0)]
        if movable.empty:
            break
        ppr = movable.exp_profit / movable.exp_spend.where(movable.exp_spend > 0, 1e-9)
        eid = ppr.idxmin()
        cand, clicks, spend, profit = ladders[eid]
        i = int(df.at[eid, "_i"]) - 1
        total += spend[i] - df.at[eid, "exp_spend"]
        df.loc[eid, ["new_bid", "exp_clicks", "exp_spend", "exp_profit", "_i"]] = [cand[i], clicks[i], spend[i], profit[i], i]
    return df.drop(columns=["_i"], errors="ignore")
