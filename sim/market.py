"""Hidden market: the true response of each ad to its bid.

Only `sim/` and `data/` may import this module. The agent (agent/, engine/,
analysis/, sandbox/) must learn these curves from the data, like a real bid
manager would.

Randomness uses common random numbers: every draw comes from an RNG seeded by
(seed, day, entity_id), so all strategies see the same luck on the same day.
"""
from __future__ import annotations

import math
import re

import numpy as np

DAY_OFFSET = 10_000  # keeps history days (<= 0) non-negative inside the RNG seed


def entity_index(entity_id: str) -> int:
    """Stable integer for an id: P07 -> 7, K012 -> 1012."""
    n = int(re.sub(r"\D", "", entity_id))
    return n if entity_id.startswith("P") else 1000 + n


def rng_for(seed: int, day: int, entity_id: str, stream: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, day + DAY_OFFSET, entity_index(entity_id), stream])


def expected_clicks(params: dict, bid: float) -> float:
    if bid <= 0:
        return 0.0
    return params["click_cap"] * (1.0 - math.exp(-params["k"] * bid))


def optimal_bid(params: dict, value_per_click: float, lo: float = 3, hi: float = 80) -> float:
    """True profit-maximising bid (used only by the generator to place stories)."""
    grid = np.linspace(lo, hi, 773)
    clicks = params["click_cap"] * (1 - np.exp(-params["k"] * grid))
    profit = clicks * (value_per_click - grid * params["cpc_ratio"])
    return float(grid[int(np.argmax(profit))])


def simulate_day(
    *,
    day: int,
    seed: int,
    entities: dict,      # entity_id -> {"product_id", ...}
    params: dict,        # entity_id -> {"click_cap", "k", "cvr", "cpc_ratio"}
    products: dict,      # product_id -> {"price", "margin_pct", "organic_sales_per_day", "restock_day", "restock_units",
                         #                "incrementality" (optional, default 1.0)}
    bids: dict,          # entity_id -> effective bid (0 = paused)
    stock: dict | None,  # product_id -> units at end of previous day; None = unlimited (history)
    budget: float,
) -> tuple[list[dict], list[dict], dict]:
    """Simulate one day. Returns (stat_rows, stock_rows, new_stock)."""
    ids = sorted(entities)

    # 1. restocks arrive at the start of the day
    restocked = {}
    if stock is not None:
        stock = dict(stock)
        for pid, p in products.items():
            if int(p.get("restock_day", -1)) == day:
                stock[pid] = stock.get(pid, 0) + int(p["restock_units"])
                restocked[pid] = int(p["restock_units"])

    # 2. clicks (Poisson) and cost
    rngs = {eid: rng_for(seed, day, eid) for eid in ids}
    clicks = {}
    for eid in ids:
        b = float(bids.get(eid, 0.0))
        lam = expected_clicks(params[eid], b)
        clicks[eid] = int(rngs[eid].poisson(lam)) if lam > 0 else 0
    cost = {eid: clicks[eid] * bids.get(eid, 0.0) * params[eid]["cpc_ratio"] for eid in ids}

    # 3. daily budget: scale clicks down proportionally, flooring guarantees cost <= budget
    total_cost = sum(cost.values())
    if total_cost > budget > 0:
        f = budget / total_cost
        for eid in ids:
            clicks[eid] = int(math.floor(clicks[eid] * f))
            cost[eid] = clicks[eid] * bids.get(eid, 0.0) * params[eid]["cpc_ratio"]

    # 4. conversions (Binomial), before stock capping
    conv = {eid: int(rngs[eid].binomial(clicks[eid], params[eid]["cvr"])) if clicks[eid] > 0 else 0 for eid in ids}

    # 5. organic sales come first, then ad sales take what is left
    organic = {}
    for pid, p in products.items():
        lam = float((p.get("organic_series") or {}).get(str(day), p["organic_sales_per_day"]))
        o = int(rng_for(seed, day, pid, stream=1).poisson(lam))
        if stock is not None:
            o = min(o, max(0, stock.get(pid, 0)))
        organic[pid] = o

    if stock is not None:
        by_product: dict[str, list[str]] = {}
        for eid in ids:
            by_product.setdefault(entities[eid]["product_id"], []).append(eid)
        for pid, eids in by_product.items():
            avail = max(0, stock.get(pid, 0) - organic[pid])
            want = sum(conv[e] for e in eids)
            if want > avail:
                scaled = {e: (conv[e] * avail) // want for e in eids}
                left = avail - sum(scaled.values())
                for e in sorted(eids, key=lambda e: (-(conv[e] * avail % want), e)):
                    if left <= 0:
                        break
                    if scaled[e] < conv[e]:
                        scaled[e] += 1
                        left -= 1
                conv.update(scaled)

    # 5b. incrementality: part of the ad sales would have happened organically anyway (cannibalization).
    #     The platform still attributes every ad sale to the ad, like real ad platforms do.
    ad_by_product: dict[str, int] = {}
    for eid in ids:
        pid = entities[eid]["product_id"]
        ad_by_product[pid] = ad_by_product.get(pid, 0) + conv[eid]
    for pid, ad_units in ad_by_product.items():
        inc = float(products[pid].get("incrementality", 1.0))
        if ad_units > 0 and inc < 1.0 and organic[pid] > 0:
            cannibal = int(rng_for(seed, day, pid, stream=2).binomial(ad_units, 1.0 - inc))
            organic[pid] = max(0, organic[pid] - cannibal)

    # 6. revenue / profit rows
    rows = []
    sold = {pid: organic[pid] for pid in products}
    for eid in ids:
        pid = entities[eid]["product_id"]
        p = products[pid]
        revenue = conv[eid] * p["price"]
        sold[pid] += conv[eid]
        rows.append({
            "day": day,
            "entity_id": eid,
            "bid": round(float(bids.get(eid, 0.0)), 2),
            "clicks": clicks[eid],
            "cost": round(cost[eid], 2),
            "conversions": conv[eid],
            "revenue": round(revenue, 2),
            "profit": round(revenue * p["margin_pct"] - cost[eid], 2),
        })

    stock_rows = []
    new_stock = None
    if stock is not None:
        new_stock = {}
        for pid in products:
            new_stock[pid] = max(0, stock.get(pid, 0) - sold[pid])
            stock_rows.append({
                "day": day,
                "product_id": pid,
                "stock_units": new_stock[pid],
                "units_sold": sold[pid],
                "restocked_units": restocked.get(pid, 0),
            })
    else:
        for pid in products:
            stock_rows.append({"day": day, "product_id": pid, "units_sold": sold[pid], "restocked_units": 0})
    return rows, stock_rows, new_stock
