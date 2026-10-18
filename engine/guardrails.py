"""Safety rules that turn proposals into Change records.

* Stock guard: < low_stock_days of stock -> bid x0.5; stock 0 -> pause; restocked -> resume;
  overstocked -> optimizer favours the upper end of its range.
* Waste cleanup: meaningful spend and clicks with 0 conversions, judged with a conversion lag and only when
  zero sales is statistically unlikely -> pause.
* Proven targets: top profit makers can be cut at most protected_max_cut per day, and cuts need approval.
* Limits: +-max_daily_change per day (stock guard and pauses exempt), min/max bid.
* Approval flag: |change| > approval_change_pct, any pause, or a top spender.
"""
from __future__ import annotations

import pandas as pd

from engine.models import Change, Evidence


def limit_bid(old: float, new: float, cfg: dict, daily_limit: bool = True) -> float:
    lo, hi = cfg["bids"]["min_bid"], cfg["bids"]["max_bid"]
    if daily_limit and old > 0:
        m = cfg["bids"]["max_daily_change"]
        new = min(old * (1 + m), max(old * (1 - m), new))
    return round(min(hi, max(lo, new)), 2)


def evidence_for(eid: str, metrics: pd.DataFrame, stock_m: pd.DataFrame, product_of: dict,
                 metrics_prev: pd.DataFrame | None = None) -> Evidence:
    m = metrics.loc[eid] if eid in metrics.index else None
    pid = product_of[eid]
    dos = float(stock_m.loc[pid, "days_of_stock"]) if pid in stock_m.index else 0.0
    if m is None:
        return Evidence(days_of_stock=dos)
    ev = Evidence(spend_7d=float(m.spend_7d), clicks_7d=float(m.clicks_7d), conversions_7d=float(m.conversions_7d),
                  revenue_7d=float(m.revenue_7d), roas_7d=float(m.roas_7d), profit_7d=float(m.profit_7d),
                  acos_7d=float(m.get("acos_7d", 0.0)), cvr_7d=float(m.get("cvr_7d", 0.0)),
                  cpc_7d=float(m.get("cpc_7d", 0.0)), days_of_stock=round(dos, 1))
    if metrics_prev is not None and eid in metrics_prev.index:
        pm = metrics_prev.loc[eid]
        ev.spend_prev_7d, ev.conversions_prev_7d = float(pm.spend_7d), float(pm.conversions_7d)
        ev.roas_prev_7d = float(pm.roas_7d)
        ev.acos_prev_7d = float(pm.get("acos_7d", 0.0))
    return ev


def last_actions(changes: pd.DataFrame) -> dict:
    """entity_id -> (workflow, action) of its most recent applied change."""
    if changes is None or changes.empty:
        return {}
    last = changes.sort_values("day", kind="stable").groupby("entity_id").tail(1)
    return {r.entity_id: (r.workflow, r.action) for r in last.itertuples()}


def stock_guard_proposals(entities: pd.DataFrame, bids: dict, paused: set, stock_m: pd.DataFrame,
                          changes: pd.DataFrame, cfg: dict) -> tuple[dict, set, set]:
    """Returns (proposals {eid: (action, new_bid)}, hold set, favour_high set)."""
    low = cfg["guardrails"]["low_stock_days"]
    over = cfg["guardrails"]["overstock_days"]
    last = last_actions(changes)
    props, hold, favour = {}, set(), set()
    for e in entities.itertuples():
        eid, pid = e.entity_id, e.product_id
        if pid not in stock_m.index:
            continue
        units = float(stock_m.loc[pid, "stock_units"])
        dos = float(stock_m.loc[pid, "days_of_stock"])
        lw = last.get(eid)
        if eid in paused:
            if lw == ("stock_guard", "pause") and units > 0:
                props[eid] = ("resume", bids[eid])
            continue
        if units <= 0:
            props[eid] = ("pause", bids[eid])
        elif dos < low:
            if lw == ("stock_guard", "lower"):
                hold.add(eid)          # already halved: keep it there until restock
            else:
                props[eid] = ("lower", limit_bid(bids[eid], bids[eid] * 0.5, cfg, daily_limit=False))
        elif dos > over:
            favour.add(eid)
    return props, hold, favour


def zero_sale_probability(clicks: float, cvr: float) -> float:
    """Chance of 0 sales in `clicks` clicks if the true conversion rate were `cvr`."""
    return float((1.0 - min(max(cvr, 0.0), 1.0)) ** max(clicks, 0.0))


def waste_proposals(stats: pd.DataFrame, entities: pd.DataFrame, products: pd.DataFrame, paused: set,
                    stock_hist: pd.DataFrame, day: int, cfg: dict) -> dict:
    """Pause entities that spend without selling, but only when the data is strong enough:
    * judged over `waste_window_days`, ignoring the newest `conversion_lag_days` of clicks (they may still convert);
    * spend and clicks over the thresholds, zero conversions;
    * zero sales is statistically unlikely: (1 - category_cvr)^clicks < 1 - waste_confidence;
    * products that were out of stock in the window are skipped (no sales there is a stock problem)."""
    g = cfg["guardrails"]
    win, lag = int(g.get("waste_window_days", 7)), int(g.get("conversion_lag_days", 0))
    conf = float(g.get("waste_confidence", 0.0))
    w = stats[(stats.day > day - win) & (stats.day <= day)]
    settled = w[w.day <= day - lag]
    agg = settled.groupby("entity_id")[["cost", "clicks"]].sum().join(w.groupby("entity_id").conversions.sum())
    ents = entities.merge(products[["product_id", "category"]], on="product_id").set_index("entity_id")
    hist = stats[stats.bid > 0].join(ents.category, on="entity_id").groupby("category")[["clicks", "conversions"]].sum()
    cat_cvr = (hist.conversions / hist.clicks.where(hist.clicks > 0)).fillna(0.0).to_dict()
    sw = stock_hist[(stock_hist.day > day - win) & (stock_hist.day <= day)]
    stocked_out = set(sw[sw.stock_units <= 0].product_id)
    props = {}
    for eid, m in agg.iterrows():
        if eid in paused or ents.loc[eid, "product_id"] in stocked_out:
            continue
        if m.cost < g["waste_min_spend_7d"] or m.clicks < g["waste_min_clicks_7d"] or m.conversions > 0:
            continue
        p0 = zero_sale_probability(m.clicks, cat_cvr.get(ents.loc[eid, "category"], 0.0))
        if p0 < 1.0 - conf:
            props[eid] = ("pause", None)
    return props


def protect_proven(old: float, new: float, cfg: dict) -> float:
    """Proven profit makers may only be cut by protected_max_cut per day."""
    floor = old * (1.0 - float(cfg["guardrails"].get("protected_max_cut", 0.10)))
    return round(max(new, floor), 2)


def flag_approval(change: Change, top: set, cfg: dict, proven: set | None = None) -> Change:
    reasons = []
    if change.action == "pause":
        reasons.append("pause")
    if change.action in ("raise", "lower") and abs(change.change_pct) > cfg["guardrails"]["approval_change_pct"]:
        reasons.append(f"bid change {abs(change.change_pct) * 100:.0f}% > {cfg['guardrails']['approval_change_pct'] * 100:.0f}%")
    if change.entity_id in top and change.action != "hold":
        reasons.append(f"top-{cfg['guardrails']['approval_top_spenders']} spender")
    if proven and change.entity_id in proven and change.action in ("lower", "pause"):
        reasons.append("cut on a proven profit maker")
    change.needs_approval = bool(reasons)
    change.approval_reason = "; ".join(reasons) or None
    return change


def make_change(eid: str, workflow: str, action: str, old: float, new: float, ents: pd.DataFrame,
                evidence: Evidence) -> Change:
    e = ents.loc[eid]
    return Change(entity_id=eid, entity_type=e.entity_type, name=e["name"], workflow=workflow, action=action,
                  old_bid=round(float(old), 2), new_bid=round(float(new), 2), evidence=evidence)
