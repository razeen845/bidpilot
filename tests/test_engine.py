import time

import numpy as np
import pandas as pd
import pytest

from engine import guardrails as G
from engine.curves import fit_clicks
from engine.models import Change, Evidence
from engine.optimizer import optimize
from sim.runner import run, summarize


def test_curve_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    bids = rng.uniform(5, 40, 200)
    clicks = rng.poisson(40 * (1 - np.exp(-0.08 * bids)))
    C, k, ok = fit_clicks(bids, clicks)
    assert ok
    assert abs(C - 40) / 40 < 0.2
    assert abs(k - 0.08) / 0.08 < 0.3


def _curves(n=5, C=50.0, k=0.08, vpc=40.0):
    return pd.DataFrame({"status": "ok", "C": C, "k": k, "cvr": 0.03, "cpc_ratio": 0.75, "price": 2000,
                         "margin_pct": 0.5, "value_per_click": vpc}, index=[f"K{i:03d}" for i in range(1, n + 1)])


def test_optimizer_stays_in_daily_range_and_fast(cfg):
    curves = _curves(160)
    bids = {eid: 10.0 for eid in curves.index}
    t = time.time()
    out = optimize(curves, bids, cfg, budget=1e9)
    assert time.time() - t < 1.0
    assert out.new_bid.between(8.0 - 1e-9, 12.0 + 1e-9).all()


def test_optimizer_moves_toward_profit(cfg):
    curves = _curves(2)
    curves.loc["K002", "value_per_click"] = 2.0  # losing money -> lower
    out = optimize(curves, {"K001": 5.0, "K002": 30.0}, cfg, budget=1e9)
    assert out.loc["K001", "new_bid"] > 5.0
    assert out.loc["K002", "new_bid"] < 30.0


def test_optimizer_respects_budget(cfg):
    curves = _curves(20, C=100.0, vpc=200.0)
    curves.loc["K001":"K010", "value_per_click"] = 60.0   # weaker half should be cut first
    bids = {eid: 30.0 for eid in curves.index}
    free = optimize(curves, bids, cfg, budget=1e9).exp_spend.sum()
    budget = 0.9 * free
    out = optimize(curves, bids, cfg, budget=budget)
    assert out.exp_spend.sum() <= budget + 1e-6
    assert out.loc["K001":"K010", "new_bid"].mean() < out.loc["K011":"K020", "new_bid"].mean()
    # infeasible within one day's range: every entity is cut to the bottom of its range
    out = optimize(curves, {eid: 50.0 for eid in curves.index}, cfg, budget=10_000)
    assert (out.new_bid == 40.0).all()


def test_learning_entities_hold(cfg):
    curves = _curves(2)
    curves.loc["K002", "status"] = "learning"
    out = optimize(curves, {"K001": 10.0, "K002": 10.0}, cfg, budget=1e9)
    assert out.loc["K002", "new_bid"] == 10.0


def test_limit_bid(cfg):
    assert G.limit_bid(10, 20, cfg) == 12.0
    assert G.limit_bid(10, 1, cfg) == 8.0
    assert G.limit_bid(10, 5, cfg, daily_limit=False) == 5.0
    assert G.limit_bid(4, 1, cfg, daily_limit=False) == cfg["bids"]["min_bid"]


def test_stock_guard_rules(cfg):
    ents = pd.DataFrame({"entity_id": ["A", "B", "C", "D"], "product_id": ["p1", "p2", "p3", "p4"]})
    stock_m = pd.DataFrame({"stock_units": [0, 2, 500, 10], "days_of_stock": [0, 1.0, 120, 10]},
                           index=["p1", "p2", "p3", "p4"])
    changes = pd.DataFrame([{"day": 1, "entity_id": "D", "workflow": "stock_guard", "action": "pause"}])
    props, hold, favour = G.stock_guard_proposals(ents, {"A": 20, "B": 20, "C": 20, "D": 20}, {"D"}, stock_m,
                                                  changes, cfg)
    assert props["A"][0] == "pause"
    assert props["B"] == ("lower", 10.0)
    assert "C" in favour
    assert props["D"][0] == "resume"
    # already halved -> hold, not halved again
    ch = pd.DataFrame([{"day": 1, "entity_id": "B", "workflow": "stock_guard", "action": "lower"}])
    props, hold, _ = G.stock_guard_proposals(ents, {"A": 20, "B": 10, "C": 20, "D": 20}, set(), stock_m, ch, cfg)
    assert "B" in hold and "B" not in props


def _waste_fixture():
    """A: 15 clicks/day, 0 sales (clear waste). B: same, but a sale on the newest day (conversion lag window).
    C: big spend but only 2 clicks/day (too little data). E: like A but its product ran out of stock.
    D: a healthy entity that sets the category conversion rate (4%)."""
    ents = pd.DataFrame({"entity_id": list("ABCDE"), "product_id": ["pa", "pb", "pc", "pd", "pe"],
                         "entity_type": "keyword", "name": list("abcde")})
    prods = pd.DataFrame({"product_id": ["pa", "pb", "pc", "pd", "pe"], "category": "X"})
    rows = []
    for day in range(-13, 1):
        for eid, clicks, cost, conv in [("A", 15, 150, 0), ("B", 15, 150, 1 if day == 0 else 0),
                                        ("C", 2, 150, 0), ("D", 500, 1000, 20), ("E", 15, 150, 0)]:
            rows.append({"day": day, "entity_id": eid, "bid": 10.0, "clicks": clicks, "cost": cost, "conversions": conv})
    stats = pd.DataFrame(rows)
    stock = pd.DataFrame([{"day": d, "product_id": pid, "stock_units": 0 if (pid == "pe" and d == -3) else 50}
                          for d in range(-13, 1) for pid in prods.product_id])
    return stats, ents, prods, stock


def test_waste_pause_needs_strong_evidence(cfg):
    from engine.guardrails import zero_sale_probability
    stats, ents, prods, stock = _waste_fixture()
    props = G.waste_proposals(stats, ents, prods, set(), stock, 0, cfg)
    assert set(props) == {"A"}
    assert zero_sale_probability(26, 0.04) > 0.05          # C: 0 sales in 26 clicks can be bad luck
    assert zero_sale_probability(195, 0.04) < 0.05         # A: 0 sales in 195 clicks is not


def test_approval_flags(cfg):
    def ch(action, old, new, eid="X"):
        return Change(entity_id=eid, entity_type="keyword", name="x", workflow="bid_adjust", action=action,
                      old_bid=old, new_bid=new, evidence=Evidence())
    assert G.flag_approval(ch("raise", 10, 12), set(), cfg).needs_approval
    assert not G.flag_approval(ch("raise", 10, 10.5), set(), cfg).needs_approval
    assert G.flag_approval(ch("raise", 10, 10.5), {"X"}, cfg).needs_approval
    p = G.flag_approval(ch("pause", 10, 10), set(), cfg)
    assert p.needs_approval and "pause" in p.approval_reason


def test_agent_nollm_beats_baselines(cfg, store):
    res = {s: summarize(run(s, cfg=cfg, out=store)) for s in ["fixed", "rules", "agent_nollm"]}
    assert res["agent_nollm"]["profit"] > res["rules"]["profit"] > 0
    assert res["agent_nollm"]["profit"] > res["fixed"]["profit"]
    assert res["agent_nollm"]["stockout_ad_spend"] < res["rules"]["stockout_ad_spend"]


def test_cautious_raise_ignores_one_lucky_sale(cfg):
    """Raises are valued with the pessimistic conversion rate: a click worth Rs 80 on average but only Rs 15 in
    the pessimistic case must not get a higher bid, although the average alone would raise it."""
    curves = pd.DataFrame({"status": "ok", "C": 60.0, "k": 0.02, "cvr": 0.0125, "cpc_ratio": 0.7, "price": 5430.0,
                           "margin_pct": 0.587, "value_per_click": 80.0, "value_per_click_low": 15.0},
                          index=["K001"])
    average_only = optimize(curves.drop(columns=["value_per_click_low"]), {"K001": 30.0}, cfg, budget=1e9)
    assert average_only.loc["K001", "new_bid"] > 30.0          # the average alone says "raise"
    cautious = optimize(curves, {"K001": 30.0}, cfg, budget=1e9)
    assert cautious.loc["K001", "new_bid"] <= 30.0              # the pessimistic case says "don't"


def test_cautious_rate_comes_from_the_data(cfg):
    """One lucky sale on 180 clicks: the pessimistic rate sits well below the shrunk estimate."""
    from engine.curves import fit_all
    rows = [{"day": d, "entity_id": "K001", "bid": 30.0 * (1 + 0.1 * ((d % 3) - 1)), "clicks": 6, "cost": 120.0,
             "conversions": 1 if d == -5 else 0, "revenue": 5430.0 if d == -5 else 0.0, "profit": 0.0}
            for d in range(-29, 1)]
    rows += [{"day": d, "entity_id": "K002", "bid": 20.0, "clicks": 400, "cost": 6000.0, "conversions": 16,
              "revenue": 16 * 5430.0, "profit": 0.0} for d in range(-29, 1)]
    ents = pd.DataFrame({"entity_id": ["K001", "K002"], "entity_type": "keyword", "name": ["cheap x", "x"],
                         "product_id": "P01"})
    prods = pd.DataFrame({"product_id": ["P01"], "category": ["C"], "price": [5430.0], "margin_pct": [0.587]})
    c = fit_all(pd.DataFrame(rows), ents, prods, cfg).loc["K001"]
    assert c.cvr_low < 0.6 * c.cvr
    assert c.value_per_click_low == pytest.approx(c.cvr_low * 5430.0 * 0.587)
