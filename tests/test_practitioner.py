"""Phase 12: practitioner upgrades from the marketing strategist's review."""
import json

import numpy as np
import pandas as pd
import pytest

from agent.analyst import PRESET_QUESTIONS, ask
from agent.explain import change_values, check_numbers, template_reason
from agent.workflows import plan_day
from analysis.metrics import entity_metrics
from engine.models import Change, Evidence
from engine.optimizer import optimize
from settings import load_config
from sim.platform import AdPlatform
from sim.runner import make_strategy, run, summarize


def test_acos_and_cvr_metrics():
    stats = pd.DataFrame({"day": [1, 2], "entity_id": ["A", "A"], "bid": [10, 10], "clicks": [50, 50],
                          "cost": [300.0, 200.0], "conversions": [2, 3], "revenue": [1000.0, 1500.0],
                          "profit": [0.0, 0.0]})
    m = entity_metrics(stats, 2, 7).loc["A"]
    assert m.acos_7d == pytest.approx(500 / 2500)
    assert m.cvr_7d == pytest.approx(5 / 100)
    assert m.roas_7d == pytest.approx(5.0)


def test_contribution_margin_breakdown(store):
    p = pd.read_csv(store / "products.csv")
    assert {"gross_margin_pct", "fee_pct", "return_rate", "margin_pct"} <= set(p.columns)
    assert np.allclose(p.gross_margin_pct, p.margin_pct + p.fee_pct + p.return_rate * 0.5, atol=0.0011)
    assert (p.fee_pct.between(0.08, 0.15)).all() and (p.return_rate.between(0.02, 0.08)).all()


def test_changes_carry_comparison_period_role_and_confidence(cfg, store):
    ch = plan_day(AdPlatform(cfg, store), cfg)["changes"]
    assert any(c.evidence.spend_prev_7d > 0 for c in ch)
    assert {c.role for c in ch} <= {"discovery", "proven", "waste_candidate", "standard"}
    assert {c.confidence for c in ch} <= {"high", "medium", "low"} and len({c.confidence for c in ch}) >= 2


def test_proven_targets_protected(cfg, store):
    planned = plan_day(AdPlatform(cfg, store), cfg)
    proven = planned["ctx"]["proven"]
    assert len(proven) == cfg["guardrails"]["protect_top_profit"]
    cuts = [c for c in planned["changes"] if c.entity_id in proven and c.action == "lower"
            and c.workflow == "bid_adjust"]
    for c in cuts:
        assert c.change_pct >= -cfg["guardrails"]["protected_max_cut"] - 1e-9
        assert c.needs_approval and "proven" in c.approval_reason
        assert c.role == "proven"


def test_new_templates_pass_number_check():
    ev = Evidence(spend_7d=2310.5, clicks_7d=140, conversions_7d=6, revenue_7d=9450.0, roas_7d=4.09,
                  profit_7d=2400.25, roas_prev_7d=3.12, spend_prev_7d=2000, days_of_stock=22.5)
    for kw in [dict(action="raise", objective="growth"), dict(action="lower", role="proven"),
               dict(action="raise"), dict(action="lower"), dict(action="pause", workflow="waste")]:
        c = Change(entity_id="K1", entity_type="keyword", name="x", workflow=kw.pop("workflow", "bid_adjust"),
                   old_bid=20.0, new_bid=18.0 if kw["action"] == "lower" else 22.0, evidence=ev, **kw)
        r = template_reason(c)
        assert check_numbers(r, change_values(c)) == [], r


def test_optimizer_growth_respects_acos_cap(cfg):
    curves = pd.DataFrame({"status": "ok", "C": 60.0, "k": 0.08, "cvr": 0.03, "cpc_ratio": 0.75, "price": 2000.0,
                           "margin_pct": 0.2, "value_per_click": 0.03 * 2000 * 0.2}, index=["K001"])
    profit_mode = optimize(curves, {"K001": 20.0}, cfg, budget=1e9).loc["K001"]
    growth = optimize(curves, {"K001": 20.0}, cfg, budget=1e9, growth={"K001": 0.35}).loc["K001"]
    assert growth.new_bid > profit_mode.new_bid
    assert growth.new_bid * 0.75 / (0.03 * 2000) <= 0.35 + 1e-9


def test_growth_objective_buys_more_traffic_within_cap(cfg, store):
    pid = "P25"
    ents = pd.read_csv(store / "entities.csv")
    ids = set(ents[ents.product_id == pid].entity_id)

    def traffic(c):
        p = AdPlatform(c, store)
        s = make_strategy("agent_nollm", c)
        for d in range(1, 15):
            s.act(p, d)
            p.step()
        st = p.get_stats()
        st = st[(st.day > 0) & st.entity_id.isin(ids)]
        return st.clicks.sum(), st.cost.sum(), st.revenue.sum()

    base = traffic(cfg)
    g = traffic(load_config(overrides={"llm": {"enabled": False}, "objectives": {pid: {"mode": "growth", "max_acos": 0.35}}}))
    assert g[0] > base[0] and g[1] > base[1]
    assert g[1] / g[2] <= 0.45            # realised ACOS near the 35% cap (noise allowed)


def test_fair_outcome_report(cfg, store):
    res = {s: summarize(run(s, cfg=cfg, out=store)) for s in ["fixed", "rules", "agent_nollm"]}
    for r in res.values():
        assert r["total_sales"] > r["attributed_sales"] > 0
        assert 0 < r["stock_availability"] <= 1 and 0 < r["acos"] < 1
    a = res["agent_nollm"]
    assert a["total_contribution_profit"] > res["rules"]["total_contribution_profit"]
    assert a["total_contribution_profit"] > res["fixed"]["total_contribution_profit"]
    assert a["stock_availability"] >= res["rules"]["stock_availability"]


def test_total_contribution_matches_units(cfg, store):
    p = AdPlatform(cfg, store)
    t = p.step()
    sh = p.get_stock_history()
    sh = sh[sh.day == 1].merge(p.get_products()[["product_id", "price", "margin_pct"]], on="product_id")
    expect = (sh.units_sold * sh.price * sh.margin_pct).sum() - t["spend"]
    assert t["total_contribution_profit"] == pytest.approx(expect, abs=0.05)


@pytest.fixture(scope="module")
def week(cfg, store):
    p = AdPlatform(cfg, store)
    s = make_strategy("agent_nollm", cfg)
    for d in range(1, 8):
        s.act(p, d)
        p.step()
    return p.public_data()


def test_preset_weak_converters_find_planted_listings(cfg, store, week, tmp_path):
    src = ask(PRESET_QUESTIONS[4], week, cfg, out=tmp_path)
    assert src["ok"], src
    t = pd.DataFrame(src["result_table"])
    planted = set(json.loads((store / "_hidden_market.json").read_text())["weak_listing_products"])
    assert len(planted & set(t.product_id)) >= 2
    # hand-written check of one row
    st, en, pr = week["stats"], week["entities"], week["products"]
    last = st[st.day > st.day.max() - 14].merge(en[["entity_id", "product_id"]]).merge(pr[["product_id", "category"]])
    row = t.iloc[0]
    sub = last[last.product_id == row.product_id]
    cat = last[last.category == row.category]
    assert row.clicks == sub.clicks.sum()
    assert row.cvr_vs_category == pytest.approx(
        (sub.conversions.sum() / sub.clicks.sum()) / (cat.conversions.sum() / cat.clicks.sum()), abs=0.006)


def test_preset_overstock_push(cfg, week, tmp_path):
    src = ask(PRESET_QUESTIONS[5], week, cfg, out=tmp_path)
    assert src["ok"], src
    t = pd.DataFrame(src["result_table"])
    assert len(t) >= 1 and (t.days_of_stock > 60).all() and (t.profit > 0).all()
    sk = week["stock"]
    last = sk.day.max()
    r = t.iloc[0]
    units = sk[(sk.day == last) & (sk.product_id == r.product_id)].stock_units.iloc[0]
    avg = sk[(sk.day > last - 7) & (sk.product_id == r.product_id)].units_sold.mean()
    assert r.days_of_stock == pytest.approx(round(units / avg, 1), abs=0.051)
