"""Phase 13: incrementality, platform rule profiles, rollback rules, production hardening."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent import experiments as X
from agent.explain import change_values, check_numbers, template_reason
from agent.orchestrator import run_agent_day
from agent.workflows import plan_day
from engine.models import Change, Evidence
from engine.platform_rules import PROFILES, waste_action
from engine.rollback import rollback_cooldown, rollback_proposals
from settings import ConfigError, load_config, read_json, write_json
from sim.market import simulate_day
from sim.platform import AdPlatform

ROOT = Path(__file__).resolve().parents[1]


# ---------------- incrementality in the simulator ----------------

def _one_product(inc):
    ents = {"K001": {"product_id": "P01"}}
    params = {"K001": {"click_cap": 400.0, "k": 0.5, "cvr": 0.05, "cpc_ratio": 0.7}}
    prods = {"P01": {"price": 1000.0, "margin_pct": 0.3, "organic_sales_per_day": 20.0, "restock_day": -1,
                     "restock_units": 0, "incrementality": inc}}
    return ents, params, prods


def test_cannibalization_reduces_organic_not_attributed():
    sold = {}
    for inc in (1.0, 0.3):
        ents, params, prods = _one_product(inc)
        rows, srows, _ = simulate_day(day=1, seed=42, entities=ents, params=params, products=prods,
                                      bids={"K001": 20.0}, stock={"P01": 10_000}, budget=1e9)
        sold[inc] = (rows[0]["conversions"], srows[0]["units_sold"])
    assert sold[1.0][0] == sold[0.3][0] > 5            # the platform attributes the same ad sales
    assert sold[0.3][1] < sold[1.0][1]                  # but fewer total units are sold


def test_switchback_design_is_balanced_and_deterministic():
    a = X.design_switchback("P13", 1, 14, 42)
    assert a == X.design_switchback("P13", 1, 14, 42)
    assert sorted(a) == list(range(1, 15))
    for b in range(1, 15, 2):
        assert {a[b], a[b + 1]} == {"on", "off"}
    assert a != X.design_switchback("P12", 1, 14, 42) or a != X.design_switchback("P13", 2, 14, 42)


def _exp_frames(on_units, off_units, attr_on, days=14):
    sched = X.design_switchback("P01", 1, days, 0)
    rng = np.random.default_rng(0)
    stats, stock = [], []
    for d, state in sched.items():
        attr = rng.poisson(attr_on) if state == "on" else 0
        units = rng.poisson(on_units if state == "on" else off_units)
        stats.append({"day": d, "entity_id": "K001", "conversions": attr})
        stock.append({"day": d, "product_id": "P01", "stock_units": 500, "units_sold": units})
    ents = pd.DataFrame({"entity_id": ["K001"], "product_id": ["P01"]})
    exp = {"product_id": "P01", "schedule": {str(k): v for k, v in sched.items()}}
    return exp, pd.DataFrame(stats), pd.DataFrame(stock), ents


def test_analyze_detects_cannibalization(cfg):
    exp, stats, stock, ents = _exp_frames(on_units=26, off_units=20, attr_on=20, days=28)   # true incrementality 0.3
    r = X.analyze(exp, stats, stock, ents, cfg)
    assert r["status"] == "ok" and r["ci90"][1] < 1.0 and r["conclusive"]
    assert r["factor"] == pytest.approx(min(1.0, r["ci90"][1]))       # conservative: CI upper bound
    assert "cannibalize" in r["verdict"]


def test_analyze_incremental_and_insufficient(cfg):
    exp, stats, stock, ents = _exp_frames(on_units=40, off_units=0.5, attr_on=40, days=28)   # fully incremental
    r = X.analyze(exp, stats, stock, ents, cfg)
    assert r["ci90"][0] <= 1.0 <= r["ci90"][1] + 0.05 and r["factor"] in (None, 1.0)
    short = {**exp, "schedule": dict(list(exp["schedule"].items())[:4])}
    assert X.analyze(short, stats, stock, ents, cfg)["status"] == "insufficient_data"


def test_experiment_lifecycle_on_live_account(cfg, fresh_store):
    """Planted low-incrementality best-seller: the switchback CI excludes 1.0 and a factor is stored."""
    p = AdPlatform(cfg, fresh_store, persist=True)
    exp = X.propose("P13", cfg, fresh_store)
    with pytest.raises(ValueError):
        X.propose("P13", cfg, fresh_store)                 # one test per product at a time
    X.approve(exp["id"], 1, cfg, fresh_store)
    ids = set(p.get_entities().query("product_id == 'P13'").entity_id)
    off_day = next(int(d) for d, s in X.load_experiments(fresh_store)[0]["schedule"].items() if s == "off")
    for _ in range(14):
        res = run_agent_day(p, auto_approve=True, cfg=cfg, out=fresh_store, run_consultant=False)
        xc = [c for c in res["plan"]["changes"] if c["workflow"] == "experiment"]
        assert all(not c["needs_approval"] for c in xc)    # pre-approved by the owner
        if res["plan"]["day"] == off_day:
            stats = p.get_stats()
            assert (stats[(stats.day == off_day) & stats.entity_id.isin(ids)].clicks == 0).all()
    e = X.load_experiments(fresh_store)[0]
    assert e["status"] == "done"
    assert e["result"]["ci90"][1] < 1.0
    assert X.load_factors(fresh_store)["P13"]["factor"] < 1.0
    assert not (p.get_paused() & ids)                     # ads are back on after the test


def test_incrementality_factor_lowers_click_value(cfg, store):
    base = plan_day(AdPlatform(cfg, store), cfg)["ctx"]
    c2 = load_config(overrides={"llm": {"enabled": False}, "incrementality": {"P13": 0.5}})
    ctx = plan_day(AdPlatform(c2, store), c2)["ctx"]
    eid = next(e for e, pid in ctx["product_of"].items() if pid == "P13" and ctx["curves"].loc[e, "status"] == "ok")
    assert ctx["curves"].loc[eid, "value_per_click"] == pytest.approx(base["curves"].loc[eid, "value_per_click"] * 0.5)


# ---------------- platform rule profiles ----------------

def test_platform_profiles():
    amazon = load_config(overrides={"platform": "amazon"})
    google = load_config(overrides={"platform": "google"})
    assert amazon["guardrails"]["conversion_lag_days"] == PROFILES["amazon"]["conversion_lag_days"]
    assert google["guardrails"]["conversion_lag_days"] == PROFILES["google"]["conversion_lag_days"]
    assert waste_action(amazon, "keyword") == "add as negative exact keyword"
    assert waste_action(google, "product") == "exclude product from Shopping campaign"
    assert amazon["platform_profile"]["primary_kpi"] == "acos"


def test_amazon_profile_changes_wording(store):
    c = load_config(overrides={"platform": "amazon", "llm": {"enabled": False}})
    ch = plan_day(AdPlatform(c, store), c)["changes"]
    assert all(x.kpi == "acos" for x in ch)
    adj = next(x for x in ch if x.workflow == "bid_adjust")
    assert "ACOS" in template_reason(adj)
    assert check_numbers(template_reason(adj), change_values(adj)) == []


def test_waste_template_names_platform_action():
    c = Change(entity_id="K1", entity_type="keyword", name="free rug", workflow="waste", action="pause", old_bid=10,
               new_bid=10, evidence=Evidence(spend_7d=2000, clicks_7d=120), platform_action="add as negative exact keyword")
    r = template_reason(c)
    assert "negative exact keyword" in r and check_numbers(r, change_values(c)) == []


# ---------------- rollback ----------------

def _rb_stats(after_profit_drop: bool):
    rows = []
    rng = np.random.default_rng(1)
    for day in range(1, 9):
        bid = 10.0 if day <= 4 else 16.0
        profit = rng.normal(800, 60) if day <= 4 or not after_profit_drop else rng.normal(-200, 60)
        rows.append({"day": day, "entity_id": "K001", "bid": bid, "clicks": 30, "cost": 200.0, "conversions": 2,
                     "revenue": 2000.0, "profit": profit})
    return pd.DataFrame(rows)


def _agent_raise(day=6):
    return pd.DataFrame([{"day": day, "entity_id": "K001", "workflow": "bid_adjust", "action": "raise"}])


def test_rollback_reverts_a_bad_bid(cfg):
    props = rollback_proposals(_rb_stats(True), {"K001": 16.0}, set(), 8, cfg, changes=_agent_raise())
    assert props["K001"][0] == "rollback" and props["K001"][1] == 10.0
    assert props["K001"][2]["p_value"] < cfg["rollback"]["p_value"]


def test_rollback_quiet_when_fine_or_not_ours(cfg):
    assert rollback_proposals(_rb_stats(False), {"K001": 16.0}, set(), 8, cfg, changes=_agent_raise()) == {}
    assert rollback_proposals(_rb_stats(True), {"K001": 16.0}, set(), 8, cfg, changes=pd.DataFrame(
        columns=["day", "entity_id", "workflow"])) == {}                           # the agent did not move it
    off = {**cfg, "rollback": {**cfg["rollback"], "enabled": False}}
    assert rollback_proposals(_rb_stats(True), {"K001": 16.0}, set(), 8, off, changes=_agent_raise()) == {}


def test_rollback_cooldown(cfg):
    ch = pd.DataFrame([{"day": 7, "entity_id": "K001", "workflow": "rollback"}])
    assert rollback_cooldown(ch, 8, cfg) == {"K001"}
    assert rollback_cooldown(ch, 7 + cfg["rollback"]["cooldown_days"], cfg) == set()


# ---------------- production hardening ----------------

def test_config_validation_errors(tmp_path):
    good = (ROOT / "config.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(good.replace("min_bid: 3", "min_bid: 100"), encoding="utf-8")
    with pytest.raises(ConfigError, match="min_bid"):
        load_config(bad)
    bad.write_text(good.replace("platform: generic", "platform: myspace"), encoding="utf-8")
    with pytest.raises(ConfigError, match="platform"):
        load_config(bad)
    bad.write_text("seed: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(bad)
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_atomic_json_roundtrip(tmp_path):
    write_json(tmp_path / "a" / "x.json", {"k": [1, 2]})
    assert read_json(tmp_path / "a" / "x.json") == {"k": [1, 2]}
    assert read_json(tmp_path / "missing.json", default={}) == {}
    assert [f.name for f in (tmp_path / "a").iterdir()] == ["x.json"]      # no temp files left behind


def test_cli_end_to_end(tmp_path):
    env = {**__import__("os").environ, "BIDPILOT_OUT": str(tmp_path), "BIDPILOT_LOG_LEVEL": "WARNING"}

    def cli(*a):
        r = subprocess.run([sys.executable, "-m", "cli", "--no-llm", *a], capture_output=True, text=True, cwd=ROOT,
                           env=env, timeout=300)
        assert r.returncode == 0, r.stderr[-1500:]
        return r.stdout
    assert "validate-config" in cli("--help")
    cli("generate")
    assert "need approval" in cli("day")
    cli("decide", "--approve-all")
    assert "Day 1:" in cli("day", "--apply")
    out = json.loads(cli("experiment", "propose", "P13"))
    assert out["status"] == "proposed"
    assert json.loads(cli("experiment", "approve", out["id"]))["start_day"] == 2
    assert "[D1]" in cli("ask", "2")
    status = json.loads(cli("status"))
    assert status["account_day"] == 1 and status["experiments"][0]["status"] == "running"
