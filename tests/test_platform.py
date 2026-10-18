import re
from pathlib import Path

import pandas as pd

from sim.platform import AdPlatform
from sim.runner import run

ROOT = Path(__file__).resolve().parents[1]


def test_runner_deterministic(cfg, store):
    a = run("rules", days=5, cfg=cfg, out=store)
    b = run("rules", days=5, cfg=cfg, out=store)
    pd.testing.assert_frame_equal(a, b)


def test_common_random_numbers(cfg, store):
    p1, p2 = AdPlatform(cfg, store), AdPlatform(cfg, store)
    p1.step()
    p2.step()
    pd.testing.assert_frame_equal(p1.get_report(1, 1), p2.get_report(1, 1))


def test_budget_never_exceeded(cfg, store):
    p = AdPlatform(cfg, store)
    p.set_bids({eid: 80 for eid in p.get_bids()})
    for _ in range(3):
        t = p.step()
        assert t["spend"] <= cfg["store"]["daily_budget"] + 0.01


def test_paused_entities_get_zero_clicks(cfg, store):
    p = AdPlatform(cfg, store)
    ids = list(p.get_bids())[:10]
    for eid in ids:
        p.pause(eid)
    p.step()
    r = p.get_report(1, 1).set_index("entity_id")
    assert (r.loc[ids, "clicks"] == 0).all()
    assert (r.loc[ids, "cost"] == 0).all()


def test_bids_clipped(cfg, store):
    p = AdPlatform(cfg, store)
    eid = next(iter(p.get_bids()))
    p.set_bids({eid: 1000})
    assert p.get_bids()[eid] == cfg["bids"]["max_bid"]
    p.set_bids({eid: 0.1})
    assert p.get_bids()[eid] == cfg["bids"]["min_bid"]


def test_persistence_roundtrip(cfg, fresh_store):
    p = AdPlatform(cfg, fresh_store, persist=True)
    p.step()
    q = AdPlatform(cfg, fresh_store, persist=True)
    assert q.day == 1
    pd.testing.assert_frame_equal(p.get_stats(), q.get_stats())


def test_public_data_has_no_hidden_market(cfg, store):
    data = AdPlatform(cfg, store).public_data()
    assert set(data) == {"stats", "stock", "products", "entities", "changes"}
    for df in data.values():
        assert not {"click_cap", "k", "cvr", "cpc_ratio"} & set(df.columns)


def test_agent_code_never_touches_hidden_market():
    pat = re.compile(r"sim\.market|from sim import market|_hidden_market")
    for pkg in ["agent", "engine", "analysis", "sandbox"]:
        for f in (ROOT / pkg).rglob("*.py"):
            assert not pat.search(f.read_text(encoding="utf-8")), f
