import json
import time

from agent.orchestrator import AgentStrategy, finish_day, run_agent_day
from agent.workflows import _match, build_context, consultant_actions, daily_checkup, plan_day
from sim.platform import AdPlatform
from sim.runner import run, summarize
from tests.fakes import FakeLLM


def _fake_llm(hold=None):
    def planner(s, u):
        props = json.loads(u.split("\n", 1)[1])["proposals"]
        ids = [p["id"] for p in props if p["workflow"] == "bid_adjust"][:1] if hold else []
        return {"focus": ["stock"], "hold": ids, "hold_reasons": {i: "thin data" for i in ids},
                "day_note": "Quiet day."}
    return FakeLLM({
        "planner": planner,
        "explainer": lambda s, u: {"reason": u.split("DRAFT: ", 1)[1]},   # echo the correct draft
        "day_report": lambda s, u: {"report": "All good."},
    })


def test_workflows_produce_changes_with_evidence(cfg, store):
    p = AdPlatform(cfg, store)
    planned = plan_day(p, cfg)
    ch = planned["changes"]
    assert len(ch) > 10
    assert {c.workflow for c in ch} >= {"bid_adjust"}
    for c in ch:
        if c.action in ("raise", "lower") and c.workflow != "stock_guard":
            assert abs(c.change_pct) <= cfg["bids"]["max_daily_change"] + 1e-6
    assert any(c.needs_approval for c in ch)
    assert all(c.evidence is not None for c in ch)


def test_daily_checkup_flags_only(cfg, store):
    p = AdPlatform(cfg, store)
    flags = daily_checkup(build_context(p, cfg))
    assert all(abs(f["change_pct"]) >= 0.30 for f in flags)   # stored rounded to 4 decimals


def test_consultant_filter_and_bounds(cfg, store):
    p = AdPlatform(cfg, store)
    ctx = build_context(p, cfg)
    action = {"filter": {"keyword_contains": ["free"]}, "action": "lower", "max_pct": 0.5}
    props = consultant_actions(ctx, [action])
    assert props
    for eid, (_action, new) in props.items():
        assert "free" in ctx["entities"].loc[eid, "name"]
        assert new >= ctx["bids"][eid] * 0.8 - 1e-9   # capped at 20%
    assert not _match({"filter": {}}, eid, ctx["entities"], {})   # empty filter matches nothing


def test_consultant_action_changes_next_day_bids(cfg, store):
    p = AdPlatform(cfg, store)
    action = {"filter": {"keyword_contains": ["free"]}, "action": "pause", "max_pct": 0.2}
    planned = plan_day(p, cfg, [action])
    paused = [c for c in planned["changes"] if c.workflow == "consultant"]
    assert paused and all(c.action == "pause" for c in paused)


def test_planner_hold_keeps_old_bid(cfg, fresh_store):
    p = AdPlatform(cfg, fresh_store)
    before = p.get_bids()
    res = run_agent_day(p, 1, auto_approve=True, cfg=cfg, llm=_fake_llm(hold=True), out=fresh_store,
                        run_consultant=False)
    held = res["plan"]["held"]
    assert len(held) == 1
    eid = held[0]["entity_id"]
    assert p.get_bids()[eid] == before[eid]
    assert res["plan"]["day_note"] == "Quiet day."
    assert (fresh_store / "day_1_plan.json").exists()


def test_manual_approval_flow(cfg, fresh_store):
    p = AdPlatform(cfg, fresh_store, persist=True)
    res = run_agent_day(p, 1, auto_approve=False, cfg=cfg, llm=None, out=fresh_store)
    plan = res["plan"]
    need = [c for c in plan["changes"] if c["needs_approval"] and c["action"] in ("raise", "lower")]
    assert len(need) >= 2
    approve, reject = need[0], need[1]
    decisions = {approve["entity_id"]: "approve", reject["entity_id"]: "reject"}
    (fresh_store / "day_1_decisions.json").write_text(json.dumps(decisions))
    out = finish_day(p, plan, cfg=cfg, out=fresh_store)
    bids = AdPlatform(cfg, fresh_store, persist=True).get_bids()
    assert bids[approve["entity_id"]] == approve["new_bid"]
    assert bids[reject["entity_id"]] == reject["old_bid"]
    assert out["applied"]["human"] == 1 and reject["entity_id"] in out["applied"]["rejected"]
    ch = p.get_changes()
    assert (ch[ch.entity_id == approve["entity_id"]].approved_by == "human").all()


def test_agent_day_with_llm_is_fast(cfg, fresh_store):
    p = AdPlatform(cfg, fresh_store)
    t = time.time()
    res = run_agent_day(p, 1, auto_approve=True, cfg=cfg, llm=_fake_llm(), out=fresh_store, run_consultant=False)
    assert time.time() - t < 60
    assert res["report"] == "All good."
    assert all(c["reason_source"] == "llm" for c in res["plan"]["changes"])


def test_agent_with_llm_at_least_rules(cfg, store):
    from tests.test_consultant import consultant_llm
    rules = summarize(run("rules", cfg=cfg, out=store))
    strat_llm = _fake_llm()
    strat_llm.handlers.update(consultant_llm().handlers)
    agent = summarize(run("agent", cfg=cfg, out=store, llm=strat_llm))
    assert agent["profit"] >= rules["profit"]
    assert "explainer" in strat_llm.calls and "consultant" in strat_llm.calls


def test_no_llm_mode_works(cfg, store):
    s = AgentStrategy(cfg, out=store, llm=None, use_llm=False)
    p = AdPlatform(cfg, store)
    s.act(p, 1)
    assert not p.get_changes().empty
