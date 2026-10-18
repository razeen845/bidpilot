import json
import re

import pytest

from agent.analyst import PRESET_CODE, PRESET_QUESTIONS
from agent.consultant import decide_recommendations, fan_out, latest_report, run_consultant
from agent.llm import run_sync
from agent.orchestrator import load_pending_actions, prepare_day_async
from sim.platform import AdPlatform
from sim.runner import make_strategy
from tests.fakes import FakeLLM

QUESTIONS = PRESET_QUESTIONS[0], PRESET_QUESTIONS[3], PRESET_QUESTIONS[2]


def _catalog(u):
    return json.loads(u.split("Sources you may cite:\n", 1)[1].split("\n\nThis recommendation failed", 1)[0])


def _by_question(cat, q):
    return next(s for s in cat if s.get("question") == q)


def _writer(s, u):
    cat = _catalog(u)
    d_kw, d_low, d_oos = (_by_question(cat, q) for q in QUESTIONS)
    w_stock = next(s for s in cat if s["id"].startswith("W") and "goinflow" in s["url"])
    if "failed the citation check" in u:
        if "DROP" in u:
            return {"recommendations": [{"title": "DROP", "finding": "Made up Rs 4242 [D99].", "recommendation": "x",
                                         "type": "owner_todo", "citations": ["D99"]}]}
        return {"recommendations": [{"title": "Fixed", "finding": f"Low-margin products took "
                                     f"{d_low['summary']['low_margin_spend_share_pct']}% of ad spend [{d_low['id']}].",
                                     "recommendation": "Review prices.", "type": "owner_todo",
                                     "expected_impact": "", "citations": [d_low["id"]]}]}
    top = d_kw["table"][0]
    return {"recommendations": [
        {"title": f"Stop '{top['word']}' searches",
         "finding": f"Keywords containing '{top['word']}' spent Rs {top['zero_sale_spend']:,} with zero sales [{d_kw['id']}].",
         "recommendation": "Pause them.", "type": "bid_action",
         "bid_action": {"filter": {"keyword_contains": [top["word"]]}, "action": "pause", "max_pct": 0.9},
         "expected_impact": f"Saves up to Rs {top['zero_sale_spend']:,} per two weeks [{d_kw['id']}].",
         "priority": 2, "confidence": "high", "risk": "A few relevant searches may contain the word.",
         "next_action": "Approve the pause tomorrow.", "citations": [d_kw["id"]]},
        {"title": "Low-margin products eat the budget",
         "finding": f"{d_low['summary']['low_margin_products']} low-margin products took "
                    f"{d_low['summary']['low_margin_spend_share_pct']}% of ad spend [{d_low['id']}].",
         "recommendation": "Raise prices or drop them.", "type": "owner_todo", "expected_impact": "",
         "priority": 1, "risk": "Raising prices can cut organic sales too.", "citations": [d_low["id"]]},
        {"title": "Restock earlier",
         "finding": f"{d_oos['summary']['products_out_of_stock']} products ran out of stock [{d_oos['id']}]; "
                    f"pause search ads until back in stock [{w_stock['id']}].",
         "recommendation": "Restock earlier.", "type": "owner_todo", "expected_impact": "",
         "citations": [d_oos["id"], w_stock["id"]]},
        {"title": "Needs a rewrite", "finding": f"Low margin share is {d_low['summary']['low_margin_spend_share_pct']}% "
                                                f"[{d_low['id']}].", "risk": "You could lose Rs 31337 a day.",
         "recommendation": "Review prices.", "type": "owner_todo", "citations": [d_low["id"]]},
        {"title": "DROP", "finding": "Invented Rs 4242 savings [D99].", "recommendation": "x",
         "type": "owner_todo", "citations": ["D99"]},
    ]}


def _analyst(s, u):
    if "Tables:" in u:
        q = u.split("Question: ", 1)[1].split("\n", 1)[0]
        return {"code": PRESET_CODE[q]}
    tag = re.search(r"Source tag: \[(D\d+)\]", u).group(1)
    return {"answer": f"See the table [{tag}]."}


def consultant_llm(delay=0.0):
    return FakeLLM({
        "consultant": lambda s, u: _writer(s, u) if "Sources you may cite" in u
        else {"questions": list(QUESTIONS), "queries": ["home textile online sales seasonality"]},
        "analyst": _analyst,
    }, delay=delay)


@pytest.fixture()
def day7(cfg, fresh_store):
    p = AdPlatform(cfg, fresh_store, persist=True)
    s = make_strategy("rules", cfg)
    for d in range(1, 8):
        s.act(p, d)
        p.step()
    return p, fresh_store


def test_consultant_report_is_cited_and_checked(cfg, day7):
    p, out = day7
    rep = run_consultant(p, 7, cfg, consultant_llm(), out=out)
    recs = {r["title"]: r for r in rep["recommendations"]}
    verified = [r for r in rep["recommendations"] if r["status"] == "verified"]
    assert len(verified) >= 3
    assert all(any(c.startswith("D") for c in r["citations"]) for r in verified)
    assert any(c.startswith("W") for r in verified for c in r["citations"])
    assert recs["Fixed"]["rewritten"] and recs["Fixed"]["status"] == "verified"
    assert recs["DROP"]["status"] == "unverified"
    # the risk text is citation-checked too: "Needs a rewrite" only had a made-up number in its risk
    assert "Needs a rewrite" not in recs
    # ranked action list: verified first, by priority
    ranked = sorted(rep["recommendations"], key=lambda r: r["rank"])
    assert ranked[0]["title"] == "Low-margin products eat the budget"
    pri = [r["priority"] for r in ranked if r["status"] == "verified"]
    assert pri == sorted(pri)
    assert all(r["status"] == "unverified" for r in ranked[len(pri):])
    # bounded bid action
    bid = next(r for r in verified if r["type"] == "bid_action")
    assert bid["bid_action"]["max_pct"] == 0.2
    # sources resolve: D has code + table, W has url + quote + date
    for s in rep["sources"]["D"]:
        assert s["code"] and s["result_table"] and (out / "analysis" / f"{s['id']}.json").exists()
    for s in rep["sources"]["W"]:
        assert s["url"].startswith("https://") and s["quote"] and s["retrieved_at"]
    assert (out / "consult" / "day_7.md").exists()
    assert latest_report(out)["day"] == 7


def test_accepted_bid_action_changes_next_day(cfg, day7):
    p, out = day7
    rep = run_consultant(p, 7, cfg, consultant_llm(), out=out)
    ids = {r["title"]: r["id"] for r in rep["recommendations"]}
    bid = next(r for r in rep["recommendations"] if r["type"] == "bid_action")
    res = decide_recommendations(7, {bid["id"]: "accept", ids["Low-margin products eat the budget"]: "accept",
                                     ids["DROP"]: "reject"}, out)
    assert len(res["pending_actions"]) == 1 and len(res["todos"]) == 1
    pending = load_pending_actions(out)
    plan, changes = run_sync(prepare_day_async(p, cfg, None, pending, out))
    word = bid["bid_action"]["filter"]["keyword_contains"][0]
    consult = [c for c in changes if c.workflow == "consultant"]
    assert consult and all(word in c.name for c in consult)
    assert all(c.action == "pause" and c.needs_approval for c in consult)


def test_fan_out_runs_in_parallel(cfg, day7):
    """Analytics questions overlap in parallel mode and never overlap in sequential mode (counts, not timings)."""
    p, out = day7
    data = p.public_data()
    questions = ["Total spend by day?", "Clicks by entity type?", "Conversions by day?"]   # free text: LLM writes code

    def fake():
        return FakeLLM({"analyst": lambda s, u: {"code": "result = stats.groupby('day').cost.sum().tail(3)"}
                        if "Tables:" in u else {"answer": "See [D1]."}}, delay=0.2)
    seq = fake()
    run_sync(fan_out(questions, ["q"], data, cfg, seq, out, parallel=False))
    par = fake()
    run_sync(fan_out(questions, ["q"], data, cfg, par, out, parallel=True))
    assert seq.max_in_flight == 1
    assert par.max_in_flight >= 2


def test_consultant_needs_llm(cfg, day7):
    p, out = day7
    with pytest.raises(RuntimeError):
        run_consultant(p, 7, cfg, None, out=out)


def test_unverified_bid_action_never_changes_bids(tmp_path):
    """Accepting a greyed-out (unverified) recommendation makes it an owner to-do, never a bid change."""
    (tmp_path / "consult").mkdir()
    action = {"filter": {"keyword_contains": ["free"]}, "action": "pause", "max_pct": 0.2}
    rep = {"day": 7, "recommendations": [
        {"id": "R1", "title": "ok", "type": "bid_action", "bid_action": action, "status": "verified",
         "recommendation": "pause", "citations": ["D1"], "decision": None},
        {"id": "R2", "title": "made up", "type": "bid_action", "bid_action": action, "status": "unverified",
         "recommendation": "pause", "citations": ["D9"], "decision": None}]}
    (tmp_path / "consult" / "day_7.json").write_text(json.dumps(rep))
    res = decide_recommendations(7, {"R1": "accept", "R2": "accept"}, tmp_path)
    assert [a["source"] for a in res["pending_actions"]] == ["day 7 R1"]
    assert [t["title"] for t in res["todos"]] == ["made up"] and res["todos"][0]["status"] == "unverified"
