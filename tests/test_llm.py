import asyncio
import json

import pytest

from agent import prompts as P
from agent.explain import check_numbers, explain_changes, extract_numbers, template_reason
from agent.llm import LLMClient, LLMError, parse_json
from engine.models import Change, Evidence
from tests.fakes import FakeLLM, FakeOpenAIServer


def _cfg(cfg, **llm):
    c = json.loads(json.dumps(cfg))
    c["llm"].update({"enabled": True, "base_url": "http://fake/v1", **llm})
    return c


def _change(**kw):
    base = dict(entity_id="K001", entity_type="keyword", name="cotton king bedsheet", workflow="bid_adjust",
                action="raise", old_bid=18.0, new_bid=21.0,
                evidence=Evidence(spend_7d=2310.5, clicks_7d=140, conversions_7d=6, revenue_7d=9450.0, roas_7d=4.09,
                                  profit_7d=2400.25, days_of_stock=22.5))
    base.update(kw)
    return Change(**base)


def test_extract_numbers_ignores_ids_and_tags():
    assert extract_numbers("Raise P14 bid from Rs 1,234.50 to 21 (+17%) [D3] [W12]") == [1234.5, 21.0, 17.0]


def test_number_check():
    vals = [18.0, 21.0, 0.1667, 2310.5, 4.09]
    assert check_numbers("Raise from Rs 18 to Rs 21 (+17%), ROAS 4.1, spend Rs 2,311", vals) == []
    assert check_numbers("ROAS 9.99 so raise", vals) == [9.99]


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 2} hope that helps') == {"a": 2}


def test_explainer_rejects_made_up_number(tmp_path):
    good, bad = _change(entity_id="K001"), _change(entity_id="K002", name="made up")
    llm = FakeLLM({"explainer": lambda s, u: {"reason": "ROAS 4.09 over 7 days; raising Rs 18 to Rs 21."}
                   if "cotton" in u else {"reason": "It earns Rs 77 per sale, so raise."}})
    out = asyncio.run(explain_changes(llm, [good, bad], tmp_path))
    assert out[0].reason_source == "llm"
    assert out[1].reason_source == "template"
    assert out[1].reason == template_reason(bad)
    assert (tmp_path / "explain_rejections.jsonl").exists()


def test_template_reasons_pass_number_check():
    from agent.explain import change_values
    for wf, action in [("bid_adjust", "raise"), ("bid_adjust", "lower"), ("waste", "pause"),
                       ("stock_guard", "pause"), ("stock_guard", "lower"), ("stock_guard", "resume"),
                       ("consultant", "lower")]:
        c = _change(workflow=wf, action=action, new_bid=9.0 if action == "lower" else 21.0)
        assert check_numbers(template_reason(c), change_values(c)) == [], (wf, action)


def test_explanations_without_llm_use_templates(tmp_path):
    out = asyncio.run(explain_changes(None, [_change()], tmp_path))
    assert out[0].reason_source == "template"


# ---------- the real client against a fake OpenAI-compatible server ----------

def test_client_json_schema_mode_and_log(cfg, tmp_path):
    srv = FakeOpenAIServer(lambda body: json.dumps({"reason": "ok"}))
    llm = LLMClient(_cfg(cfg), tmp_path, transport=srv.transport())
    out = asyncio.run(llm.chat_json("explainer", "sys", "user", P.ExplainOut))
    assert out == {"reason": "ok"}
    assert srv.requests[0]["response_format"]["type"] == "json_schema"
    log = [json.loads(line) for line in (tmp_path / "llm_log.jsonl").read_text().splitlines()]
    assert log[0]["ok"] and log[0]["tokens_in"] == 11 and log[0]["tokens_out"] == 7 and log[0]["role"] == "explainer"


def test_client_falls_back_to_prompt_mode(cfg, tmp_path):
    srv = FakeOpenAIServer(lambda body: '```json\n{"reason": "fallback"}\n```', reject_schema=True)
    llm = LLMClient(_cfg(cfg), tmp_path, transport=srv.transport())
    out = asyncio.run(llm.chat_json("explainer", "sys", "user", P.ExplainOut))
    assert out == {"reason": "fallback"}
    assert llm.json_mode == "prompt"
    assert "JSON schema" in srv.requests[-1]["messages"][0]["content"]


def test_client_retries_once_then_raises(cfg, tmp_path):
    srv = FakeOpenAIServer(lambda body: "not json at all")
    llm = LLMClient(_cfg(cfg), tmp_path, transport=srv.transport())
    with pytest.raises(LLMError):
        asyncio.run(llm.chat_json("explainer", "sys", "user", P.ExplainOut))
    assert len(srv.requests) == 2


def test_client_runs_concurrently_with_semaphore(cfg, tmp_path):
    srv = FakeOpenAIServer(lambda body: json.dumps({"reason": "Strong ROAS, so the bid goes from Rs 18.00 to Rs 21.00."}),
                           delay=0.05)
    llm = LLMClient(_cfg(cfg, max_concurrency=8), tmp_path, transport=srv.transport())
    changes = [_change(entity_id=f"K{i:03d}") for i in range(1, 41)]
    out = asyncio.run(explain_changes(llm, changes, tmp_path))
    assert all(c.reason_source == "llm" for c in out)
    assert 1 < srv.max_in_flight <= 8


def test_explainer_guards_meaning(tmp_path):
    """Real small models attach numbers to the wrong meaning or talk about prices; those fall back to the draft."""
    cases = {
        "ROAS fell by 3.12 so the bid moves from Rs 18 to Rs 21.": "number check",       # 3.12 not in the draft
        "Raising the price helps profit, bid from Rs 18 to Rs 21.": "price",               # bid is not a price
        "Raising the bid because ROAS is strong.": "missing bid amounts",                  # no bid amounts
    }
    pause = _change(workflow="stock_guard", action="pause", new_bid=18.0)
    llm = FakeLLM({"explainer": lambda s, u: {"reason": "Pause bid from Rs 2311 to Rs 0, out of stock."}})
    assert asyncio.run(explain_changes(llm, [pause], tmp_path))[0].reason_source == "template"
    for reply, why in cases.items():
        llm = FakeLLM({"explainer": lambda s, u, r=reply: {"reason": r}})
        out = asyncio.run(explain_changes(llm, [_change()], tmp_path))
        assert out[0].reason_source == "template", reply
        last = json.loads((tmp_path / "explain_rejections.jsonl").read_text().splitlines()[-1])
        assert why in last["why"], (reply, last)
    ok = FakeLLM({"explainer": lambda s, u: {"reason": "Good returns (ROAS 4.09), so the bid goes from Rs 18 to Rs 21."}})
    assert asyncio.run(explain_changes(ok, [_change()], tmp_path))[0].reason_source == "llm"


def test_explainer_gets_a_draft():
    from agent.prompts import explainer_user
    msg = explainer_user(_change().model_dump(), "DRAFT TEXT 1")
    assert "DRAFT: DRAFT TEXT 1" in msg and "Facts" not in msg
    assert "Facts" in explainer_user(_change().model_dump())
