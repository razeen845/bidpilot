"""Headless run of the Streamlit dashboard (no LLM): store, plan + approve/reject + apply, race, ask the data."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


@pytest.fixture()
def app(fresh_store, monkeypatch):
    monkeypatch.setenv("BIDPILOT_OUT", str(fresh_store))
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")   # nothing listens -> --no-llm behaviour
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at, fresh_store


def _button(at, label_start):
    return next(b for b in at.button if b.label.startswith(label_start))


def test_store_tab_renders(app):
    at, _ = app
    assert any(m.label == "Products" and m.value == "40" for m in at.metric)


def test_plan_approve_reject_apply(app):
    at, out = app
    _button(at, "Plan day 1").click().run()
    assert not at.exception, at.exception
    assert (out / "day_1_plan.json").exists()
    radios = [r for r in at.radio if r.key and r.key.startswith("dec_1_")]
    assert len(radios) >= 2
    radios[0].set_value("approve")
    radios[1].set_value("reject")
    at.run()
    _button(at, "Apply decisions").click().run()
    assert not at.exception, at.exception
    dec = json.loads((out / "day_1_decisions.json").read_text())
    assert sorted(dec.values()) == ["approve", "reject"]
    assert (out / "day_1_report.json").exists()
    assert any(m.label == "Account day" and m.value == "1" for m in at.metric)


def test_race_and_ask(app):
    at, out = app
    _button(at, "Run the 14-day race").click().run()
    assert not at.exception, at.exception
    assert (out / "race.csv").exists()
    _button(at, "Which category gives the most").click().run()
    assert not at.exception, at.exception
    assert any("[D1]" in m.value for m in at.markdown)


def test_objectives_editor_saves_growth_products(app):
    at, out = app
    ms = next(m for m in at.multiselect if m.label == "Growth products")
    ms.set_value(["P25"]).run()
    _button(at, "Save objectives").click().run()
    assert not at.exception, at.exception
    obj = json.loads((out / "objectives.json").read_text())
    assert obj["P25"]["mode"] == "growth" and 0.1 <= obj["P25"]["max_acos"] <= 0.8


def test_consultant_tab_renders_ranked_report(cfg, fresh_store, monkeypatch):
    from agent.consultant import run_consultant
    from sim.platform import AdPlatform
    from sim.runner import make_strategy
    from tests.test_consultant import consultant_llm
    p = AdPlatform(cfg, fresh_store, persist=True)
    s = make_strategy("rules", cfg)
    for d in range(1, 8):
        s.act(p, d)
        p.step()
    run_consultant(p, 7, cfg, consultant_llm(), out=fresh_store)
    monkeypatch.setenv("BIDPILOT_OUT", str(fresh_store))
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    heads = [m.value for m in at.markdown if m.value.startswith("### #")]
    assert heads and heads[0].startswith("### #1")
    assert any("**Risk:**" in m.value for m in at.markdown)
    assert any(c.value.startswith("External evidence") for c in at.caption)
