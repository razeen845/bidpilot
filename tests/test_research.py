import httpx
import pytest

import agent.research as R
from agent.citations import CitationRegistry
from tests.fakes import FakeLLM

PAGE = ("Bedding demand rises before winter as shoppers prepare for colder nights. "
        "Online home textile sales grew 12% last year according to the survey. "
        "Retailers should pause search ads for products that are out of stock.")


def test_quote_check_exact_and_normalised():
    assert R.quote_check("Bedding demand rises before winter", PAGE)[0]
    assert R.quote_check("bedding   demand rises\nbefore winter.", PAGE)[0]
    page = "It’s best to “pause” ads"
    assert R.quote_check("It's best to \"pause\" ads", page)[0]


def test_quote_check_rejects_invented_or_long():
    assert not R.quote_check("Bedding demand triples before winter", PAGE)[0]
    assert not R.quote_check("", PAGE)[0]
    long = " ".join(PAGE.split()[:30])
    assert not R.quote_check(long, PAGE)[0]


def test_live_research_keeps_only_verified_quotes(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "search", lambda q, c, n: [{"url": "https://example.com/a", "title": "A", "content": PAGE * 3}])
    llm = FakeLLM({"research": lambda s, u: {"facts": [
        {"fact": "Bedding demand rises before winter.", "quote": "Bedding demand rises before winter"},
        {"fact": "Sales tripled.", "quote": "Online home textile sales tripled last year"},
    ]}})
    c = {**cfg, "research": {**cfg["research"], "mode": "live", "api_key": "x"}}
    run = R.research(["home textile seasonality"], c, llm=llm, out=tmp_path)
    assert [s["quote"] for s in run["sources"]] == ["Bedding demand rises before winter"]
    assert run["sources"][0]["id"] == "W1" and run["sources"][0]["verified"]
    assert len(run["rejected"]) == 1 and run["rejected"][0]["check"] == "quote not found in page text"
    # cache mode now replays that run and re-verifies it against the saved page text
    cached = R.research(["anything"], cfg, llm=None, out=tmp_path)
    assert cached["mode"] == "cache" and len(cached["sources"]) == 1


def test_cache_mode_needs_no_network(cfg, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "get", boom)
    monkeypatch.setattr(httpx, "post", boom)
    run = R.research(["home textile online sales seasonality"], cfg, llm=None, out=tmp_path)
    assert run["mode"] == "cache"
    assert len(run["sources"]) >= 5
    for s in run["sources"]:
        assert s["verified"] and s["url"].startswith("https://") and s["retrieved_at"]
        assert R.quote_check(s["quote"], R.page_text_for(s, tmp_path))[0]


def test_tampered_cache_quote_is_dropped(cfg, tmp_path):
    import json
    import shutil
    src = R.latest_run_file(tmp_path)
    rdir = tmp_path / "research"
    shutil.copytree(src.parent / "pages", rdir / "pages")
    run = json.loads(src.read_text(encoding="utf-8"))
    run["sources"][0]["quote"] = "This sentence was never on the page"
    (rdir / "run_99999999_000000.json").write_text(json.dumps(run), encoding="utf-8")
    loaded = R.load_cached(cfg, tmp_path)
    assert len(loaded["sources"]) == len(run["sources"]) - 1


# ---------------- citation registry ----------------

def _registry():
    reg = CitationRegistry()
    reg.add({"id": "D1", "question": "zero sales in the last 14 days?", "result_table": [{"word": "free", "spend": 9812.45}],
             "result_summary": {"share_pct": 0.234}})
    reg.add({"id": "W1", "fact": "Bedding demand rises before winter.", "quote": "demand rises 30% before winter",
             "verified": True})
    reg.add({"id": "W2", "fact": "x", "quote": "y", "verified": False})
    return reg


def test_registry_accepts_backed_text():
    reg = _registry()
    assert reg.check_text("Keywords with 'free' spent Rs 9,812 over 14 days with no sales [D1].", ["D1"]).ok
    assert reg.check_text("That is 23.4% of spend [D1]; demand rises 30% before winter [W1].").ok


def test_registry_rejects_wrong_currency():
    reg = _registry()
    assert not reg.check_text("Keywords with 'free' spent $9,812 [D1].", ["D1"]).ok
    reg.add({"id": "D2", "question": "Which real Facebook ads spent the most?", "result_table": [{"spend": 541.7}],
             "result_summary": {}})
    assert reg.check_text("One real ad spent $541.70 with no purchases [D2].", ["D2"]).ok


@pytest.mark.parametrize("text,ids,err", [
    ("Keywords spent Rs 5,000 [D1].", ["D1"], "numbers not found"),
    ("Something happened [D7].", [], "unknown source"),
    ("Something happened.", [], "no citations"),
    ("Experts agree [W2].", ["W2"], "not a verified"),
])
def test_registry_rejects(text, ids, err):
    res = _registry().check_text(text, ids)
    assert not res.ok and any(err in e for e in res.errors)
