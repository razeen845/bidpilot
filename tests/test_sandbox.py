import json

import numpy as np
import pandas as pd
import pytest

from agent.analyst import PRESET_CODE, PRESET_QUESTIONS, ask
from sandbox.exec_runner import run_code
from sandbox.policy import check_code
from sim.platform import AdPlatform
from sim.runner import make_strategy
from tests.fakes import FakeLLM


@pytest.mark.parametrize("code", [
    "import os",
    "import subprocess",
    "from os import path",
    "import pandas.io",
    "open('x.txt')",
    "__import__('os')",
    "getattr(stats, 'x')",
    "x = stats.__class__",
    "stats.to_csv('x.csv')",
    "pd.read_csv('x.csv')",
    "eval('1+1')",
    "exec('x=1')",
    "stats.query('day.__class__ > 0')",
    "np.save('x', 1)",
    "x = '__class__'",
    "class A: pass",
])
def test_policy_blocks(code):
    assert not check_code(code).ok


def test_policy_allows_query():
    assert check_code("result = stats.query('day > 0 and clicks >= 10').head()").ok


def test_policy_allows_normal_pandas():
    code = ("import numpy as np\nimport math\nw = stats[stats.day > stats.day.max() - 7]\n"
            "result = w.groupby('entity_id').agg(spend=('cost', 'sum')).sort_values('spend').head(5)\n"
            "summary = {'n': len(result), 'x': math.sqrt(4)}")
    assert check_code(code).ok, check_code(code).reasons


def test_presets_pass_policy():
    for q, code in PRESET_CODE.items():
        assert check_code(code).ok, (q, check_code(code).reasons)


@pytest.fixture(scope="module")
def sim_data(cfg, store):
    """Public data after 8 agent days (so there are changes and stockouts)."""
    p = AdPlatform(cfg, store)
    s = make_strategy("rules", cfg)
    for d in range(1, 9):
        s.act(p, d)
        p.step()
    return p.public_data()


def test_run_code_ok_and_truncates(cfg, sim_data):
    r = run_code("result = stats.head(80)\nsummary = {'rows': 80}", sim_data, cfg)
    assert r["ok"], r
    assert len(r["result_table"]) == cfg["sandbox"]["max_result_rows"]
    assert r["truncated"] and r["total_rows"] == 80
    assert r["result_summary"] == {"rows": 80}


def test_run_code_policy_reject_is_not_run(cfg, sim_data):
    r = run_code("import os\nresult = os.listdir('.')", sim_data, cfg)
    assert not r["ok"] and "policy rejected" in r["error"]


def test_run_code_timeout(cfg, sim_data):
    c = json.loads(json.dumps(cfg))
    c["sandbox"]["timeout_s"] = 2
    r = run_code("x = 0\nwhile True:\n    x += 1", sim_data, c)
    assert not r["ok"] and "timeout" in r["error"]


def test_run_code_errors_reported(cfg, sim_data):
    r = run_code("x = 1", sim_data, cfg)
    assert not r["ok"] and "result" in r["error"]
    r = run_code("result = stats['nope']", sim_data, cfg)
    assert not r["ok"] and "KeyError" in r["error"]


def test_sandbox_refuses_hidden_columns(cfg, sim_data):
    bad = dict(sim_data)
    bad["entities"] = sim_data["entities"].assign(cvr=0.1)
    with pytest.raises(ValueError):
        run_code("result = 1", bad, cfg)


# ---------------- preset answers vs hand-written pandas ----------------

def _table(src):
    return pd.DataFrame(src["result_table"])


def test_preset_zero_sale_patterns(cfg, sim_data, tmp_path):
    src = ask(PRESET_QUESTIONS[0], sim_data, cfg, llm=None, out=tmp_path)
    assert src["ok"] and src["id"] == "D1" and src["code_author"] == "reviewed"
    st, en = sim_data["stats"], sim_data["entities"]
    last = st[st.day > st.day.max() - 14]
    kw = en[en.entity_type == "keyword"].set_index("entity_id").name
    tot = last.groupby("entity_id")[["cost", "conversions"]].sum()
    tot = tot[tot.index.isin(kw.index)]
    zero = tot[tot.conversions == 0]
    spend_free = sum(c for eid, c in zero.cost.items() if "free" in kw[eid].split())
    t = _table(src).set_index("word")
    assert "free" in t.index
    assert abs(t.loc["free", "zero_sale_spend"] - round(spend_free, 2)) < 0.01
    assert abs(src["result_summary"]["zero_sale_spend_total"] - round(zero.cost.sum(), 2)) < 0.01
    assert "[D1]" in src["answer"]


def test_preset_category_profit_per_rupee(cfg, sim_data, tmp_path):
    src = ask(PRESET_QUESTIONS[1], sim_data, cfg, out=tmp_path)
    st, en, pr = sim_data["stats"], sim_data["entities"], sim_data["products"]
    last = st[st.day > st.day.max() - 14]
    cat = last.entity_id.map(en.set_index("entity_id").product_id).map(pr.set_index("product_id").category)
    ppr = {c: last.profit[cat == c].sum() / last.cost[cat == c].sum() for c in cat.unique()}
    t = _table(src)
    assert t.category.iloc[0] == max(ppr, key=ppr.get)
    for r in t.itertuples():
        assert abs(r.profit_per_rupee - round(ppr[r.category], 2)) < 0.011


def test_preset_out_of_stock_spend(cfg, sim_data, tmp_path):
    src = ask(PRESET_QUESTIONS[2], sim_data, cfg, out=tmp_path)
    st, en, sk = sim_data["stats"], sim_data["entities"], sim_data["stock"]
    start = sk.assign(start=sk.stock_units + sk.units_sold).set_index(["day", "product_id"]).start
    pid = st.entity_id.map(en.set_index("entity_id").product_id)
    key = list(zip(st.day, pid))
    oos = np.array([start.get(k, 1) == 0 for k in key])
    expected = st.cost[oos].sum()
    assert expected > 0      # rules keeps advertising products with no stock
    assert abs(src["result_summary"]["total_ad_spend_while_out"] - round(expected, 2)) < 0.01


def test_preset_low_margin_share(cfg, sim_data, tmp_path):
    src = ask(PRESET_QUESTIONS[3], sim_data, cfg, out=tmp_path)
    st, en, pr = sim_data["stats"], sim_data["entities"], sim_data["products"]
    last = st[st.day > st.day.max() - 14]
    margin = last.entity_id.map(en.set_index("entity_id").product_id).map(pr.set_index("product_id").margin_pct)
    share = 100 * last.cost[margin < 0.2].sum() / last.cost.sum()
    assert abs(src["result_summary"]["low_margin_spend_share_pct"] - round(share, 2)) < 0.01
    t = _table(src)
    assert (t.margin_pct < 0.2).all() and t.cost.is_monotonic_decreasing


# ---------------- analyst with a (fake) LLM ----------------

def test_analyst_llm_fix_attempt_and_number_check(cfg, sim_data, tmp_path):
    codes = iter(["result = stats['nope']", "result = stats.groupby('day').cost.sum().tail(3)\nsummary = {'days': 3}"])
    llm = FakeLLM({"analyst": lambda s, u: {"code": next(codes)} if "Tables:" in u
                   else {"answer": "Spend was Rs 123456789 on the last day [D1]."}})
    src = ask("What did we spend on the last 3 days?", sim_data, cfg, llm=llm, out=tmp_path)
    assert src["ok"] and src["code_author"] == "llm"
    assert len(src["attempts"]) == 2 and not src["attempts"][0]["ok"]
    assert src["answer_source"] == "template"            # made-up number rejected
    assert src["answer_rejected"]["bad_numbers"] == [123456789.0]
    assert (tmp_path / "analysis" / "D1.json").exists()


def test_free_text_needs_llm(cfg, sim_data, tmp_path):
    r = ask("Anything interesting?", sim_data, cfg, llm=None, out=tmp_path)
    assert not r["ok"] and "preset" in r["error"]


def test_known_questions_use_reviewed_code_even_with_llm(cfg, sim_data, tmp_path):
    """The reviewed library wins for known questions: the LLM only writes the answer sentence, never the code."""
    asked = []
    llm = FakeLLM({"analyst": lambda s, u: asked.append("code") or {"code": "result = 1"} if "Tables:" in u
                   else {"answer": "See [D1]."}})
    src = ask(PRESET_QUESTIONS[1], sim_data, cfg, llm=llm, out=tmp_path)
    assert src["ok"] and src["code_author"] == "reviewed" and src["code"] == PRESET_CODE[PRESET_QUESTIONS[1]]
    assert asked == [] and src["answer_source"] == "llm"
