"""Ask the data: question -> code -> policy -> sandbox -> saved [D#] source -> cited answer.

Known questions use a library of REVIEWED analyses (PRESET_CODE): reliable and fast; the LLM (if any) only writes
the plain-English answer. New free-text questions get LLM-written code (one fix attempt on error). Without an LLM
only the library works. Every D source records who wrote the code (`code_author`: reviewed | llm).
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path

from agent import prompts as P
from agent.explain import check_numbers, numeric_values
from agent.llm import run_sync
from sandbox.exec_runner import run_code
from settings import out_dir, write_json

_ID_LOCK = threading.Lock()

PRESET_QUESTIONS = [
    "Which keyword patterns spent the most with zero sales in the last 14 days?",
    "Which category gives the most profit per rupee of ad spend?",
    "How much ad money was spent on products while they were out of stock?",
    "Which low-margin products take the biggest share of the budget?",
    "Which products get many clicks but convert below their category average?",
    "Which overstocked products are profitable enough to push harder?",
]
# questions on the REAL ad export (data/real/KAG_conversion_data.csv); shown when it is available
REAL_ADS_QUESTIONS = [
    "Which real Facebook ads spent the most with zero purchases?",
    "Which age group and gender get the cheapest purchases in the real Facebook ads?",
]

PRESET_CODE = {
    PRESET_QUESTIONS[0]: '''max_day = stats.day.max()
w = stats[stats.day > max_day - 14]
kw = entities[entities.entity_type == "keyword"][["entity_id", "name"]]
per = w.groupby("entity_id")[["cost", "conversions"]].sum().reset_index().merge(kw, on="entity_id")
per["zero_sales"] = per.conversions == 0
per["word"] = per["name"].str.split()
words = per.explode("word")
g = words.groupby("word").agg(keywords=("entity_id", "nunique"), zero_sale_keywords=("zero_sales", "sum"))
g["zero_sale_spend"] = words[words.zero_sales].groupby("word").cost.sum()
g = g.fillna(0).reset_index()
g["zero_sale_share"] = (g.zero_sale_keywords / g.keywords).round(2)
result = g[(g.zero_sale_keywords >= 2) & (g.zero_sale_share >= 0.5)].sort_values("zero_sale_spend", ascending=False).head(10)
result["zero_sale_spend"] = result["zero_sale_spend"].round(2)
zero = per[per.zero_sales]
summary = {"zero_sale_keywords": int(len(zero)), "zero_sale_spend_total": round(float(zero.cost.sum()), 2)}
if len(result):
    summary["top_word_spend"] = float(result.zero_sale_spend.iloc[0])
''',
    PRESET_QUESTIONS[1]: '''max_day = stats.day.max()
w = stats[stats.day > max_day - 14]
df = w.merge(entities[["entity_id", "product_id"]], on="entity_id").merge(products[["product_id", "category"]], on="product_id")
g = df.groupby("category")[["cost", "revenue", "profit"]].sum()
g["profit_per_rupee"] = g.profit / g.cost
result = g.sort_values("profit_per_rupee", ascending=False).round(2).reset_index()
summary = {"best_profit_per_rupee": float(result.profit_per_rupee.iloc[0]),
           "worst_profit_per_rupee": float(result.profit_per_rupee.iloc[-1]),
           "total_ad_spend_14d": round(float(g.cost.sum()), 2)}
''',
    PRESET_QUESTIONS[2]: '''s = stock.copy()
s["start_stock"] = s.stock_units + s.units_sold
oos = s[s.start_stock == 0][["day", "product_id"]]
days = oos.groupby("product_id").day.agg(["count", "min", "max"]).rename(columns={"count": "days_out_of_stock", "min": "first_day", "max": "last_day"})
spent = stats.merge(entities[["entity_id", "product_id"]], on="entity_id").merge(oos, on=["day", "product_id"])
days["ad_spend_while_out"] = spent.groupby("product_id").cost.sum()
days = days.fillna(0).reset_index().merge(products[["product_id", "name"]], on="product_id")
result = days.sort_values(["ad_spend_while_out", "days_out_of_stock"], ascending=False).round(2)
summary = {"total_ad_spend_while_out": round(float(days.ad_spend_while_out.sum()), 2),
           "products_out_of_stock": int(len(days)), "total_days_out_of_stock": int(days.days_out_of_stock.sum())}
''',
    PRESET_QUESTIONS[3]: '''max_day = stats.day.max()
w = stats[stats.day > max_day - 14]
df = w.merge(entities[["entity_id", "product_id"]], on="entity_id").merge(products[["product_id", "name", "margin_pct"]], on="product_id")
total = df.cost.sum()
g = df.groupby(["product_id", "name", "margin_pct"])[["cost", "revenue", "profit"]].sum().reset_index()
g["spend_share_pct"] = 100 * g.cost / total
low = g[g.margin_pct < 0.2].sort_values("cost", ascending=False)
result = low.head(10).round(2)
summary = {"low_margin_products": int(len(low)),
           "low_margin_spend_share_pct": round(float(100 * low.cost.sum() / total), 2),
           "low_margin_ad_spend": round(float(low.cost.sum()), 2),
           "low_margin_profit": round(float(low.profit.sum()), 2)}
''',
    PRESET_QUESTIONS[4]: '''max_day = stats.day.max()
w = stats[stats.day > max_day - 14]
df = w.merge(entities[["entity_id", "product_id"]], on="entity_id").merge(products[["product_id", "name", "category"]], on="product_id")
per = df.groupby(["product_id", "name", "category"])[["clicks", "conversions", "cost"]].sum().reset_index()
cat = per.groupby("category")[["clicks", "conversions"]].sum()
per["cvr"] = per.conversions / per.clicks.where(per.clicks > 0)
per["category_cvr"] = per.category.map(cat.conversions / cat.clicks)
per["cvr_vs_category"] = per.cvr / per.category_cvr
weak = per[(per.clicks >= 100) & (per.cvr_vs_category < 0.6)].sort_values("clicks", ascending=False)
result = weak.round({"cvr": 4, "category_cvr": 4, "cvr_vs_category": 2, "cost": 2}).head(10)
summary = {"weak_converters": int(len(weak)), "their_clicks_14d": int(weak.clicks.sum()),
           "their_ad_spend_14d": round(float(weak.cost.sum()), 2)}
''',
    PRESET_QUESTIONS[5]: '''max_day = stats.day.max()
cur = stock[stock.day == max_day].set_index("product_id").stock_units
avg = stock[stock.day > max_day - 7].groupby("product_id").units_sold.mean()
dos = (cur / avg.where(avg > 0)).fillna(999.0)
w = stats[stats.day > max_day - 14].merge(entities[["entity_id", "product_id"]], on="entity_id")
ads = w.groupby("product_id")[["cost", "revenue", "profit"]].sum()
df = pd.DataFrame({"stock_units": cur, "days_of_stock": dos}).join(ads).fillna(0.0).reset_index()
df = df.merge(products[["product_id", "name", "margin_pct"]], on="product_id")
df["ad_profit_per_rupee"] = df.profit / df.cost.where(df.cost > 0)
over = df[df.days_of_stock > 60]
push = over[over.profit > 0].sort_values("ad_profit_per_rupee", ascending=False)
result = push.round({"days_of_stock": 1, "cost": 2, "revenue": 2, "profit": 2, "ad_profit_per_rupee": 2})
summary = {"overstocked_products": int(len(over)), "profitable_to_push": int(len(push)),
           "overstock_units": int(over.stock_units.sum())}
''',
}


PRESET_CODE.update({
    REAL_ADS_QUESTIONS[0]: '''ads = real_ads.copy()
zero = ads[ads.purchases == 0].sort_values("spend", ascending=False)
result = zero[["ad_id", "campaign_id", "age", "gender", "clicks", "spend", "enquiries"]].head(10)
summary = {"zero_purchase_ads": int(len(zero)), "zero_purchase_spend": round(float(zero.spend.sum()), 2),
           "total_spend": round(float(ads.spend.sum()), 2),
           "zero_purchase_share_pct": round(float(100 * zero.spend.sum() / ads.spend.sum()), 2)}
''',
    REAL_ADS_QUESTIONS[1]: '''g = real_ads.groupby(["age", "gender"])[["spend", "clicks", "purchases"]].sum().reset_index()
g["cost_per_purchase"] = g.spend / g.purchases.where(g.purchases > 0)
result = g.sort_values("cost_per_purchase").round(2)
summary = {"cheapest_cost_per_purchase": float(result.cost_per_purchase.iloc[0]),
           "most_expensive_cost_per_purchase": float(result.cost_per_purchase.max())}
''',
})


def _next_id(adir: Path) -> int:
    nums = [int(m.group(1)) for p in adir.glob("D*.json") if (m := re.fullmatch(r"D(\d+)\.json", p.name))]
    return max(nums, default=0) + 1


def save_source(src: dict, out=None) -> dict:
    adir = out_dir(out) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    with _ID_LOCK:
        n = _next_id(adir)
        src["id"] = f"D{n}"
        write_json((adir / f"D{n}.json"), src)
    return src


def load_source(sid: str, out=None) -> dict | None:
    p = out_dir(out) / "analysis" / f"{sid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def latest_for(question: str, out=None) -> dict | None:
    adir = out_dir(out) / "analysis"
    best = None
    for p in adir.glob("D*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("question") == question and (best is None or int(d["id"][1:]) > int(best["id"][1:])):
            best = d
    return best


def template_answer(src: dict) -> str:
    tag = f"[{src['id']}]"
    parts = [f"{k} {v:,}" if isinstance(v, (int, float)) else f"{k} {v}" for k, v in list(src["result_summary"].items())[:3]]
    if not parts and src["result_table"]:
        parts = [f"{k} {v}" for k, v in list(src["result_table"][0].items())[:4]]
    return f"{'; '.join(parts) or 'No rows matched'} {tag}."


def extract_code(text: str) -> str:
    """Code from a ```python block; also accepts {"code": ...} JSON or bare code."""
    text = (text or "").strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    if text.startswith("{"):
        try:
            return P.AnalystCodeOut.model_validate(json.loads(text)).code
        except (ValueError, TypeError):
            pass
    return text


async def _write_code(llm, question: str, schema: str, error=None, code=None) -> str:
    # Code inside a JSON string is hard for small models (they emit {"code": "1"}), so the code comes back as a
    # fenced block and is validated into the AnalystCodeOut schema; the policy check and sandbox still apply.
    # first try deterministic; the fix attempt gets some randomness so it does not repeat the same code
    text = await llm.chat_text("analyst", P.ANALYST_CODE_SYSTEM, P.analyst_code_user(question, schema, error, code),
                               max_tokens=900, temperature=0.0 if error is None else 0.5)
    return P.AnalystCodeOut(code=extract_code(text)).code


REAL_ADS_WORDS = re.compile(r"\b(real|facebook|meta|campaigns?|ad export)\b", re.I)


def relevant_tables(question: str, data: dict) -> dict:
    """Only show the real-ads table for questions about it (extra tables confuse small models)."""
    if "real_ads" in data and not REAL_ADS_WORDS.search(question):
        return {k: v for k, v in data.items() if k != "real_ads"}
    return data


async def ask_async(question: str, data: dict, cfg: dict, llm=None, out=None) -> dict:
    """Returns the saved D source (with `answer`), or {"ok": False, "error": ...}."""
    t0 = time.perf_counter()
    data = relevant_tables(question, data)
    attempts = []
    author = None
    code = None
    res = None
    if llm is not None and question not in PRESET_CODE:
        schema = P.schema_text(data)
        try:
            code, author = await _write_code(llm, question, schema), "llm"
            res = await asyncio.to_thread(run_code, code, data, cfg)
            attempts.append({"code": code, "ok": res["ok"], "error": res.get("error")})
            for _ in range(int(cfg["sandbox"].get("max_fix_attempts", 1))):
                if res["ok"]:
                    break
                code = await _write_code(llm, question, schema, res["error"], code)
                res = await asyncio.to_thread(run_code, code, data, cfg)
                attempts.append({"code": code, "ok": res["ok"], "error": res.get("error")})
        except Exception as e:  # noqa: BLE001
            attempts.append({"code": code, "ok": False, "error": f"llm error: {e}"[:300]})
            res = None
    if (res is None or not res["ok"]) and question in PRESET_CODE:
        code, author = PRESET_CODE[question], ("reviewed" if not attempts else "reviewed (llm code failed)")
        res = await asyncio.to_thread(run_code, code, data, cfg)
    if res is None:
        return {"ok": False, "question": question,
                "error": "no LLM available: only the preset questions work in --no-llm mode", "attempts": attempts}
    if not res["ok"]:
        return {"ok": False, "question": question, "error": res["error"], "attempts": attempts}

    src = save_source({
        "id": None,
        "question": question,
        "code": code,
        "code_author": author,
        "result_table": res["result_table"],
        "result_summary": res["result_summary"],
        "total_rows": res.get("total_rows"),
        "truncated": res.get("truncated", False),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sandbox_backend": res["sandbox_backend"],
        "data_through_day": int(data["stats"].day.max()) if len(data["stats"]) else None,
        "attempts": attempts,
    }, out)

    answer, source = template_answer(src), "template"
    if llm is not None:
        try:
            a = (await llm.chat_json("analyst", P.ANALYST_ANSWER_SYSTEM,
                                     P.analyst_answer_user(question, src["id"], src["result_table"], src["result_summary"]),
                                     P.AnalystAnswerOut, max_tokens=250))["answer"].strip()
            ok_tag = f"[{src['id']}]" in a
            bad = check_numbers(a, numeric_values([src["result_table"], src["result_summary"]]))
            if a and ok_tag and not bad:
                answer, source = a, "llm"
            else:
                src["answer_rejected"] = {"text": a, "bad_numbers": bad, "missing_tag": not ok_tag}
        except Exception as e:  # noqa: BLE001
            src["answer_rejected"] = {"error": str(e)[:200]}
    src.update({"ok": True, "answer": answer, "answer_source": source, "seconds": round(time.perf_counter() - t0, 3)})
    write_json((out_dir(out) / "analysis" / f"{src['id']}.json"), src)
    return src


def ask(question: str, data: dict, cfg: dict, llm=None, out=None) -> dict:
    return run_sync(ask_async(question, data, cfg, llm, out))
