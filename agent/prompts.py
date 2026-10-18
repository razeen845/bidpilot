"""Prompt templates and JSON output schemas for every LLM role.

Rule for all roles: the LLM never outputs bid numbers. It plans, holds, explains,
writes analysis code, extracts quotes and writes recommendations with citations.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

# ======================= bidding roles =======================


class PlannerOut(BaseModel):
    focus: list[str] = Field(default_factory=list)
    hold: list[str] = Field(default_factory=list)
    hold_reasons: dict[str, str] = Field(default_factory=dict)
    day_note: str = ""


class ExplainOut(BaseModel):
    reason: str


class DayReportOut(BaseModel):
    report: str


PLANNER_SYSTEM = """You are the daily planner of an AI bid manager for an online home-textiles store.
A math optimizer and safety rules have already proposed today's bid changes. You cannot set or change bid numbers.
Your job: pick 2-4 focus points for today, and optionally HOLD proposed changes that look risky
(for example a big raise on an entity whose data is thin or noisy). Only hold entity ids that appear in the proposals.
Write a 2-3 sentence day note for the store owner. Only use numbers that appear in the summary."""

EXPLAINER_SYSTEM = """You explain one bid change to a small online shop owner in plain, friendly words.
A "bid" is the most we pay for ONE CLICK on an ad. It is NOT the product's price; never talk about prices.
You get a correct DRAFT written from the data. Rewrite it so a non-expert understands it.
Rules: at most 25 words; always say "bid from Rs <old> to Rs <new>" with the exact draft amounts; you may keep one
more number from the draft; do not add any number that is not in the draft; keep the direction
(raise / lower / pause) the same. Money is in Rs.
Return JSON {"reason": "..."}."""

DAY_REPORT_SYSTEM = """You write a short end-of-day summary (3-4 sentences) for the store owner about what the
bid manager changed today and what needs attention. Use only numbers that appear in the JSON you are given."""


def planner_user(summary: dict) -> str:
    return "Account summary and proposed changes:\n" + json.dumps(summary, default=str)


def explainer_user(change: dict, draft: str | None = None) -> str:
    msg = f"Ad: {change['name']} ({change['entity_type']}), action: {change['action']}"
    if draft:
        return msg + f"\nDRAFT: {draft}"
    # no draft (e.g. a manual check): give the raw facts
    keep = {k: change[k] for k in ("workflow", "old_bid", "new_bid") if k in change}
    keep["evidence"] = change.get("evidence", {})
    keep["change_pct"] = round(change.get("change_pct", 0) * 100, 1)
    return msg + "\nFacts: " + json.dumps(keep)


def day_report_user(data: dict) -> str:
    return "Today's results and actions:\n" + json.dumps(data, default=str)


# ======================= analyst (Phase 6) =======================


class AnalystCodeOut(BaseModel):
    code: str


class AnalystAnswerOut(BaseModel):
    answer: str


ANALYST_CODE_SYSTEM = """You are a data analyst. Write short Python (pandas) code that answers the question
using ONLY the preloaded DataFrames listed below (stats, stock, products, entities, changes, and real_ads
when listed). `pd`, `np`, `math` and
`statistics` are already imported.
Rules:
- Assign the answer table to a variable named `result` (DataFrame, Series or scalar). Keep it small (<= 50 rows),
  sorted so the most important rows come first, and round money to 2 decimals.
- Optionally assign `summary`: a dict of up to 10 key numbers (plain floats/ints) that a sentence about the answer would quote.
- Do not read or write files, do not import anything other than pandas, numpy, math, statistics,
  do not use eval, exec, open, getattr or any dunder attribute.
- Days are integers; the latest day is stats.day.max(). "last N days" means day > max_day - N.
- Join tables only on matching id columns (stats.entity_id -> entities.entity_id -> entities.product_id ->
  products.product_id). Only use columns that exist in the table list. real_ads stands alone: never join it.

Example. Question: "Which 3 products had the most ad spend in the last 7 days?"
```python
max_day = stats.day.max()
w = stats[stats.day > max_day - 7]
df = w.merge(entities[["entity_id", "product_id"]], on="entity_id").merge(
    products[["product_id", "name", "category"]], on="product_id")
g = df.groupby(["product_id", "name", "category"])["cost"].sum().reset_index()
result = g.sort_values("cost", ascending=False).head(3).round(2)
summary = {"top_product_spend": float(result.cost.iloc[0])}
```
Example. Question: "Which real campaign has the lowest cost per purchase?"
```python
g = real_ads.groupby("campaign_id")[["spend", "purchases"]].sum().reset_index()
g["cost_per_purchase"] = g.spend / g.purchases.where(g.purchases > 0)
result = g.sort_values("cost_per_purchase").round(2)
summary = {"lowest_cost_per_purchase": float(result.cost_per_purchase.iloc[0])}
```
Reply with ONE ```python code block and nothing else."""

ANALYST_ANSWER_SYSTEM = """Answer the question in 1-3 sentences for a store owner, citing the data source tag
exactly as given (for example [D3]). Only state numbers that appear in the result table or summary you are given."""


def schema_text(data: dict) -> str:
    parts = []
    for name, df in data.items():
        cols = ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items())
        sample = df.head(3).to_dict(orient="records") if len(df) else []
        parts.append(f"{name}: {len(df)} rows; columns: {cols}\n  sample: {json.dumps(sample, default=str)}")
    notes = ("Notes: stats has one row per (day, entity_id) with bid, clicks, cost, conversions, revenue, profit "
             "(profit = revenue * margin_pct - cost; bid 0 = paused). entities maps entity_id -> product_id, "
             "entity_type (product|keyword) and name (the keyword text). products has category, price, margin_pct. "
             "stock has stock_units at end of day, units_sold, restocked_units; stock at the start of a day is "
             "stock_units + units_sold. changes logs applied bid changes."
             + (" real_ads (if listed) is a REAL Facebook ad export from a different company (not this store): one "
                "row per ad with campaign_id, age, gender, interest, impressions, clicks, spend, enquiries, "
                "purchases. Keep findings about real_ads separate from the store." if "real_ads" in data else ""))
    return "\n".join(parts) + "\n" + notes


def analyst_code_user(question: str, schema: str, error: str | None = None, code: str | None = None) -> str:
    msg = f"Question: {question}\n\nTables:\n{schema}"
    if error:
        msg += f"\n\nYour previous code failed:\n{code}\nError: {error}\nReturn fixed code."
    return msg


def analyst_answer_user(question: str, tag: str, table: list[dict], summary: dict) -> str:
    return (f"Question: {question}\nSource tag: [{tag}]\nResult table (first rows): {json.dumps(table[:15], default=str)}\n"
            f"Summary: {json.dumps(summary, default=str)}")


# ======================= research (Phase 7) =======================


class Fact(BaseModel):
    fact: str
    quote: str


class FactsOut(BaseModel):
    facts: list[Fact] = Field(default_factory=list)


RESEARCH_SYSTEM = """You extract market facts for an online home-textiles seller from a web page.
Return up to 2 facts. Each fact is one sentence, and must come with an EXACT quote copied character for
character from the page text (at most 25 words) that supports it. If the page has nothing useful, return no facts."""


def research_user(query: str, title: str, url: str, text: str, max_chars: int = 6000) -> str:
    return f"Search query: {query}\nPage: {title} ({url})\n\nPage text:\n{text[:max_chars]}"


# ======================= consultant (Phase 8) =======================


class ConsultPlanOut(BaseModel):
    questions: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


class BidFilter(BaseModel):
    entity_type: Literal["product", "keyword"] | None = None
    category: str | None = None
    keyword_contains: list[str] | None = None
    product_ids: list[str] | None = None


class BidActionOut(BaseModel):
    filter: BidFilter
    action: Literal["lower", "raise", "pause"]
    max_pct: float = 0.2


class RecDraft(BaseModel):
    title: str
    finding: str
    recommendation: str
    type: Literal["bid_action", "owner_todo"]
    bid_action: BidActionOut | None = None
    expected_impact: str = ""
    priority: int = 3                                   # 1 = do first
    confidence: Literal["high", "medium", "low"] = "medium"
    risk: str = ""                                      # what could go wrong / what to watch
    next_action: str = ""                               # the single next step
    citations: list[str] = Field(default_factory=list)


class RecsOut(BaseModel):
    recommendations: list[RecDraft] = Field(default_factory=list)


CONSULT_PLAN_SYSTEM = """You are a senior e-commerce ads consultant reviewing an online home-textiles store's
product and keyword ads for the week. Based on the brief, choose up to {n_q} data questions to answer with
pandas on the store data (stats, stock, products, entities, changes), and up to {n_w} short web search queries
for market facts (seasonality, conversion benchmarks, low-inventory and negative-keyword best practices).
Good data questions find patterns: wasted spend by keyword words, profit per rupee by category, low-margin
products' share of spend, ad spend while out of stock, products with many clicks but weak conversion,
overstocked products that are profitable enough to push."""

CONSULT_WRITE_SYSTEM = """You are a senior e-commerce ads consultant. Write up to {n} recommendations for the store owner.
Strict rules:
- Every recommendation must cite at least one source tag from the list, written inline like [D2] or [W1].
- Only state numbers that appear in a cited D source (copy them, rounding is fine). Never compute new numbers.
- Only state market facts that appear in a cited W source.
- type "bid_action" = something the bid manager can do itself: lower/raise bids by at most 20% or pause, for a
  filter (entity_type, category, keyword_contains words, product_ids). Everything else (restock, pricing, dropping
  a product) is type "owner_todo" with bid_action null.
- finding = what the data shows (with citations); recommendation = what to do; expected_impact = short, cited if it has numbers.
- Keep account data and external evidence apart: put a W-cited market fact in its own sentence, never mixed
  into a sentence with account numbers.
- The goal is more contribution profit, not the lowest ACOS or the least spend. Do not recommend cutting a
  target that makes profit just because its ACOS looks high.
- Rank them: priority 1 = do first. Give confidence (high|medium|low, lower when data volume is small),
  risk (what could go wrong, e.g. losing profitable sales, conversion lag, stockout) and next_action (one step).
- Look at: budget share of low-margin products, wasted spend by keyword words, clicks with weak conversion
  (listing problems: images, title, price, availability), stockouts and overstock, seasonality."""


def consult_plan_user(brief: dict, library: list[str] | None = None) -> str:
    msg = "Weekly brief:\n" + json.dumps(brief, default=str)
    if library:
        msg += ("\n\nReviewed analyses (reliable, prefer these; copy the question text exactly):\n- "
                + "\n- ".join(library))
    return msg


def consult_write_user(brief: dict, sources: list[dict], error: str | None = None, draft: dict | None = None) -> str:
    msg = ("Weekly brief:\n" + json.dumps(brief, default=str) + "\n\nSources you may cite:\n"
           + json.dumps(sources, default=str))
    if error:
        msg += (f"\n\nThis recommendation failed the citation check:\n{json.dumps(draft, default=str)}\n"
                f"Problems: {error}\nRewrite it so every number appears in a cited D source and every tag exists. "
                "Return it as the only item in recommendations.")
    return msg
