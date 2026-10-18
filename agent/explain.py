"""Explanations for bid changes, guarded by a number check.

The maths writes a correct draft (template reason); the LLM rewrites it in plain words. Every number in
the LLM's sentence must appear in that draft (rounding allowed), so a number can't be attached to the wrong
meaning. Otherwise the draft is used as-is and the rejection is logged.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from pathlib import Path

from agent import prompts as P
from engine.models import Change
from settings import out_dir

NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")
TAG_RE = re.compile(r"\[[DW]\d+\]")
ALWAYS_OK = {0.0, 1.0, 7.0}   # "0 sales", "Rs 1 spent", "7 days"
MAX_WORDS = 35


def extract_numbers(text: str) -> list[float]:
    text = TAG_RE.sub(" ", text or "")
    out = []
    for m in NUM_RE.finditer(text):
        s = m.group(0).replace(",", "")
        try:
            out.append(abs(float(s)))
        except ValueError:
            continue
    return out


def numeric_values(obj) -> list[float]:
    """All numbers inside a nested dict/list (strings that look like numbers count too)."""
    vals: list[float] = []

    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            vals.append(abs(float(o)))
        elif isinstance(o, str):
            vals.extend(extract_numbers(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(obj)
    return vals


def number_matches(n: float, values: Iterable[float]) -> bool:
    if n in ALWAYS_OK:
        return True
    for v in values:
        if v != v:  # NaN
            continue
        for cand in (v, v * 100 if v <= 10 else None):   # fractions may be written as percentages
            if cand is None:
                continue
            # n is cand rounded to d decimals (any rounding convention)
            for d in (0, 1, 2):
                if abs(n - round(n, d)) < 1e-9 and abs(n - cand) <= 0.5 * 10 ** -d + 1e-9:
                    return True
    return False


def check_numbers(text: str, values: Iterable[float]) -> list[float]:
    """Numbers in `text` that do not match any allowed value."""
    values = list(values)
    return [n for n in extract_numbers(text) if not number_matches(n, values)]


def change_values(c: Change) -> list[float]:
    ev = c.evidence.model_dump()
    vals = [c.old_bid, c.new_bid, abs(c.change_pct), abs(c.new_bid - c.old_bid)] + list(ev.values())
    return [abs(float(v)) for v in vals]


def template_reason(c: Change) -> str:
    e = c.evidence
    if c.workflow == "stock_guard":
        if c.action == "pause":
            return f"Out of stock. Pausing so ads stop spending; Rs {e.spend_7d:,.0f} spent in the last 7 days."
        if c.action == "resume":
            return f"Back in stock with {e.days_of_stock:.1f} days of stock. Resuming at Rs {c.new_bid:.2f}."
        return (f"Only {e.days_of_stock:.1f} days of stock left. Bid cut {abs(c.change_pct) * 100:.0f}% "
                f"from Rs {c.old_bid:.2f} to Rs {c.new_bid:.2f} until restock.")
    if c.workflow == "waste":
        act = (c.platform_action or "pause").capitalize()
        return (f"Spent Rs {e.spend_7d:,.0f} on {e.clicks_7d:.0f} clicks in 7 days with 0 sales; at this volume "
                f"zero sales is very unlikely to be bad luck. {act}.")
    if c.workflow == "experiment":
        return ("Approved switchback test: ads are OFF today to measure how many sales would happen anyway."
                if c.action == "pause" else "Approved switchback test: ads back ON at the previous bid.")
    if c.workflow == "rollback":
        return (f"Rollback: profit fell after recent bid moves, so the bid returns from Rs {c.old_bid:.2f} to "
                f"Rs {c.new_bid:.2f} and stays there for a few days.")
    if c.workflow == "consultant":
        verb = {"pause": "Pausing", "lower": "Lowering", "raise": "Raising"}.get(c.action, "Changing")
        return (f"Accepted consultant recommendation. {verb} bid from Rs {c.old_bid:.2f} to Rs {c.new_bid:.2f}; "
                f"7-day spend Rs {e.spend_7d:,.0f}.")
    if c.kpi == "acos":
        trend = f"ACOS {e.acos_7d * 100:.1f}% this week vs {e.acos_prev_7d * 100:.1f}% the week before"
    else:
        trend = f"ROAS {e.roas_7d:.2f} this week vs {e.roas_prev_7d:.2f} the week before"
    if c.objective == "growth" and c.action == "raise":
        return (f"Growth objective: {trend}. Raising Rs {c.old_bid:.2f} to Rs {c.new_bid:.2f} for more sales while "
                f"expected ACOS stays under your cap.")
    if c.action == "raise":
        return (f"{trend}, profit Rs {e.profit_7d:,.0f}. The response curve says Rs {c.new_bid:.2f} "
                f"earns more profit than Rs {c.old_bid:.2f}.")
    if c.role == "proven":
        return (f"Proven profit maker (Rs {e.profit_7d:,.0f} in 7 days), so the cut is capped: Rs {c.old_bid:.2f} to "
                f"Rs {c.new_bid:.2f}. {trend}.")
    return (f"{trend}, profit Rs {e.profit_7d:,.0f}. Lowering from Rs {c.old_bid:.2f} to "
            f"Rs {c.new_bid:.2f} cuts cost faster than it cuts sales.")


def _log_rejection(out: Path, c: Change, reason: str, bad: list, why: str) -> None:
    with open(out / "explain_rejections.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"entity_id": c.entity_id, "reason": reason, "bad_numbers": bad, "why": why}) + "\n")


async def explain_one(llm, c: Change, out: Path) -> Change:
    try:
        draft = template_reason(c)    # correct by construction; the LLM only makes it readable
        data = await llm.chat_json("explainer", P.EXPLAINER_SYSTEM, P.explainer_user(c.model_dump(), draft),
                                   P.ExplainOut, max_tokens=120)
        reason = data["reason"].strip()
        # every number must come from the draft (stricter than "anywhere in the evidence": no misattribution)
        bad = check_numbers(reason, extract_numbers(draft))
        why = "number check" if bad else "length" if not reason or len(reason.split()) > MAX_WORDS else None
        if why is None and c.action in ("raise", "lower", "resume"):
            got = extract_numbers(reason)        # a bid change must name both bids
            if not all(any(abs(n - b) < 0.006 for n in got) for b in (c.old_bid, c.new_bid)):
                why = "missing bid amounts"
        if why is None and re.search(r"\bprices?\b", reason, re.I):
            why = "talks about price, not the bid"
        if why is None and c.action == "pause" and re.search(r"\bbid from\b|\bto rs\.? ?0\b", reason, re.I):
            why = "a pause does not move the bid"
        if why:
            _log_rejection(out, c, reason, bad, why)
            c.reason, c.reason_source = template_reason(c), "template"
        else:
            c.reason, c.reason_source = reason, "llm"
    except Exception as e:  # noqa: BLE001
        _log_rejection(out, c, "", [], f"llm error: {e}"[:200])
        c.reason, c.reason_source = template_reason(c), "template"
    return c


async def explain_changes(llm, changes: list[Change], out=None) -> list[Change]:
    """Fill `reason` on every change; concurrent LLM calls when an LLM is given, templates otherwise."""
    if llm is None:
        for c in changes:
            c.reason, c.reason_source = template_reason(c), "template"
        return changes
    o = out_dir(out)
    return list(await asyncio.gather(*(explain_one(llm, c, o) for c in changes)))
