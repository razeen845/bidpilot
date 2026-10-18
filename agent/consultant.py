"""Weekly consultant: brief -> plan (LLM) -> analytics + research in parallel -> recommendations (LLM)
-> citation check (one rewrite) -> saved report -> human accepts/rejects each recommendation.

Accepted bid_actions become out/pending_consultant_actions.json for the next day (they still pass guardrails
and approval rules). Accepted owner_todos go to out/todos.json.
"""
from __future__ import annotations

import asyncio
import json
import re
import time

from agent import prompts as P
from agent.analyst import ask_async
from agent.citations import CitationRegistry
from agent.llm import run_sync
from agent.research import research_async
from analysis.metrics import account_totals, entity_metrics, stock_metrics, top_spenders
from settings import out_dir, write_json, write_text

TAG_ANY = re.compile(r"[DdWw]\d+")


def build_brief(platform, cfg: dict) -> dict:
    day = platform.day
    stats, ents = platform.get_stats(), platform.get_entities()
    names = dict(zip(ents.entity_id, ents.name))
    ch = platform.get_changes()
    recent = ch[ch.day > day - 7] if not ch.empty else ch
    sm = stock_metrics(platform.get_stock_history(), day)
    sh = platform.get_stock_history()
    w = sh[(sh.day > day - 7) & (sh.day <= day)]
    m = entity_metrics(stats, day, 7, ents)
    return {
        "day": day,
        "last_7_days": account_totals(stats, day, 7),
        "previous_7_days": account_totals(stats, day - 7, 7),
        "changes_last_7_days": (recent.groupby(["workflow", "action"]).size().rename("n").reset_index()
                                .to_dict(orient="records") if not recent.empty else []),
        "approved_by": recent.approved_by.value_counts().to_dict() if not recent.empty else {},
        "paused_entities": [names[e] for e in sorted(platform.get_paused())][:20],
        "products_out_of_stock_last_7_days": sorted(set(w[w.stock_units <= 0].product_id)),
        "low_stock_now": sm[sm.days_of_stock < cfg["guardrails"]["low_stock_days"]].index.tolist(),
        "top_spenders_7d": [names[e] for e in top_spenders(m, 5)],
        "tables_available": ["stats", "stock", "products", "entities", "changes"],
    }


async def _plan(llm, brief: dict, cfg: dict) -> dict:
    c = cfg["consultant"]
    sys_msg = P.CONSULT_PLAN_SYSTEM.format(n_q=c["max_analysis_questions"], n_w=cfg["research"]["max_queries"])
    from agent.analyst import PRESET_QUESTIONS, REAL_ADS_QUESTIONS
    from analysis.real_ads import available as real_ads_available
    library = PRESET_QUESTIONS + (REAL_ADS_QUESTIONS if real_ads_available() else [])
    out = await llm.chat_json("consultant", sys_msg, P.consult_plan_user(brief, library), P.ConsultPlanOut,
                              max_tokens=500)
    return {"questions": [q for q in out["questions"] if q.strip()][: c["max_analysis_questions"]],
            "queries": [q for q in out["queries"] if q.strip()][: cfg["research"]["max_queries"]]}


async def fan_out(questions: list[str], queries: list[str], data: dict, cfg: dict, llm, out, parallel: bool = True):
    """Analytics questions and web research, at the same time (or one after another for the speed test)."""
    jobs = [ask_async(q, data, cfg, llm, out) for q in questions] + [research_async(queries, cfg, llm, out)]
    if parallel:
        results = await asyncio.gather(*jobs)
    else:
        results = [await j for j in jobs]
    return results[:-1], results[-1]


def clean_citations(raw) -> list[str]:
    """Models write "[D2]", " w1 ", "D2, W1" or "" - keep only well-formed D#/W# tags, sorted."""
    if isinstance(raw, str):
        raw = [raw]
    found = set()
    for item in raw or []:
        found.update(t.upper() for t in TAG_ANY.findall(str(item)))
    return sorted(found, key=lambda t: (t[0], int(t[1:])))


def _clean(rec: dict) -> dict:
    if rec.get("type") == "bid_action" and rec.get("bid_action"):
        rec["bid_action"]["max_pct"] = min(abs(float(rec["bid_action"].get("max_pct") or 0.2)), 0.2)
        f = rec["bid_action"].get("filter") or {}
        if not any(f.get(k) for k in ("entity_type", "category", "keyword_contains", "product_ids")):
            rec["type"], rec["bid_action"] = "owner_todo", None   # an unbounded filter is not actionable
    else:
        rec["bid_action"] = None
        rec["type"] = "owner_todo" if rec.get("type") != "bid_action" else rec["type"]
    rec["priority"] = int(min(max(int(rec.get("priority") or 3), 1), 5))
    rec["citations"] = clean_citations(rec.get("citations"))
    return rec


def _check(reg: CitationRegistry, rec: dict):
    text = f"{rec['finding']} {rec.get('expected_impact', '')} {rec.get('risk', '')}"
    res = reg.check_text(text, rec["citations"])
    if not rec["citations"] and not res.errors:
        res.ok, res.errors = False, ["no citations"]
    return res


async def _write(llm, brief, reg, cfg) -> list[dict]:
    sys_msg = P.CONSULT_WRITE_SYSTEM.format(n=cfg["consultant"]["max_recommendations"])
    out = await llm.chat_json("consultant", sys_msg, P.consult_write_user(brief, reg.catalog()), P.RecsOut,
                              max_tokens=2000)
    return out["recommendations"][: cfg["consultant"]["max_recommendations"]]


async def _rewrite(llm, brief, reg, rec, errors, cfg) -> dict | None:
    sys_msg = P.CONSULT_WRITE_SYSTEM.format(n=1)
    try:
        out = await llm.chat_json("consultant", sys_msg,
                                  P.consult_write_user(brief, reg.catalog(), "; ".join(errors), rec), P.RecsOut,
                                  max_tokens=800)
        return out["recommendations"][0] if out["recommendations"] else None
    except Exception:  # noqa: BLE001
        return None


def to_markdown(rep: dict) -> str:
    lines = [f"# Consultant report - day {rep['day']}", "", f"Generated {rep['created_at']}", ""]
    for r in rep["recommendations"]:
        lines += [f"## #{r.get('rank', '')} {r['id']}. {r['title']} ({r['status']}, confidence {r.get('confidence')})",
                  "", f"**Finding:** {r['finding']}", "",
                  f"**Recommendation:** {r['recommendation']}", "", f"**Expected impact:** {r['expected_impact']}",
                  "", f"**Risk:** {r.get('risk') or '-'}", "", f"**Next action:** {r.get('next_action') or '-'}",
                  "", f"Type: {r['type']} - citations: {', '.join(r['citations'])}", ""]
    lines += ["## Sources", ""]
    for s in rep["sources"]["D"]:
        lines.append(f"- [{s['id']}] {s['question']} (sandbox: {s['sandbox_backend']}, code by {s['code_author']})")
    for s in rep["sources"]["W"]:
        lines.append(f"- [{s['id']}] {s['title']} - <{s['url']}> - \"{s['quote']}\" (retrieved {s['retrieved_at']})")
    return "\n".join(lines) + "\n"


async def run_consultant_async(platform, day: int, cfg: dict, llm, out=None, parallel: bool = True) -> dict:
    if llm is None:
        raise RuntimeError("the consultant needs an LLM (skipped in --no-llm mode)")
    o = out_dir(out)
    t0 = time.perf_counter()
    brief = build_brief(platform, cfg)
    plan = await _plan(llm, brief, cfg)
    t1 = time.perf_counter()
    from analysis.real_ads import with_real_ads
    analyses, run = await fan_out(plan["questions"], plan["queries"], with_real_ads(platform.public_data()), cfg, llm,
                                  o, parallel)
    t2 = time.perf_counter()

    reg = CitationRegistry()
    d_sources = [a for a in analyses if a.get("ok")]
    w_sources = [s for s in run.get("sources", []) if s.get("verified")]
    reg.add_many(d_sources)
    reg.add_many(w_sources)

    recs = []
    if reg.ids():
        try:
            drafts = await _write(llm, brief, reg, cfg)
        except Exception as e:  # noqa: BLE001
            drafts = []
            brief["writer_error"] = str(e)[:200]
        for i, draft in enumerate(drafts, 1):
            rec = _clean(dict(draft))
            res = _check(reg, rec)
            rewritten = False
            if not res.ok:
                new = await _rewrite(llm, brief, reg, rec, res.errors, cfg)
                if new:
                    cand = _clean(dict(new))
                    cres = _check(reg, cand)
                    rewritten = True
                    rec, res = (cand, cres) if cres.ok or len(cres.errors) <= len(res.errors) else (rec, res)
            rec.update({"id": f"R{i}", "status": "verified" if res.ok else "unverified",
                        "check_errors": res.errors, "rewritten": rewritten, "decision": None})
            recs.append(rec)
    # ranked action list: verified first, then by the consultant's priority
    recs.sort(key=lambda r: (r["status"] != "verified", r["priority"]))
    for i, r in enumerate(recs, 1):
        r["rank"] = i
    t3 = time.perf_counter()

    rep = {
        "day": day,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "brief": brief,
        "plan": plan,
        "failed_questions": [{"question": a["question"], "error": a.get("error")} for a in analyses if not a.get("ok")],
        "research": {"mode": run.get("mode"), "run_id": run.get("run_id"), "created_at": run.get("created_at")},
        "sources": {"D": d_sources, "W": w_sources},
        "recommendations": recs,
        "timings": {"plan_s": round(t1 - t0, 3), "fanout_s": round(t2 - t1, 3), "write_check_s": round(t3 - t2, 3),
                    "total_s": round(t3 - t0, 3), "parallel": parallel},
    }
    cdir = o / "consult"
    cdir.mkdir(parents=True, exist_ok=True)
    write_json((cdir / f"day_{day}.json"), rep)
    write_text(cdir / f"day_{day}.md", to_markdown(rep))
    return rep


def run_consultant(platform, day: int, cfg: dict, llm, out=None, parallel: bool = True) -> dict:
    return run_sync(run_consultant_async(platform, day, cfg, llm, out, parallel))


def latest_report(out=None) -> dict | None:
    cdir = out_dir(out) / "consult"
    files = sorted(cdir.glob("day_*.json"), key=lambda p: int(p.stem.split("_")[1])) if cdir.exists() else []
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def decide_recommendations(day: int, decisions: dict, out=None) -> dict:
    """decisions: {"R1": "accept" | "reject"}. Unverified recommendations can be accepted only as owner to-dos;
    their bid actions are never applied."""
    o = out_dir(out)
    path = o / "consult" / f"day_{day}.json"
    rep = json.loads(path.read_text(encoding="utf-8"))
    pending_path, todo_path = o / "pending_consultant_actions.json", o / "todos.json"
    pending = json.loads(pending_path.read_text(encoding="utf-8")) if pending_path.exists() else []
    todos = json.loads(todo_path.read_text(encoding="utf-8")) if todo_path.exists() else []
    for r in rep["recommendations"]:
        d = decisions.get(r["id"])
        if d not in ("accept", "reject") or r.get("decision"):
            continue
        r["decision"] = d
        if d != "accept":
            continue
        if r["type"] == "bid_action" and r.get("bid_action") and r["status"] == "verified":
            pending.append({**r["bid_action"], "source": f"day {day} {r['id']}"})
        else:
            todos.append({"from": f"day {day} {r['id']}", "title": r["title"], "todo": r["recommendation"],
                          "next_action": r.get("next_action", ""), "priority": r.get("priority"),
                          "citations": r["citations"], "status": r["status"], "done": False})
    write_json(path, rep)
    write_json(pending_path, pending)
    write_json(todo_path, todos)
    return {"pending_actions": pending, "todos": todos}
