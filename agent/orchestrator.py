"""One agent day.

run_agent_day(platform, day, auto_approve, pending_consultant_actions):
 1. pull report + stock          2. run workflows (+ approved consultant actions, through guardrails)
 3. planner (LLM, may only hold) 4. explainers (concurrent)
 5. split auto / needs approval, save out/day_{n}_plan.json
 6. apply (auto-approve, or decisions from out/day_{n}_decisions.json)
 7. day report                   8. platform.step(); consultant on consultant.run_on_days
"""
from __future__ import annotations

import json
import time

from agent import prompts as P
from agent.explain import check_numbers, explain_changes, numeric_values
from agent.llm import run_sync
from agent.workflows import plan_day
from analysis.metrics import account_totals
from engine.models import Change
from settings import get_logger, load_config, out_dir, read_json, write_json

log = get_logger("orchestrator")


def live_cfg(cfg: dict, out=None) -> dict:
    """Live (persistent) account: add measured incrementality factors from concluded experiments."""
    from agent.experiments import factors_for_cfg
    factors = factors_for_cfg(out)
    if not factors:
        return cfg
    return {**cfg, "incrementality": {**(cfg.get("incrementality") or {}), **factors}}

MAX_PLANNER_ROWS = 40


def _summary(ctx: dict, changes: list[Change], flags: list[dict]) -> dict:
    by = {}
    for c in changes:
        by[f"{c.workflow}:{c.action}"] = by.get(f"{c.workflow}:{c.action}", 0) + 1
    rows = sorted(changes, key=lambda c: (not c.needs_approval, -abs(c.change_pct)))[:MAX_PLANNER_ROWS]
    return {
        "day": ctx["day"] + 1,
        "last_7_days": account_totals(ctx["stats"], ctx["day"], 7),
        "n_changes": len(changes),
        "n_need_approval": sum(c.needs_approval for c in changes),
        "changes_by_type": by,
        "movers": flags[:10],
        "low_stock_products": ctx["stock_m"][ctx["stock_m"].days_of_stock < ctx["cfg"]["guardrails"]["low_stock_days"]]
        .reset_index()[["product_id", "stock_units", "days_of_stock"]].to_dict(orient="records"),
        "proposals": [{"id": c.entity_id, "name": c.name, "workflow": c.workflow, "action": c.action,
                       "old_bid": c.old_bid, "new_bid": c.new_bid, "change_pct": round(c.change_pct * 100, 1),
                       "roas_7d": c.evidence.roas_7d, "clicks_7d": c.evidence.clicks_7d,
                       "conversions_7d": c.evidence.conversions_7d} for c in rows],
    }


def template_day_note(summary: dict) -> str:
    t = summary["last_7_days"]
    return (f"{summary['n_changes']} changes proposed, {summary['n_need_approval']} need your approval. "
            f"Last 7 days: spend Rs {t['spend']:,.0f}, profit Rs {t['profit']:,.0f}.")


async def prepare_day_async(platform, cfg: dict, llm=None, pending_consultant_actions=None, out=None,
                            save: bool = True, experiments: list | None = None) -> tuple[dict, list[Change]]:
    """save=True is the live account: plans are saved and approved experiments / measured incrementality apply.
    save=False is a simulation run (the race): nothing outside the run influences it."""
    t0 = time.perf_counter()
    o = out_dir(out)
    day = platform.day + 1
    if save:
        from agent.experiments import load_experiments
        cfg = live_cfg(cfg, o)
        experiments = load_experiments(o) if experiments is None else experiments
    planned = plan_day(platform, cfg, pending_consultant_actions, experiments)
    changes, flags, ctx = planned["changes"], planned["flags"], planned["ctx"]
    summary = _summary(ctx, changes, flags)

    # planner: focus + day note + optional holds (never bids)
    note, focus, held = template_day_note(summary), [], []
    if llm is not None:
        try:
            pl = await llm.chat_json("planner", P.PLANNER_SYSTEM, P.planner_user(summary), P.PlannerOut, max_tokens=600)
            focus = pl["focus"][:4]
            if not check_numbers(pl["day_note"], numeric_values(summary)) and pl["day_note"].strip():
                note = pl["day_note"].strip()
            holdable = {c.entity_id for c in changes if c.workflow != "stock_guard"}
            hold_ids = set(pl["hold"]) & holdable
            for c in changes:
                if c.entity_id in hold_ids:
                    held.append({"entity_id": c.entity_id, "name": c.name, "proposed": c.action,
                                 "old_bid": c.old_bid, "new_bid": c.new_bid,
                                 "reason": pl["hold_reasons"].get(c.entity_id, "held by planner")})
            changes = [c for c in changes if c.entity_id not in hold_ids]
        except Exception as e:  # noqa: BLE001
            focus = [f"planner unavailable: {type(e).__name__}"]

    changes = await explain_changes(llm, changes, o)
    plan = {
        "day": day,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "llm_used": llm is not None,
        "day_note": note,
        "focus": focus,
        "flags": flags,
        "held": held,
        "changes": [c.model_dump() for c in changes],
        "n_auto": sum(not c.needs_approval for c in changes),
        "n_need_approval": sum(c.needs_approval for c in changes),
        "summary": summary,
        "consultant_actions_used": pending_consultant_actions or [],
        "seconds": round(time.perf_counter() - t0, 3),
    }
    if save:
        write_json(o / f"day_{day}_plan.json", plan)
    log.info("day %s planned: %d changes (%d need approval), llm=%s, %.2fs", day, len(changes),
             plan["n_need_approval"], llm is not None, plan["seconds"])
    return plan, changes


def apply_decisions(platform, plan: dict, decisions: dict | None = None, auto_approve: bool = False) -> dict:
    """decisions: {entity_id: "approve" | "reject"}. Approval rows without a decision are not applied."""
    decisions = decisions or {}
    apply, by, rejected, pending = [], {}, [], []
    for c in plan["changes"]:
        eid = c["entity_id"]
        if not c["needs_approval"]:
            apply.append(c)
            by[eid] = "auto"
        elif auto_approve:
            apply.append(c)
            by[eid] = "auto"
        elif decisions.get(eid) == "approve":
            apply.append(c)
            by[eid] = "human"
        elif decisions.get(eid) == "reject":
            rejected.append(eid)
        else:
            pending.append(eid)
    platform.apply_changes(apply, approved_by=by)
    return {"applied": len(apply), "auto": sum(v == "auto" for v in by.values()),
            "human": sum(v == "human" for v in by.values()), "rejected": rejected, "not_decided": pending}


def template_day_report(plan: dict, applied: dict, totals: dict | None) -> str:
    s = (f"Day {plan['day']}: {applied['applied']} changes applied ({applied['auto']} automatic, "
         f"{applied['human']} approved by you), {len(applied['rejected'])} rejected.")
    if totals:
        s += f" Result: spend Rs {totals['spend']:,.0f}, revenue Rs {totals['revenue']:,.0f}, profit Rs {totals['profit']:,.0f}."
    return s


async def day_report_async(llm, plan: dict, applied: dict, totals: dict | None) -> str:
    tmpl = template_day_report(plan, applied, totals)
    if llm is None:
        return tmpl
    data = {"day": plan["day"], "applied": applied, "results": totals, "day_note": plan["day_note"],
            "held_by_planner": len(plan["held"]), "flags": plan["flags"][:5]}
    try:
        r = (await llm.chat_json("day_report", P.DAY_REPORT_SYSTEM, P.day_report_user(data), P.DayReportOut,
                                 max_tokens=300))["report"].strip()
        return r if r and not check_numbers(r, numeric_values(data)) else tmpl
    except Exception:  # noqa: BLE001
        return tmpl


def finish_day(platform, plan: dict, decisions: dict | None = None, auto_approve: bool = False, cfg: dict | None = None,
               llm=None, out=None, run_consultant: bool = True) -> dict:
    """Steps 6-8 for the dashboard flow: apply decisions, step the platform, write the day report."""
    cfg = cfg or load_config()
    o = out_dir(out)
    day = plan["day"]
    if platform.day + 1 != day:
        raise ValueError(f"plan is for day {day} but the account is at day {platform.day}")
    if decisions is None:
        decisions = read_json(o / f"day_{day}_decisions.json", default={})
    applied = apply_decisions(platform, plan, decisions, auto_approve)
    if plan.get("consultant_actions_used"):
        (o / "pending_consultant_actions.json").unlink(missing_ok=True)   # consumed (one-time instructions)
    totals = platform.step()
    from agent.experiments import update as update_experiments
    update_experiments(platform, cfg, o)
    report = run_sync(day_report_async(llm, plan, applied, totals))
    result = {"day": day, "applied": applied, "totals": totals, "report": report}
    write_json(o / f"day_{day}_report.json", result)
    log.info("day %s done: %d applied (%d by you), %d rejected, profit Rs %.2f", day, applied["applied"],
             applied["human"], len(applied["rejected"]), totals["profit"])
    if run_consultant and llm is not None and day in cfg["consultant"]["run_on_days"]:
        from agent.consultant import run_consultant as consult
        try:
            result["consultant"] = consult(platform, day, cfg, llm, out=o)["day"]
        except Exception as e:  # noqa: BLE001 - the weekly report must never break daily bidding
            log.exception("consultant failed on day %s", day)
            result["consultant_error"] = f"{type(e).__name__}: {e}"[:300]
    return result


def run_agent_day(platform, day: int | None = None, auto_approve: bool = True, pending_consultant_actions=None,
                  cfg: dict | None = None, llm=None, out=None, run_consultant: bool = True) -> dict:
    cfg = cfg or load_config()
    if day is not None and day != platform.day + 1:
        raise ValueError(f"platform is at day {platform.day}; next day is {platform.day + 1}, not {day}")
    plan, _ = run_sync(prepare_day_async(platform, cfg, llm, pending_consultant_actions, out))
    if not auto_approve:
        return {"plan": plan, "status": "waiting_for_decisions"}
    return {"plan": plan, **finish_day(platform, plan, auto_approve=True, cfg=cfg, llm=llm, out=out,
                                       run_consultant=run_consultant)}


def load_pending_actions(out=None) -> list[dict]:
    p = out_dir(out) / "pending_consultant_actions.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


class AgentStrategy:
    """Runner strategy. agent_nollm: optimizer + guardrails + templates. agent: + planner, explainers,
    consultant on run_on_days with its verified bid actions auto-accepted for the next day."""

    def __init__(self, cfg: dict, out=None, llm=None, use_llm: bool = False):
        self.cfg, self.out = cfg, out
        self.llm = llm
        if use_llm and llm is None:
            from agent.llm import make_llm
            self.llm = make_llm(cfg, out)
        self.name = "agent" if use_llm else "agent_nollm"
        self.pending: list[dict] = []
        self.consult_reports: list[dict] = []

    def act(self, platform, day: int) -> None:
        plan, _ = run_sync(prepare_day_async(platform, self.cfg, self.llm, self.pending, self.out, save=False))
        self.pending = []
        apply_decisions(platform, plan, auto_approve=True)

    def after_step(self, platform, day: int) -> None:
        if self.llm is None or day not in self.cfg["consultant"]["run_on_days"]:
            return
        from agent.consultant import run_consultant
        rep = run_consultant(platform, day, self.cfg, self.llm, out=self.out)
        self.consult_reports.append(rep)
        self.pending = [r["bid_action"] for r in rep["recommendations"]
                        if r["status"] == "verified" and r["type"] == "bid_action" and r.get("bid_action")]
