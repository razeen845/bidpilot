"""The daily bidding workflows and the merge that turns them into Change records.

1. daily_checkup  - flags movers (cost or conversions changed > 30% vs the previous 7 days)
2. bid_adjust     - optimizer output (click value scaled by measured incrementality)
3. waste_cleanup  - pauses from guardrails
4. stock_guard    - stock rules
+ experiment      - switchback incrementality tests the owner approved (pause on OFF days, resume on ON days)
+ rollback        - revert bid moves that significantly hurt profit
Merge priority per entity: pause > stock_guard > experiment > rollback > consultant > waste > bid_adjust.
"""
from __future__ import annotations

import pandas as pd

from analysis.metrics import entity_metrics, stock_metrics, top_spenders
from engine import guardrails as G
from engine.curves import fit_all
from engine.models import Change
from engine.optimizer import optimize
from engine.platform_rules import primary_kpi, waste_action
from engine.rollback import rollback_cooldown, rollback_proposals

MOVER_THRESHOLD = 0.30
DISCOVERY_CLICKS = 100          # fewer lifetime clicks -> still a discovery target
PRIORITY = ["stock_guard", "experiment", "rollback", "consultant", "waste", "bid_adjust"]


def build_context(platform, cfg: dict) -> dict:
    day = platform.day
    stats = platform.get_stats()
    ents = platform.get_entities()
    products = platform.get_products()
    stock_hist = platform.get_stock_history()
    metrics = entity_metrics(stats, day, 7, ents)
    g = cfg["guardrails"]
    winners = metrics[metrics.profit_7d > 0].sort_values("profit_7d", ascending=False)
    proven = set(winners.head(int(g.get("protect_top_profit", 0))).index)
    m14 = entity_metrics(stats, day, 14, ents)
    hist = stats[stats.bid > 0].groupby("entity_id")[["clicks", "conversions"]].sum().reindex(ents.entity_id).fillna(0)
    roles = {}
    for eid in ents.entity_id:
        if eid in proven:
            roles[eid] = "proven"
        elif m14.loc[eid, "conversions_14d"] == 0 and m14.loc[eid, "clicks_14d"] >= g["waste_min_clicks_7d"]:
            roles[eid] = "waste_candidate"
        elif hist.loc[eid, "clicks"] < DISCOVERY_CLICKS:
            roles[eid] = "discovery"
        else:
            roles[eid] = "standard"
    objectives = cfg.get("objectives") or {}
    growth = {e.entity_id: float(objectives[e.product_id].get("max_acos", 0.5)) for e in ents.itertuples()
              if (objectives.get(e.product_id) or {}).get("mode") == "growth"}
    return {
        "cfg": cfg,
        "day": day,
        "stats": stats,
        "entities": ents.set_index("entity_id"),
        "entities_df": ents,
        "products": products,
        "stock_hist": stock_hist,
        "bids": platform.get_bids(),
        "paused": platform.get_paused(),
        "changes": platform.get_changes(),
        "metrics": metrics,
        "metrics_prev": entity_metrics(stats, day - 7, 7, ents),
        "stock_m": stock_metrics(stock_hist, day, 7),
        "product_of": dict(zip(ents.entity_id, ents.product_id)),
        "top": set(top_spenders(metrics, cfg["guardrails"]["approval_top_spenders"])),
        "proven": proven,
        "roles": roles,
        "hist": hist,
        "growth": growth,
    }


def daily_checkup(ctx: dict) -> list[dict]:
    """Flags only: entities whose 7-day cost or conversions moved > 30% vs the 7 days before."""
    cur, prev, ents = ctx["metrics"], ctx["metrics_prev"], ctx["entities"]
    flags = []
    for eid in cur.index:
        for metric, col, floor in (("cost", "spend_7d", 300.0), ("conversions", "conversions_7d", 3.0)):
            p, c = float(prev.loc[eid, col]), float(cur.loc[eid, col])
            if p >= floor and abs(c / p - 1) > MOVER_THRESHOLD:
                flags.append({"entity_id": eid, "name": ents.loc[eid, "name"], "metric": metric,
                              "prev_7d": round(p, 2), "last_7d": round(c, 2), "change_pct": round(c / p - 1, 4)})
    return flags


def stock_guard(ctx: dict) -> tuple[dict, set, set]:
    return G.stock_guard_proposals(ctx["entities_df"], ctx["bids"], ctx["paused"], ctx["stock_m"],
                                   ctx["changes"], ctx["cfg"])


def waste_cleanup(ctx: dict) -> dict:
    return G.waste_proposals(ctx["stats"], ctx["entities_df"], ctx["products"], ctx["paused"], ctx["stock_hist"],
                             ctx["day"], ctx["cfg"])


def bid_adjust(ctx: dict, favour_high: set) -> dict:
    curves = fit_all(ctx["stats"], ctx["entities_df"], ctx["products"], ctx["cfg"])
    factors = ctx["cfg"].get("incrementality") or {}
    if factors:   # a click is only worth the sales it actually adds
        f = curves.index.map(lambda eid: float(factors.get(ctx["product_of"][eid], 1.0)))
        curves["value_per_click"] = curves["value_per_click"] * f.to_numpy()
        curves["value_per_click_low"] = curves["value_per_click_low"] * f.to_numpy()
    opt = optimize(curves, ctx["bids"], ctx["cfg"], paused=ctx["paused"], favour_high=favour_high,
                   growth=ctx["growth"])
    ctx["curves"], ctx["optimizer"] = curves, opt
    props = {}
    for eid, r in opt.iterrows():
        if r.status == "ok" and abs(r.new_bid - r.old_bid) >= 0.01:
            props[eid] = ("raise" if r.new_bid > r.old_bid else "lower", float(r.new_bid))
    return props


def _match(action: dict, eid: str, ents: pd.DataFrame, cats: dict) -> bool:
    f = action.get("filter") or {}
    e = ents.loc[eid]
    if f.get("entity_type") and e.entity_type != f["entity_type"]:
        return False
    if f.get("category") and cats.get(e.product_id, "").lower() != str(f["category"]).lower():
        return False
    kw = f.get("keyword_contains")
    if kw:
        words = [kw] if isinstance(kw, str) else list(kw)
        if e.entity_type != "keyword" or not any(w.lower() in e["name"].lower() for w in words):
            return False
    if f.get("product_ids") and e.product_id not in set(f["product_ids"]):
        return False
    return bool(f)


def consultant_actions(ctx: dict, pending: list[dict] | None) -> dict:
    """Bounded consultant instructions (lower/raise by up to max_pct, or pause) -> proposals."""
    props = {}
    if not pending:
        return props
    ents = ctx["entities"]
    cats = dict(zip(ctx["products"].product_id, ctx["products"].category))
    cap = ctx["cfg"]["bids"]["max_daily_change"]
    for a in pending:
        act = a.get("action")
        pct = min(abs(float(a.get("max_pct") or cap)), cap)
        for eid in ents.index:
            if eid in ctx["paused"] or not _match(a, eid, ents, cats):
                continue
            old = ctx["bids"][eid]
            if act == "pause":
                props[eid] = ("pause", old)
            elif act == "lower":
                props[eid] = ("lower", old * (1 - pct))
            elif act == "raise":
                props[eid] = ("raise", old * (1 + pct))
    return props


def experiments_wf(ctx: dict, experiments: list[dict] | None) -> dict:
    from agent.experiments import experiment_proposals
    if not experiments:
        return {}
    last = G.last_actions(ctx["changes"])
    props = experiment_proposals(experiments, ctx["day"] + 1, ctx["entities_df"], ctx["paused"], last)
    out = {}
    for eid, (action, _) in props.items():
        pid = ctx["product_of"][eid]
        if action == "resume" and pid in ctx["stock_m"].index and ctx["stock_m"].loc[pid, "stock_units"] <= 0:
            continue      # do not switch ads back on for a product with no stock
        out[eid] = (action, ctx["bids"][eid])
    return out


def rollback_wf(ctx: dict) -> tuple[dict, set]:
    cooldown = rollback_cooldown(ctx["changes"], ctx["day"] + 1, ctx["cfg"])
    props = rollback_proposals(ctx["stats"], ctx["bids"], ctx["paused"], ctx["day"], ctx["cfg"], cooldown,
                               ctx["changes"])
    ctx["rollback_meta"] = {eid: meta for eid, (_, _, meta) in props.items()}
    out = {eid: ("raise" if new > ctx["bids"][eid] else "lower", new) for eid, (_, new, _) in props.items()}
    return out, cooldown


def merge(ctx: dict, by_workflow: dict[str, dict], hold: set) -> list[Change]:
    cfg, bids = ctx["cfg"], ctx["bids"]
    changes = []
    all_ids = sorted(set().union(*[set(p) for p in by_workflow.values()]))
    for eid in all_ids:
        cands = [(wf, by_workflow[wf][eid]) for wf in PRIORITY if eid in by_workflow.get(wf, {})]
        pauses = [c for c in cands if c[1][0] == "pause"]
        wf, (action, new) = pauses[0] if pauses else cands[0]
        if eid in hold and wf in ("consultant", "bid_adjust") and action != "pause":
            continue
        old = bids[eid]
        if action in ("pause", "resume"):
            new = old
        elif action in ("raise", "lower"):
            new = G.limit_bid(old, new, cfg, daily_limit=(wf not in ("stock_guard", "rollback")))
            if wf in ("bid_adjust", "consultant") and eid in ctx["proven"] and new < old:
                new = G.protect_proven(old, new, cfg)
            if abs(new - old) < 0.01:
                continue
            action = "raise" if new > old else "lower"
        ev = G.evidence_for(eid, ctx["metrics"], ctx["stock_m"], ctx["product_of"], ctx["metrics_prev"])
        ch = G.make_change(eid, wf, action, old, new, ctx["entities"], ev)
        ch.role = ctx["roles"].get(eid, "standard")
        ch.confidence = confidence(ctx, eid, wf)
        ch.objective = "growth" if eid in ctx["growth"] else "profit"
        ch.kpi = primary_kpi(cfg)
        if wf == "waste" and action == "pause":
            ch.platform_action = waste_action(cfg, ch.entity_type)
        ch = G.flag_approval(ch, ctx["top"], cfg, ctx["proven"])
        if wf == "experiment":    # the owner approved the whole experiment up front
            ch.needs_approval, ch.approval_reason = False, None
            ch.platform_action = "switchback test: ads OFF today" if action == "pause" else "switchback test: ads ON"
        if wf == "rollback":
            meta = ctx.get("rollback_meta", {}).get(eid, {})
            ch.platform_action = (f"revert to earlier bid (profit fell Rs {meta.get('profit_drop_per_day', 0):,.0f}/day, "
                                  f"p={meta.get('p_value')})")
        changes.append(ch)
    return changes


def confidence(ctx: dict, eid: str, workflow: str) -> str:
    """Rule-based actions (stock, waste) rest on hard facts; optimizer moves on how much data we have."""
    if workflow in ("stock_guard", "waste", "experiment", "rollback"):
        return "high"
    clicks, conv = ctx["hist"].loc[eid, "clicks"], ctx["hist"].loc[eid, "conversions"]
    if clicks >= 300 and conv >= 10:
        return "high"
    return "medium" if clicks >= 100 else "low"


def plan_day(platform, cfg: dict, pending_consultant_actions: list[dict] | None = None,
             experiments: list[dict] | None = None) -> dict:
    """Run all workflows for the next day. Returns {"changes", "flags", "ctx"}."""
    ctx = build_context(platform, cfg)
    flags = daily_checkup(ctx)
    sg, hold, favour = stock_guard(ctx)
    rb, cooldown = rollback_wf(ctx)
    hold = hold | cooldown
    by_wf = {
        "stock_guard": sg,
        "experiment": experiments_wf(ctx, experiments),
        "rollback": rb,
        "consultant": consultant_actions(ctx, pending_consultant_actions),
        "waste": waste_cleanup(ctx),
        "bid_adjust": bid_adjust(ctx, favour),
    }
    return {"changes": merge(ctx, by_wf, hold), "flags": flags, "ctx": ctx}
