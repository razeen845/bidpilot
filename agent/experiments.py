"""Incrementality: switchback holdout tests.

Ad platforms attribute every ad sale to the ad, but some of those sales would have happened organically.
A switchback test turns a product's ads ON and OFF on alternating days (randomized within 2-day blocks)
and compares TOTAL units sold (ads + organic) on ON vs OFF days:

    incrementality = (units/day ON - units/day OFF) / attributed ad units/day ON

with a 90% Welch confidence interval. If the CI excludes 1.0 (ads significantly cannibalize organic sales)
the product gets an incrementality factor equal to the CI's upper bound - deliberately conservative, so a noisy
test never over-cuts bids. A narrow CI around 1.0 confirms the ads are incremental (factor 1.0).
The optimizer then values a click at cvr x price x margin x incrementality.
Experiments pause ads, so they need the owner's approval to start.

State lives in out/experiments.json (list) and out/incrementality.json ({product_id: factor record}).
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

from settings import out_dir, read_json, write_json

EXPERIMENTS_FILE = "experiments.json"
FACTORS_FILE = "incrementality.json"


def _cfg(cfg: dict) -> dict:
    e = cfg.get("experiments") or {}
    return {"days": int(e.get("days", 14)), "max_ci_width": float(e.get("max_ci_width", 0.6)),
            "min_days_each": int(e.get("min_days_each", 4)), "factor_floor": float(e.get("factor_floor", 0.1))}


def design_switchback(product_id: str, start_day: int, days: int, seed: int) -> dict[int, str]:
    """Blocked randomization: every 2-day block has one ON and one OFF day, order random per block."""
    rng = np.random.default_rng([seed, 7, int("".join(c for c in product_id if c.isdigit()) or 0), start_day])
    sched: dict[int, str] = {}
    for b in range(0, days, 2):
        order = ["on", "off"] if rng.random() < 0.5 else ["off", "on"]
        for i, state in enumerate(order):
            if b + i < days:
                sched[start_day + b + i] = state
    return sched


def load_experiments(out=None) -> list[dict]:
    return read_json(out_dir(out) / EXPERIMENTS_FILE, default=[])


def save_experiments(exps: list[dict], out=None) -> None:
    write_json(out_dir(out) / EXPERIMENTS_FILE, exps)


def load_factors(out=None) -> dict:
    return read_json(out_dir(out) / FACTORS_FILE, default={})


def propose(product_id: str, cfg: dict, out=None, reason: str = "") -> dict:
    exps = load_experiments(out)
    if any(e["product_id"] == product_id and e["status"] in ("proposed", "running") for e in exps):
        raise ValueError(f"an experiment for {product_id} is already proposed or running")
    exp = {"id": f"X{len(exps) + 1}", "product_id": product_id, "status": "proposed", "reason": reason,
           "days": _cfg(cfg)["days"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "schedule": {},
           "start_day": None, "result": None}
    exps.append(exp)
    save_experiments(exps, out)
    return exp


def approve(exp_id: str, next_day: int, cfg: dict, out=None) -> dict:
    """Owner approval: schedule the switchback starting on the next simulated day."""
    exps = load_experiments(out)
    exp = next(e for e in exps if e["id"] == exp_id)
    if exp["status"] != "proposed":
        raise ValueError(f"{exp_id} is {exp['status']}, not proposed")
    sched = design_switchback(exp["product_id"], next_day, exp["days"], int(cfg["seed"]))
    exp.update({"status": "running", "start_day": next_day, "schedule": {str(k): v for k, v in sched.items()},
                "approved_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    save_experiments(exps, out)
    return exp


def cancel(exp_id: str, out=None) -> None:
    exps = load_experiments(out)
    for e in exps:
        if e["id"] == exp_id and e["status"] in ("proposed", "running"):
            e["status"] = "cancelled"
    save_experiments(exps, out)


def experiment_proposals(experiments: list[dict], day: int, entities: pd.DataFrame, paused: set,
                         last: dict) -> dict:
    """Pause/resume proposals for `day` (the day about to be simulated) from running experiments."""
    props = {}
    for e in experiments:
        if e["status"] != "running":
            continue
        state = e["schedule"].get(str(day))
        ids = entities[entities.product_id == e["product_id"]].entity_id
        for eid in ids:
            if state == "off" and eid not in paused:
                props[eid] = ("pause", None)
            elif (state == "on" or state is None) and eid in paused and last.get(eid) == ("experiment", "pause"):
                props[eid] = ("resume", None)   # also resumes after the experiment ends
    return props


def analyze(exp: dict, stats: pd.DataFrame, stock: pd.DataFrame, entities: pd.DataFrame, cfg: dict) -> dict:
    c = _cfg(cfg)
    pid = exp["product_id"]
    sched = {int(k): v for k, v in exp["schedule"].items()}
    last_day = int(stats.day.max())
    ids = set(entities[entities.product_id == pid].entity_id)
    attr = stats[stats.entity_id.isin(ids)].groupby("day").conversions.sum()
    sk = stock[stock.product_id == pid].set_index("day")
    rows = []
    for d, state in sorted(sched.items()):
        if d > last_day or d not in sk.index:
            continue
        start_stock = int(sk.loc[d, "stock_units"] + sk.loc[d, "units_sold"])
        if start_stock <= 0:
            continue                      # no sales possible: tells us nothing about ads
        rows.append({"day": d, "state": state, "units_sold": int(sk.loc[d, "units_sold"]),
                     "attributed_units": int(attr.get(d, 0))})
    table = pd.DataFrame(rows, columns=["day", "state", "units_sold", "attributed_units"])
    on, off = table[table.state == "on"], table[table.state == "off"]
    res = {"product_id": pid, "days_on": int(len(on)), "days_off": int(len(off)), "table": table.to_dict("records")}
    if len(on) < c["min_days_each"] or len(off) < c["min_days_each"]:
        return {**res, "status": "insufficient_data", "conclusive": False}
    m_on, m_off = float(on.units_sold.mean()), float(off.units_sold.mean())
    a_on = float(on.attributed_units.mean())
    if a_on <= 0:
        return {**res, "status": "no_ad_sales", "conclusive": False}
    v_on, v_off = float(on.units_sold.var(ddof=1)), float(off.units_sold.var(ddof=1))
    se_diff = math.sqrt(v_on / len(on) + v_off / len(off))
    if se_diff == 0:
        dof = float(len(on) + len(off) - 2)
    else:
        dof = se_diff ** 4 / ((v_on / len(on)) ** 2 / (len(on) - 1) + (v_off / len(off)) ** 2 / (len(off) - 1) or 1e-12)
    tcrit = float(sps.t.ppf(0.95, max(dof, 1.0)))
    est = (m_on - m_off) / a_on
    half = tcrit * se_diff / a_on
    ci = [round(est - half, 3), round(est + half, 3)]
    cannibalizes = ci[1] < 1.0                       # significantly below fully incremental
    conclusive = cannibalizes or (ci[1] - ci[0]) <= c["max_ci_width"]
    # conservative factor: the CI upper bound (the most generous value the data supports), never above 1
    factor = round(min(1.0, max(c["factor_floor"], ci[1])), 3) if conclusive else None
    verdict = ("ads partly cannibalize organic sales" if cannibalizes
               else "ads look incremental" if conclusive else "inconclusive: run longer")
    return {**res, "status": "ok", "units_per_day_on": round(m_on, 3), "units_per_day_off": round(m_off, 3),
            "attributed_per_day_on": round(a_on, 3), "incrementality": round(est, 3), "ci90": ci,
            "conclusive": bool(conclusive), "verdict": verdict, "factor": factor}


def update(platform, cfg: dict, out=None) -> list[dict]:
    """Analyze running experiments; finish those whose schedule is over and store conclusive factors."""
    exps = load_experiments(out)
    data = platform.public_data()
    factors = load_factors(out)
    changed = False
    for e in exps:
        if e["status"] != "running":
            continue
        e["result"] = analyze(e, data["stats"], data["stock"], data["entities"], cfg)
        end = max(int(k) for k in e["schedule"])
        if platform.day >= end:
            e["status"] = "done"
            e["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            if e["result"].get("conclusive"):
                factors[e["product_id"]] = {"factor": e["result"]["factor"], "ci90": e["result"]["ci90"],
                                            "experiment": e["id"], "measured_through_day": platform.day}
        changed = True
    if changed:
        save_experiments(exps, out)
        write_json(out_dir(out) / FACTORS_FILE, factors)
    return exps


def factors_for_cfg(out=None) -> dict:
    """{product_id: factor} for the optimizer."""
    return {pid: float(v["factor"]) for pid, v in load_factors(out).items()}
