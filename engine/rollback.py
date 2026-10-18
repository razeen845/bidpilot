"""Rollback rules: undo bid moves that made things significantly worse.

Only the agent's own decisions are judged: an entity is eligible only if the agent changed its bid
(bid_adjust or consultant) during the window being judged. For each eligible entity: compare the last `window_days` with the `window_days` before them. If the average bid
moved by at least `min_bid_move` and daily profit dropped by at least `min_drop_per_day` with a one-sided Welch
t-test p < `p_value`, propose reverting to the earlier (median) bid, then keep the optimizer off that entity
for `cooldown_days`. Small samples rarely reach significance, so rollbacks only fire on clear damage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def rollback_cooldown(changes: pd.DataFrame, day: int, cfg: dict) -> set:
    rb = cfg.get("rollback") or {}
    if changes is None or changes.empty or "workflow" not in changes:
        return set()
    recent = changes[(changes.workflow == "rollback") & (changes.day > day - int(rb.get("cooldown_days", 3)))]
    return set(recent.entity_id)


def rollback_proposals(stats: pd.DataFrame, bids: dict, paused: set, day: int, cfg: dict,
                       cooldown: set | None = None, changes: pd.DataFrame | None = None) -> dict:
    rb = cfg.get("rollback") or {}
    if not rb.get("enabled", True):
        return {}
    w = int(rb.get("window_days", 4))
    cooldown = cooldown or set()
    if changes is None or changes.empty or "workflow" not in changes:
        return {}
    ours = changes[changes.workflow.isin(["bid_adjust", "consultant"]) & (changes.day > day - w) & (changes.day <= day)]
    eligible = set(ours.entity_id)
    win = stats[(stats.day > day - 2 * w) & (stats.day <= day) & stats.entity_id.isin(eligible)]
    props = {}
    for eid, g in win.groupby("entity_id"):
        if eid in paused or eid in cooldown or len(g) < 2 * w or (g.bid <= 0).any():
            continue
        before, after = g[g.day <= day - w], g[g.day > day - w]
        b0, b1 = float(before.bid.mean()), float(after.bid.mean())
        if b0 <= 0 or abs(b1 / b0 - 1) < float(rb.get("min_bid_move", 0.15)):
            continue
        drop = float(before.profit.mean() - after.profit.mean())
        if drop < float(rb.get("min_drop_per_day", 300)):
            continue
        if np.var(before.profit.values) == 0 and np.var(after.profit.values) == 0:
            p = 0.0
        else:
            p = float(sps.ttest_ind(before.profit, after.profit, equal_var=False, alternative="greater").pvalue)
        if p < float(rb.get("p_value", 0.05)):
            props[eid] = ("rollback", round(float(before.bid.median()), 2), {"p_value": round(p, 4),
                                                                             "profit_drop_per_day": round(drop, 2)})
    return props
