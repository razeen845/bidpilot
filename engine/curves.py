"""Learn each entity's response curve from history.

clicks(b) = C * (1 - exp(-k * b))   fitted per entity (grid start + bounded curve_fit)
cvr       = (conv + a) / (clicks + a + b), shrunk toward the category average
cpc_ratio = cost / (clicks * bid)
cvr_low   = 10th percentile of the Beta posterior: a pessimistic conversion rate used for RAISING bids, so one
            lucky sale on little data cannot justify paying more
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import beta as beta_dist

K_GRID = np.geomspace(0.005, 0.6, 80)
K_BOUNDS = (0.002, 1.0)
PRIOR_CLICKS = 60.0   # strength of the category prior on conversion rate
CAUTIOUS_QUANTILE = 0.10


def _model(b, C, k):
    return C * (1.0 - np.exp(-k * b))


def fit_clicks(bids: np.ndarray, clicks: np.ndarray) -> tuple[float, float, bool]:
    bids = np.asarray(bids, float)
    clicks = np.asarray(clicks, float)
    if len(bids) < 3 or clicks.sum() <= 0:
        return 0.0, 0.0, False
    # grid over k with closed-form least-squares C
    X = 1.0 - np.exp(-np.outer(K_GRID, bids))
    C = (X @ clicks) / np.maximum((X * X).sum(axis=1), 1e-12)
    sse = ((X * C[:, None] - clicks) ** 2).sum(axis=1)
    i = int(np.argmin(sse))
    c0, k0 = float(C[i]), float(K_GRID[i])
    c_hi = 50.0 * clicks.max() + 50.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            (c1, k1), _ = curve_fit(_model, bids, clicks, p0=[min(c0, c_hi * 0.99), k0],
                                    bounds=([0.0, K_BOUNDS[0]], [c_hi, K_BOUNDS[1]]), maxfev=2000)
        if np.isfinite(c1) and np.isfinite(k1) and c1 > 0:
            return float(c1), float(k1), True
    except (RuntimeError, ValueError):
        pass
    return c0, k0, c0 > 0


def fit_all(stats: pd.DataFrame, entities: pd.DataFrame, products: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per entity: status (ok|learning), C, k, cvr, cpc_ratio, value_per_click."""
    min_clicks = cfg["guardrails"]["min_clicks_to_act"]
    active = stats[stats.bid > 0]
    ents = entities.merge(products[["product_id", "category", "price", "margin_pct"]], on="product_id")

    tot = active.groupby("entity_id")[["clicks", "conversions"]].sum()
    tot = tot.join(ents.set_index("entity_id")[["category"]], how="left")
    cat = tot.groupby("category")[["clicks", "conversions"]].sum()
    cat_cvr = (cat.conversions / cat.clicks.where(cat.clicks > 0, 1)).to_dict()

    rows = []
    groups = dict(tuple(active.groupby("entity_id")))
    for e in ents.itertuples():
        g = groups.get(e.entity_id)
        n_clicks = float(g.clicks.sum()) if g is not None else 0.0
        row = {"entity_id": e.entity_id, "status": "learning", "C": 0.0, "k": 0.0, "cvr": 0.0, "cvr_low": 0.0,
               "cpc_ratio": 0.75, "price": e.price, "margin_pct": e.margin_pct, "hist_clicks": n_clicks}
        if g is not None and n_clicks >= min_clicks:
            C, k, ok = fit_clicks(g.bid.values, g.clicks.values)
            m = cat_cvr.get(e.category, 0.02)
            a, b = m * PRIOR_CLICKS, (1 - m) * PRIOR_CLICKS
            denom = float((g.clicks * g.bid).sum())
            row.update({
                "status": "ok" if ok else "learning",
                "C": C, "k": k,
                "cvr": (float(g.conversions.sum()) + a) / (n_clicks + a + b),
                "cvr_low": float(beta_dist.ppf(CAUTIOUS_QUANTILE, float(g.conversions.sum()) + a,
                                               max(n_clicks - float(g.conversions.sum()), 0.0) + b)),
                "cpc_ratio": float(g.cost.sum()) / denom if denom > 0 else 0.75,
            })
        row["value_per_click"] = row["cvr"] * row["price"] * row["margin_pct"]
        row["value_per_click_low"] = row["cvr_low"] * row["price"] * row["margin_pct"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("entity_id")
