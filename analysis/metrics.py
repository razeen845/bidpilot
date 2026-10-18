"""Rolling per-entity and per-product metrics computed from platform data."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_OF_STOCK_CAP = 999.0


def entity_metrics(stats: pd.DataFrame, day: int, lookback: int = 7, entities: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rolling sums over (day - lookback, day], one row per entity."""
    w = stats[(stats.day > day - lookback) & (stats.day <= day)]
    g = w.groupby("entity_id").agg(spend=("cost", "sum"), clicks=("clicks", "sum"),
                                   conversions=("conversions", "sum"), revenue=("revenue", "sum"),
                                   profit=("profit", "sum"))
    if entities is not None:
        g = g.reindex(entities.entity_id).fillna(0.0)
    g.columns = [f"{c}_{lookback}d" for c in g.columns]
    s, c = g[f"spend_{lookback}d"], g[f"clicks_{lookback}d"]
    g[f"roas_{lookback}d"] = np.where(s > 0, g[f"revenue_{lookback}d"] / s.where(s > 0, 1), 0.0)
    g[f"cpc_{lookback}d"] = np.where(c > 0, s / c.where(c > 0, 1), 0.0)
    rev, conv = g[f"revenue_{lookback}d"], g[f"conversions_{lookback}d"]
    g = g.round(2)
    # ACOS = spend / attributed sales (0 when there were no sales); fractions kept at 4 decimals
    g[f"acos_{lookback}d"] = (s / rev.where(rev > 0, np.nan)).fillna(0.0).round(4)
    g[f"cvr_{lookback}d"] = (conv / c.where(c > 0, np.nan)).fillna(0.0).round(4)
    g.index.name = "entity_id"
    return g


def stock_metrics(stock_df: pd.DataFrame, day: int, lookback: int = 7) -> pd.DataFrame:
    """Per product: current stock, average daily units sold, days of stock left, restock flag."""
    w = stock_df[(stock_df.day > day - lookback) & (stock_df.day <= day)]
    cur = stock_df[stock_df.day == day].set_index("product_id")
    avg = w.groupby("product_id").units_sold.mean()
    out = pd.DataFrame({"stock_units": cur.stock_units, "avg_daily_sales": avg.reindex(cur.index).fillna(0.0)})
    dos = np.where(out.avg_daily_sales > 0, out.stock_units / out.avg_daily_sales.where(out.avg_daily_sales > 0, 1),
                   np.where(out.stock_units > 0, DAYS_OF_STOCK_CAP, 0.0))
    out["days_of_stock"] = np.minimum(dos, DAYS_OF_STOCK_CAP)
    out["restocked_today"] = cur.get("restocked_units", pd.Series(0, index=cur.index)).fillna(0) > 0
    out.index.name = "product_id"
    return out.round(2)


def top_spenders(metrics: pd.DataFrame, n: int, lookback: int = 7) -> list[str]:
    col = f"spend_{lookback}d"
    return metrics[metrics[col] > 0].sort_values(col, ascending=False).head(n).index.tolist()


def account_totals(stats: pd.DataFrame, day: int, lookback: int = 7) -> dict:
    w = stats[(stats.day > day - lookback) & (stats.day <= day)]
    spend = float(w.cost.sum())
    return {
        "days": lookback,
        "spend": round(spend, 2),
        "revenue": round(float(w.revenue.sum()), 2),
        "profit": round(float(w.profit.sum()), 2),
        "clicks": int(w.clicks.sum()),
        "conversions": int(w.conversions.sum()),
        "roas": round(float(w.revenue.sum()) / spend, 2) if spend else 0.0,
        "acos": round(spend / float(w.revenue.sum()), 4) if w.revenue.sum() else 0.0,
        "cvr": round(float(w.conversions.sum()) / float(w.clicks.sum()), 4) if w.clicks.sum() else 0.0,
    }
