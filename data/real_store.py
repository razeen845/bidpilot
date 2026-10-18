"""Real-data store: products, prices and day-by-day demand from a real online retailer.

Set `store.source: uci_online_retail` in config.yaml. What is REAL vs SIMULATED in this mode:
  real       product names, categories, prices (GBP converted at `real_data.gbp_to_rs`), and the shape of daily
             demand (trend, weekday pattern, autumn seasonality) of each product from 2011-09-15 to 2011-10-28
  scaled     units are wholesale-sized, so demand is scaled down to a small shop (same pattern, smaller numbers)
             and smoothed with a 7-day centred mean (the shop did not trade on Saturdays)
  simulated  ad clicks/costs/conversions, margins, stock levels, keywords' response to bids (no public dataset
             has these for a real account)

Source: Daqing Chen, "Online Retail", UCI Machine Learning Repository (2015), CC BY 4.0.
"""
from __future__ import annotations

import re

import pandas as pd

from settings import ROOT

REAL = ROOT / "data" / "real"
CITATION = ('Daqing Chen, "Online Retail", UCI Machine Learning Repository (2015), CC BY 4.0, '
            "https://archive.ics.uci.edu/dataset/352/online+retail")

HEADS = [("HOT WATER BOTTLE", "hot water bottle"), ("DOORMAT", "doormat"), ("TEA TOWELS", "tea towels"),
         ("TEA TOWEL", "tea towel"), ("OVEN GLOVE", "oven glove"), ("CHILDRENS APRON", "kids apron"),
         ("CHILDREN'S APRON", "kids apron"), ("APRON", "apron"), ("CUSHION COVER", "cushion cover"),
         ("CUSHION", "cushion")]
MODIFIERS = {
    "Doormats": ["novelty", "outdoor", "welcome", "coir", "funny", "printed", "front door", "non slip"],
    "Hot water bottles": ["knitted", "fleece", "cute", "novelty", "cosy", "large", "soft", "animal"],
    "Tea towels & oven gloves": ["cotton", "printed", "kitchen", "vintage", "novelty", "retro", "linen", "gift"],
    "Aprons": ["kids", "cotton", "printed", "cute", "baking", "retro", "girls", "boys"],
    "Cushions": ["felt", "decorative", "kids", "animal", "handmade", "cute", "novelty", "craft"],
}
STOP = {"AND", "THE", "OF", "IN", "I", "AM", "SO", "SET", "PACK", "2", "3", "DESIGN", "FELTCRAFT", "A", "TO"}


def _head(desc: str) -> tuple[str, str]:
    match = next(((pat, h) for pat, h in HEADS if pat in desc), None)
    if match:
        head, rest = match[1], desc.replace(match[0], " ")
    else:
        head, rest = desc.split()[-1].lower(), " ".join(desc.split()[:-1])
    words = [w for w in re.split(r"[\s/]+", rest) if w and w.upper() not in STOP and not w.isdigit()]
    return head, " ".join(words[:2]).lower() or head


def load_real_catalog(cfg: dict) -> dict:
    """{"products": [...], "dates": {day: "YYYY-MM-DD"}, "best_category": str, "citation": str}"""
    rd = cfg.get("real_data") or {}
    fx = float(rd.get("gbp_to_rs", 360))
    target = float(rd.get("target_organic_per_day", 1.2))
    meta = pd.read_csv(REAL / "uci_textiles.csv", dtype={"stock_code": str})
    daily = pd.read_csv(REAL / "uci_textiles_daily.csv", dtype={"stock_code": str})
    dates = sorted(daily.date.unique())
    hist, sim = int(cfg["store"]["history_days"]), int(cfg["store"]["sim_days"])
    if len(dates) < hist + sim:
        raise ValueError(f"real data has {len(dates)} days, config needs {hist + sim}")
    dates = dates[-(hist + sim):]
    day_of = {d: i - (hist - 1) for i, d in enumerate(dates)}          # first history day = -(hist-1), last = sim
    wide = daily.pivot(index="date", columns="stock_code", values="units").reindex(dates).fillna(0)
    smooth = wide.rolling(7, center=True, min_periods=1).mean()
    scale = target / float(smooth.mean().median())

    products = []
    n_by_cat: dict[str, int] = {}
    for r in meta.itertuples():
        head, design = _head(r.description)
        k = n_by_cat.get(r.category, 0)
        n_by_cat[r.category] = k + 1
        mods = MODIFIERS.get(r.category, ["best"])
        series = {str(day_of[d]): round(float(smooth.loc[d, r.stock_code]) * scale, 3) for d in dates}
        products.append({
            "cat": r.category, "name": r.description.title(), "head": head, "a": design,
            "b": mods[k % len(mods)], "price": float(round(r.price_gbp * fx, -1)), "units": float(r.units_window),
            "stock_code": r.stock_code, "series": series,
        })
    best_cat = meta.groupby("category").units_window.sum().idxmax()
    return {"products": products, "dates": {str(day_of[d]): d for d in dates}, "best_category": best_cat,
            "citation": CITATION, "scale": round(scale, 5), "gbp_to_rs": fx}


def available() -> bool:
    return (REAL / "uci_textiles.csv").exists() and (REAL / "uci_textiles_daily.csv").exists()


__all__ = ["load_real_catalog", "available", "CITATION"]
