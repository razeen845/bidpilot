"""Real ad-account audit (read-only).

Loads a real ad performance export and applies BidPilot's waste rule to it, so the logic can be tested on real
numbers, not only on the simulator. Nothing is changed in any ad account: the output is a list of
recommendations.

Default dataset: "Sales Conversion Optimization" - 1,143 real Facebook ads of an anonymous company
(3 campaigns) with impressions, clicks, spend, enquiries (Total_Conversion) and purchases
(Approved_Conversion). Kaggle, uploaded by Chirag Chauhan / Madis Lemsalu; copy in data/real/KAG_conversion_data.csv.
Currency is as in the source (not stated; amounts are small, typical of USD).

Any other export works if it has an id, spend, clicks and conversions column (see COLUMN_MAP).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from settings import ROOT

DEFAULT_FILE = ROOT / "data" / "real" / "KAG_conversion_data.csv"
SOURCE = ('"Sales Conversion Optimization" (Kaggle): 1,143 real Facebook ads from an anonymous company, '
          "3 campaigns; impressions, clicks, spend, enquiries and approved purchases per ad")
COLUMN_MAP = {"ad_id": "ad_id", "xyz_campaign_id": "campaign_id", "age": "age", "gender": "gender",
              "interest": "interest", "Impressions": "impressions", "Clicks": "clicks", "Spent": "spend",
              "Total_Conversion": "enquiries", "Approved_Conversion": "purchases"}


def available(path: str | Path | None = None) -> bool:
    return Path(path or DEFAULT_FILE).exists()


def load_real_ads(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DEFAULT_FILE).rename(columns=COLUMN_MAP)
    missing = {"ad_id", "spend", "clicks", "purchases"} - set(df.columns)
    if missing:
        raise ValueError(f"ad export is missing columns: {sorted(missing)}")
    df["ad_id"] = df.ad_id.astype(str)
    df["spend"] = df.spend.astype(float).round(2)
    if "campaign_id" in df:
        df["campaign_id"] = df.campaign_id.astype(str)
    return df


def with_real_ads(data: dict) -> dict:
    """Public store data plus the real ad export (as table `real_ads`) when it is available."""
    if available():
        return {**data, "real_ads": load_real_ads()}
    return data


def audit(df: pd.DataFrame, min_spend: float = 20.0, min_clicks: int = 20, confidence: float = 0.95) -> dict:
    """Waste rule on real ads: spend and clicks above thresholds, zero purchases, and zero purchases is
    statistically unlikely at the campaign's own purchase rate (P(0) < 1 - confidence)."""
    group = "campaign_id" if "campaign_id" in df else None
    ref = (df.groupby(group)[["purchases", "clicks"]].sum() if group else
           pd.DataFrame({"purchases": [df.purchases.sum()], "clicks": [df.clicks.sum()]}, index=["all"]))
    ref_cvr = (ref.purchases / ref.clicks.where(ref.clicks > 0)).fillna(0.0)
    d = df.copy()
    d["ref_cvr"] = d[group].map(ref_cvr) if group else float(ref_cvr.iloc[0])
    d["p_zero"] = np.power(1.0 - d.ref_cvr.clip(0, 1), d.clicks.clip(lower=0))
    d["cost_per_purchase"] = (d.spend / d.purchases.where(d.purchases > 0)).round(2)
    waste = d[(d.spend >= min_spend) & (d.clicks >= min_clicks) & (d.purchases == 0) & (d.p_zero < 1 - confidence)]
    low_data = d[(d.spend >= min_spend) & (d.purchases == 0) & ~d.index.isin(waste.index)]
    by_campaign = (d.groupby(group or (lambda _: "all")).agg(ads=("ad_id", "count"), spend=("spend", "sum"),
                                                             clicks=("clicks", "sum"), purchases=("purchases", "sum"))
                   .assign(cost_per_purchase=lambda x: (x.spend / x.purchases.where(x.purchases > 0)).round(2))
                   .round(2).reset_index())
    total_spend = float(d.spend.sum())
    return {
        "source": SOURCE,
        "ads": int(len(d)),
        "total_spend": round(total_spend, 2),
        "total_purchases": int(d.purchases.sum()),
        "waste_ads": waste.sort_values("spend", ascending=False),
        "waste_spend": round(float(waste.spend.sum()), 2),
        "waste_share_pct": round(100 * float(waste.spend.sum()) / total_spend, 2) if total_spend else 0.0,
        "low_data_ads": int(len(low_data)),
        "low_data_spend": round(float(low_data.spend.sum()), 2),
        "by_campaign": by_campaign,
        "rule": f"spend >= {min_spend}, clicks >= {min_clicks}, 0 purchases, and P(0 purchases) < "
                f"{1 - confidence:.0%} at the campaign's purchase rate",
    }
