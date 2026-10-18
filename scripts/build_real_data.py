"""Extract the real-data store from the UCI "Online Retail" dataset.

Source: Daqing Chen, "Online Retail", UCI Machine Learning Repository (2015), CC BY 4.0,
https://archive.ics.uci.edu/dataset/352/online+retail - every transaction of a real UK online retailer that
sells home goods, 2010-12-01 to 2011-12-09.

We keep 40 home-textile products (doormats, hot-water-bottle covers, tea towels & oven gloves, aprons,
cushions), their real median unit price and their real daily units sold in a 44-day window
(2011-09-15 .. 2011-10-28: 30 history days + 14 simulated days, the autumn run-up to Christmas).

    python scripts/build_real_data.py        # needs data/real/Online Retail.xlsx (download link above)

Writes data/real/uci_textiles.csv and data/real/uci_textiles_daily.csv (small, committed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
WINDOW_START, WINDOW_END = "2011-09-15", "2011-10-28"
PER_CATEGORY = 8
CATEGORIES = [   # (category, pattern) - first match wins
    ("Doormats", r"\bDOORMAT\b"),
    ("Hot water bottles", r"HOT WATER BOTTLE"),
    ("Tea towels & oven gloves", r"TEA TOWEL|OVEN GLOVE"),
    ("Aprons", r"\bAPRON\b"),
    ("Cushions", r"\bCUSHION\b"),
]


def main() -> int:
    raw_parquet, raw_xlsx = REAL / "online_retail_raw.parquet", REAL / "Online Retail.xlsx"
    if raw_parquet.exists():
        df = pd.read_parquet(raw_parquet)
    elif raw_xlsx.exists():
        df = pd.read_excel(raw_xlsx, dtype={"InvoiceNo": str, "StockCode": str, "Description": str})
        df.to_parquet(raw_parquet, index=False)
    else:
        print(f"missing {raw_xlsx}: download https://archive.ics.uci.edu/static/public/352/online+retail.zip")
        return 1
    df = df[(df.Quantity > 0) & (df.UnitPrice > 0) & df.Description.notna()].copy()
    df["Description"] = df.Description.str.strip()
    df["date"] = df.InvoiceDate.dt.normalize()
    df["category"] = None
    for cat, pat in CATEGORIES:
        hit = df.category.isna() & df.Description.str.contains(pat, regex=True)
        df.loc[hit, "category"] = cat
    df = df[df.category.notna()]
    win = df[(df.date >= WINDOW_START) & (df.date <= WINDOW_END)]

    meta = (df.groupby("StockCode").agg(description=("Description", lambda s: s.mode().iloc[0]),
                                        category=("category", "first"), price_gbp=("UnitPrice", "median"))
            .join(win.groupby("StockCode").agg(units_window=("Quantity", "sum"), days_sold=("date", "nunique"))))
    meta = meta[meta.days_sold >= 10].sort_values("units_window", ascending=False)
    picked = pd.concat([g.head(PER_CATEGORY) for _, g in meta.groupby("category", sort=False)])
    if len(picked) < len(CATEGORIES) * PER_CATEGORY:
        print(f"only {len(picked)} products qualified")
    picked = picked.reset_index().rename(columns={"StockCode": "stock_code"})

    days = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    daily = (win[win.StockCode.isin(picked.stock_code)].groupby(["StockCode", "date"]).Quantity.sum()
             .unstack(fill_value=0).reindex(columns=days, fill_value=0))
    long = daily.stack().rename("units").reset_index()
    long.columns = ["stock_code", "date", "units"]
    long["date"] = long.date.dt.strftime("%Y-%m-%d")

    picked[["stock_code", "description", "category", "price_gbp", "units_window", "days_sold"]].to_csv(
        REAL / "uci_textiles.csv", index=False)
    long.to_csv(REAL / "uci_textiles_daily.csv", index=False)
    print(picked.groupby("category").size().to_string())
    print(f"{len(picked)} products, {len(days)} days -> {REAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
