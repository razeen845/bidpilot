# Real data used by BidPilot

| File | What | Source and licence |
|---|---|---|
| `uci_textiles.csv`, `uci_textiles_daily.csv` | 40 home-textile products (doormats, hot-water-bottle covers, tea towels and oven gloves, aprons, cushions) with real median prices and real daily units sold, 2011-09-15 to 2011-10-28 | Daqing Chen, "Online Retail", UCI Machine Learning Repository (2015), https://archive.ics.uci.edu/dataset/352/online+retail, CC BY 4.0. Extracted by `scripts/build_real_data.py` |
| `KAG_conversion_data.csv` | 1,143 real Facebook ads of an anonymous company (3 campaigns): impressions, clicks, spend, enquiries, approved purchases | "Sales Conversion Optimization" dataset (Kaggle); copy from github.com/Nithilaa/KAG_conversion_data_WebAnalytics |

How it is used:
- `store.source: uci_online_retail` builds the store from the real products, prices (GBP x 360 = Rs) and daily demand
  pattern (smoothed 7-day, scaled down to a small shop). Ad clicks, costs, conversions, margins and stock are simulated.
- `python -m cli audit-ads` / the "Real ads audit" tab run BidPilot's waste rule on the real ads (read-only), and
  Ask the data can query them as the `real_ads` table.

The raw UCI file (`Online Retail.xlsx`, 23 MB) is not committed; download it from the link above to rebuild.
