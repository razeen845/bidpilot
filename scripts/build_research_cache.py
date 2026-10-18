"""Build the shipped research cache from hand-collected sources (PLAN Phase 7 fallback).

No search API key was available while building, so these sources were found by hand. Each page is fetched
live with the same `fetch_text` the live pipeline uses, every quote goes through the same `quote_check`,
and only verified sources are written to data/research_cache/run_<timestamp>.json (+ the page text).

    python scripts/build_research_cache.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.research import SHIPPED_CACHE, fetch_text, quote_check  # noqa: E402

Q_SEASON = "home textile online sales seasonality"
Q_CVR = "e-commerce home goods average conversion rate"
Q_STOCK = "shopping ads bidding low inventory best practice"
Q_NEG = "search ads negative keywords free cheap"

CURATED = [
    (Q_SEASON, "https://www.salehoo.com/trends/bedsheet-sets", "Bedsheet Sets Market Trends (SaleHoo)",
     "Demand for bed sheet sets is steady year-round but spikes in November with holiday shopping.",
     "Bed sheet sets sell on a mix of routine replacement and gifting, so demand is steady year-round but spikes "
     "in November during holiday shopping."),
    (Q_SEASON, "https://www.salehoo.com/trends/luxury-comforter", "Luxury Comforter Market Trends (SaleHoo)",
     "Luxury comforters have a clear demand peak in November as holiday upgrades and gifts.",
     "Luxury comforters sell like holiday upgrades and gifts, with a clear demand peak in November"),
    (Q_CVR, "https://ecdb.com/blog/average-conversion-rate-for-e-commerce-what-is-an-average-conversion-rate-by-industry/5229",
     "Average Conversion Rate for E-Commerce by Industry (ECDB)",
     "Furniture & Homeware has the lowest e-commerce conversion rate of ECDB's seven categories, at 1.94%.",
     "Conversion rates range from 1.94% in Furniture & Homeware to 4.39% in Grocery across the seven top-level "
     "categories ECDB tracks"),
    (Q_CVR, "https://clevertap.com/blog/average-e-commerce-conversion-rate/", "Average E-commerce Conversion Rate (CleverTap)",
     "Home goods e-commerce converts at about 2%, held back by high product costs and shipping logistics.",
     "Home goods e-commerce, including furniture and decor, has a low conversion rate of about 2%, largely due to "
     "high product costs and shipping logistics."),
    (Q_STOCK, "https://www.goinflow.com/blog/ppc-strategies-out-of-stock-products/", "PPC Strategies for Out-of-Stock Products (Inflow)",
     "Search ads keep running for out-of-stock products, so they should be paused until the product is back in stock.",
     "if the product in your Google Search Ads is no longer available, we seriously recommend pausing those ads "
     "until it's back in stock."),
    (Q_STOCK, "https://payability.com/blog/advertising-out-of-stock-items-tactics-you-can-use-now",
     "Advertising Out-of-Stock Items: Tactics You Can Use Now (Payability)",
     "Lowering bids and budgets while restocking keeps campaigns balanced during low inventory.",
     "It makes sense to lower the bids and budgets to balance ad campaigns while you restock the inventory."),
    (Q_STOCK, "https://support.google.com/merchants/answer/11909930?hl=en", "Pause [pause] - Google Merchant Center Help",
     "Google Merchant Center's pause attribute is meant for short pauses of 14 days or less.",
     "Only pause a product for 14 days or less."),
    (Q_NEG, "https://www.sellersprite.com/en/blog/Negative-Keywords-Guide-Reducing-Wasted-Ad-Spend",
     "Negative Keywords Guide: Reducing Wasted Ad Spend (SellerSprite)",
     "Negative keywords block low-value searches such as \"free\" and \"DIY\" to cut wasted ad spend.",
     "Prevent your ads from showing for unrelated or low-value search terms like \"free,\" \"used,\" or \"DIY.\""),
    (Q_NEG, "https://www.attnagency.com/blog/google-ads-negative-keywords-strategy",
     "Google Ads Negative Keywords Strategy (ATTN Agency)",
     "An agency estimates the average e-commerce account wastes 25-35% of ad spend on irrelevant traffic.",
     "The average e-commerce account wastes 25-35% of ad spend on irrelevant traffic."),
]


def main() -> int:
    SHIPPED_CACHE.mkdir(parents=True, exist_ok=True)
    (SHIPPED_CACHE / "pages").mkdir(exist_ok=True)
    sources, rejected, pages = [], [], []
    for query, url, title, fact, quote in CURATED:
        try:
            text = fetch_text(url)
        except Exception as e:  # noqa: BLE001
            print(f"SKIP fetch failed: {url}: {e}")
            pages.append({"query": query, "url": url, "error": str(e)[:200]})
            continue
        sha = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        (SHIPPED_CACHE / "pages" / f"{sha}.txt").write_text(text, encoding="utf-8")
        pages.append({"query": query, "url": url, "title": title, "text_sha": sha})
        ok, why = quote_check(quote, text)
        rec = {"query": query, "url": url, "title": title, "fact": fact, "quote": quote, "verified": ok,
               "check": why, "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "text_sha": sha}
        (sources if ok else rejected).append(rec)
        print(f"{'OK  ' if ok else 'FAIL'} {why:<28} {url}")
    for i, s in enumerate(sources, 1):
        s["id"] = f"W{i}"
    run = {"run_id": time.strftime("%Y%m%d_%H%M%S"), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "mode": "live", "provider": "hand-collected (no search API key)",
           "queries": sorted({c[0] for c in CURATED}), "pages": pages, "rejected": rejected, "sources": sources}
    path = SHIPPED_CACHE / f"run_{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(f"{len(sources)} verified sources -> {path}")
    return 0 if len(sources) >= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
