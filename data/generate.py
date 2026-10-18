"""Generate the fake home-textiles store, its hidden market and 30 days of history.

Deliberate stories planted for the agent and the consultant to find:
  * ~15% junk keywords (near-zero conversion), most share "free" / "cheap" / "diy";
  * low-margin best-sellers that eat a big share of the budget;
  * products that run out of stock during the 14-day sim (some restock, some don't);
  * overstocked products;
  * under-bid winners and over-bid losers;
  * one category (Bedding) that converts clearly better than the rest;
  * three products with weak listings: plenty of clicks, conversion far below their category;
  * low incrementality on the low-margin best-sellers: most of their ad sales would have happened organically.

products.csv: margin_pct is the CONTRIBUTION margin (after fees and returns); gross_margin_pct,
fee_pct and return_rate show the breakdown.

Usage: python -m data.generate [--out out]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from settings import load_config, out_dir
from sim.market import optimal_bid, simulate_day

# (name, head term, attribute A, attribute B)
CATALOG = {
    "Bedding": [
        ("Cotton King Bedsheet Set", "bedsheet", "cotton king", "king size"),
        ("Percale Queen Bedsheet", "bedsheet", "percale", "queen size"),
        ("Microfiber Duvet Cover", "duvet cover", "microfiber", "double"),
        ("Quilted Comforter", "comforter", "quilted", "winter"),
        ("Satin Pillow Covers", "pillow cover", "satin", "silk look"),
        ("Fitted Single Bedsheet", "fitted sheet", "single bed", "elastic"),
        ("Fleece Winter Blanket", "blanket", "fleece", "warm"),
        ("Jacquard Bedspread", "bedspread", "jacquard", "luxury"),
    ],
    "Curtains": [
        ("Blackout Curtains 7ft", "curtains", "blackout", "7ft"),
        ("Sheer Voile Curtains", "curtains", "sheer voile", "white"),
        ("Linen Look Curtains", "curtains", "linen look", "beige"),
        ("Velvet Eyelet Curtains", "curtains", "velvet", "eyelet"),
        ("Kids Print Curtains", "curtains", "kids room", "cartoon print"),
        ("Kitchen Cafe Curtains", "curtains", "kitchen", "short cafe"),
        ("Thermal Door Curtain", "door curtain", "thermal", "insulated"),
        ("Embroidered Curtain Pair", "curtains", "embroidered", "pair"),
    ],
    "Towels": [
        ("Egyptian Cotton Bath Towel", "bath towel", "egyptian cotton", "large"),
        ("Hand Towel Set of 4", "hand towel", "set of 4", "soft"),
        ("Quick-Dry Gym Towel", "gym towel", "quick dry", "microfiber"),
        ("Jumbo Bath Sheet", "bath sheet", "jumbo", "oversized"),
        ("Kids Hooded Towel", "hooded towel", "kids", "baby"),
        ("Bamboo Face Towels", "face towel", "bamboo", "organic"),
        ("Striped Beach Towel", "beach towel", "striped", "summer"),
        ("Hotel Bath Robe", "bath robe", "hotel", "terry"),
    ],
    "Cushions": [
        ("Velvet Cushion Covers", "cushion cover", "velvet", "16x16"),
        ("Round Floor Cushion", "floor cushion", "round", "seating"),
        ("Embroidered Cushion Set", "cushion set", "embroidered", "5 piece"),
        ("Outdoor Seat Pad", "seat pad", "outdoor", "garden"),
        ("Memory Foam Pillow", "pillow", "memory foam", "orthopedic"),
        ("Travel Neck Pillow", "neck pillow", "travel", "u shape"),
        ("Jute Cushion Covers", "cushion cover", "jute", "boho"),
        ("Kids Animal Cushion", "cushion", "animal", "kids"),
    ],
    "Rugs": [
        ("Shaggy Area Rug 5x7", "rug", "shaggy", "5x7"),
        ("Cotton Dhurrie Runner", "runner", "dhurrie", "cotton"),
        ("Anti-Slip Bath Mat", "bath mat", "anti slip", "absorbent"),
        ("Coir Doormat", "doormat", "coir", "welcome"),
        ("Kilim Rug 4x6", "rug", "kilim", "4x6"),
        ("Hallway Runner 2x8", "runner", "hallway", "2x8"),
        ("Kids Play Mat", "play mat", "kids", "foam"),
        ("Jute Round Rug", "rug", "jute round", "natural"),
    ],
}
BEST_CATEGORY = "Bedding"
PRICE_RANGE = {"Bedding": (1500, 6000), "Curtains": (1200, 5000), "Towels": (800, 3000),
               "Cushions": (800, 3500), "Rugs": (1500, 6000)}
JUNK_PATTERNS = ["free", "cheap", "diy"]
JUNK_OTHER = ["{h} meaning", "{h} drawing", "how to wash {h}", "{h} wallpaper"]
WEAK_LISTING_CVR = 0.45   # conversion multiplier for the planted weak-listing products
RETURN_COST_SHARE = 0.5   # a return costs ~half the price (shipping both ways, damaged stock)
SUFFIXES = ["online", "price", "sale", "buy", "best", "shop"]


def _build(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    seed = cfg["seed"]
    rng = np.random.default_rng([seed, 1])
    n_products = cfg["store"]["n_products"]
    bids_cfg = cfg["bids"]

    # ---------- products ----------
    real = None
    if cfg["store"].get("source", "synthetic") == "uci_online_retail":
        from data.real_store import load_real_catalog
        real = load_real_catalog(cfg)
        prods = [(r["cat"], r["name"], r["head"], r["a"], r["b"]) for r in real["products"]][:n_products]
    else:
        prods = []
        for cat, items in CATALOG.items():
            for item in items:
                prods.append((cat, *item))
        prods = prods[:n_products]
    n = len(prods)
    pids = [f"P{i + 1:02d}" for i in range(n)]
    best_category = real["best_category"] if real else BEST_CATEGORY

    order = rng.permutation(n)
    if real:   # the real top sellers are the best-sellers; the other stories are assigned at random
        top = sorted(range(n), key=lambda i: -real["products"][i]["units"])[:3]
        order = list(top) + [i for i in order if i not in top]
    low_margin = set(order[:10])
    best_sellers = set(order[:3])              # low margin AND high volume
    stockout = list(order[10:15])               # near stockout during the sim
    overstock = set(order[15:20])
    restockers = set(stockout[:3])

    products, prod_story = [], {}
    for i, (cat, name, head, a, b) in enumerate(prods):
        if real:
            price = real["products"][i]["price"]
        else:
            lo, hi = PRICE_RANGE[cat]
            price = float(round(rng.uniform(lo, hi), -1))
        margin = rng.uniform(0.08, 0.15) if i in low_margin else rng.uniform(0.40, 0.60)
        story = ("best_seller_low_margin" if i in best_sellers else "low_margin" if i in low_margin
                 else "stockout" if i in stockout else "overstock" if i in overstock else "normal")
        prod_story[pids[i]] = story
        products.append({"product_id": pids[i], "name": name, "category": cat, "price": price,
                         "margin_pct": round(float(margin), 3), "head": head, "a": a, "b": b, "idx": i})

    # ---------- entities: 1 product ad + 3 keywords per product ----------
    n_kw = cfg["store"]["n_keywords"]
    kw_per_product = n_kw // n
    junk_products = set(rng.choice([i for i in range(n) if i not in best_sellers], size=int(round(0.15 * n_kw)), replace=False))
    junk_list = sorted(junk_products)
    entities, used = [], set()
    kcount = 0
    for p in products:
        i = p["idx"]
        entities.append({"entity_id": p["product_id"], "entity_type": "product", "name": p["name"],
                         "product_id": p["product_id"], "junk": False})
        names = [f"{p['a']} {p['head']}", f"{p['b']} {p['head']}", f"{p['head']} {SUFFIXES[i % len(SUFFIXES)]}"]
        junk_name = None
        if i in junk_products:
            j = junk_list.index(i)
            if j < len(junk_list) - len(JUNK_OTHER):
                junk_name = f"{JUNK_PATTERNS[j % 3]} {p['head']}"
            else:
                junk_name = JUNK_OTHER[j - (len(junk_list) - len(JUNK_OTHER))].format(h=p["head"])
            names[2] = junk_name
        for kname in names[:kw_per_product]:
            base, extra = kname, 0
            while kname in used:
                extra += 1
                kname = f"{base} {SUFFIXES[(i + extra) % len(SUFFIXES)]}"
            used.add(kname)
            kcount += 1
            entities.append({"entity_id": f"K{kcount:03d}", "entity_type": "keyword", "name": kname,
                             "product_id": p["product_id"], "junk": kname == junk_name})

    # ---------- hidden market params ----------
    by_pid = {p["product_id"]: p for p in products}
    params = {}
    all_ids = [e["entity_id"] for e in entities]
    eligible_win = [e["entity_id"] for e in entities
                    if not e["junk"] and prod_story[e["product_id"]] in ("normal", "overstock")]
    perm = rng.permutation(eligible_win)
    underbid, overbid = set(perm[:8]), set(perm[8:16])
    # weak listings: good traffic, poor product page (images/title/price) -> conversion well below category
    normal = sorted(pid for pid, st in prod_story.items() if st == "normal")
    weak_listing = set(np.random.default_rng([seed, 4]).choice(normal, size=3, replace=False))
    for pid in weak_listing:
        prod_story[pid] = "weak_listing"
    for e in entities:
        p = by_pid[e["product_id"]]
        cvr = rng.uniform(0.025, 0.045) * (2000.0 / p["price"]) ** 0.6
        if p["category"] == best_category:
            cvr *= 1.5
        if p["product_id"] in weak_listing:
            cvr *= WEAK_LISTING_CVR
        story = "normal"
        if e["junk"]:
            cvr, story = 0.0003, "junk"
        elif prod_story[p["product_id"]] == "best_seller_low_margin":
            cvr, story = rng.uniform(0.07, 0.09), "best_seller_low_margin"
        cap = float(np.exp(rng.normal(np.log(30), 0.45)))
        if story == "best_seller_low_margin":
            cap *= 3.0
        if e["entity_type"] == "product":
            cap *= 1.3
        pr = {"click_cap": cap, "k": float(rng.uniform(0.04, 0.12)), "cvr": float(cvr),
              "cpc_ratio": float(rng.uniform(0.6, 0.9))}
        v = pr["cvr"] * p["price"] * p["margin_pct"]
        opt = optimal_bid(pr, v, bids_cfg["min_bid"], bids_cfg["max_bid"])
        if story in ("junk", "best_seller_low_margin"):
            start = rng.uniform(15, 35) if story == "junk" else rng.uniform(35, 50)
        elif e["entity_id"] in underbid:
            start, story = opt * 0.5, "underbid_winner"
        elif e["entity_id"] in overbid:
            start, story = opt * 1.7, "overbid"
        else:
            start = opt * rng.uniform(0.8, 1.25)
        pr["true_optimal_bid"] = round(opt, 2)
        pr["story"] = story
        params[e["entity_id"]] = pr
        e["start_bid"] = float(round(np.clip(start, 5, 60), 1))

    # scale click caps so the "do nothing" account spends ~80% of the daily budget
    spend = sum(params[e["entity_id"]]["click_cap"] * (1 - np.exp(-params[e["entity_id"]]["k"] * e["start_bid"]))
                * e["start_bid"] * params[e["entity_id"]]["cpc_ratio"] for e in entities)
    scale = 0.8 * cfg["store"]["daily_budget"] / spend
    for eid in all_ids:
        params[eid]["click_cap"] = round(params[eid]["click_cap"] * scale, 3)

    # ---------- stock ----------
    exp_ad_sales = {pid: 0.0 for pid in by_pid}
    for e in entities:
        pr = params[e["entity_id"]]
        exp_ad_sales[e["product_id"]] += pr["click_cap"] * (1 - np.exp(-pr["k"] * e["start_bid"])) * pr["cvr"]
    rows = []
    for p in products:
        i, pid = p["idx"], p["product_id"]
        organic = round(float(rng.uniform(0.3, 2.0)), 2)
        if real:                       # real demand: average of the history window (what the seller can see)
            s = real["products"][i]["series"]
            organic = round(float(np.mean([v for d, v in s.items() if int(d) <= 0])), 2)
        elif i in best_sellers:        # best-sellers also sell well organically (ads partly cannibalize this)
            organic = round(organic * 3 + 3, 2)
        daily = organic + exp_ad_sales[pid]
        restock_day, restock_units = -1, 0
        if i in stockout:
            days_left = rng.uniform(1.5, 4.5)
            if i in restockers:
                restock_day = int(rng.integers(5, 10))
                restock_units = int(round(daily * 40))
        elif i in overstock:
            days_left = rng.uniform(70, 120)
        else:
            days_left = rng.uniform(20, 50)
        rows.append({"product_id": pid, "name": p["name"], "category": p["category"], "price": p["price"],
                     "margin_pct": p["margin_pct"], "stock_units": int(max(1, round(daily * days_left))),
                     "organic_sales_per_day": organic, "restock_day": restock_day, "restock_units": restock_units})
    products_df = pd.DataFrame(rows)
    # contribution margin breakdown: margin_pct is what is left after marketplace/fulfilment fees and returns
    frng = np.random.default_rng([seed, 3])
    fee = frng.uniform(0.08, 0.15, len(products_df)).round(3)
    returns = frng.uniform(0.02, 0.08, len(products_df)).round(3)
    products_df.insert(products_df.columns.get_loc("margin_pct"), "gross_margin_pct",
                       (products_df.margin_pct + fee + returns * RETURN_COST_SHARE).round(3))
    products_df.insert(products_df.columns.get_loc("margin_pct"), "fee_pct", fee)
    products_df.insert(products_df.columns.get_loc("margin_pct"), "return_rate", returns)
    entities_df = pd.DataFrame(entities)[["entity_id", "entity_type", "name", "product_id", "start_bid"]]
    # incrementality: share of ad sales that would NOT have happened organically
    irng = np.random.default_rng([seed, 5])
    product_params = {}
    for r in products_df.itertuples():
        st = prod_story[r.product_id]
        if st == "best_seller_low_margin":
            inc = irng.uniform(0.35, 0.55)     # strong organic demand: ads mostly take sales we'd get anyway
        elif r.organic_sales_per_day >= 1.5:
            inc = irng.uniform(0.60, 0.80)
        else:
            inc = irng.uniform(0.85, 1.00)
        product_params[r.product_id] = {"incrementality": round(float(inc), 3)}
        if real:
            product_params[r.product_id]["organic_series"] = real["products"][int(r.product_id[1:]) - 1]["series"]
    if real:
        products_df["source_id"] = [real["products"][i]["stock_code"] for i in range(n)]
    hidden = {
        "seed": seed,
        "product_params": product_params,
        "params": params,
        "product_story": prod_story,
        "junk_keywords": [e["entity_id"] for e in entities if e["junk"]],
        "best_category": best_category,
        "store_source": "uci_online_retail" if real else "synthetic",
        "weak_listing_products": sorted(weak_listing),
    }
    if real:
        hidden["real_meta"] = {"citation": real["citation"], "dates": real["dates"], "demand_scale": real["scale"],
                               "gbp_to_rs": real["gbp_to_rs"]}
    return products_df, entities_df, hidden


def _history(cfg, products_df, entities_df, hidden):
    seed, days = cfg["seed"], cfg["store"]["history_days"]
    rng = np.random.default_rng([seed, 2])
    ents = {r.entity_id: {"product_id": r.product_id} for r in entities_df.itertuples()}
    prods = {r.product_id: {"price": r.price, "margin_pct": r.margin_pct, "organic_sales_per_day": r.organic_sales_per_day,
                            "restock_day": -1, "restock_units": 0,
                            "incrementality": hidden["product_params"][r.product_id]["incrementality"],
                            "organic_series": hidden["product_params"][r.product_id].get("organic_series", {})}
             for r in products_df.itertuples()}
    start = dict(zip(entities_df.entity_id, entities_df.start_bid))
    stats, sold = [], []
    for day in range(-(days - 1), 1):
        bids = {eid: round(b * rng.uniform(0.7, 1.3), 2) for eid, b in start.items()}
        rows, srows, _ = simulate_day(day=day, seed=seed, entities=ents, params=hidden["params"], products=prods,
                                      bids=bids, stock=None, budget=cfg["store"]["daily_budget"])
        stats += rows
        sold += srows
    stats_df = pd.DataFrame(stats)
    sold_df = pd.DataFrame(sold)
    # walk stock backwards from today's stock level
    stock_rows = []
    level = dict(zip(products_df.product_id, products_df.stock_units))
    for day in range(0, -days, -1):
        today = sold_df[sold_df.day == day].set_index("product_id")
        for pid in products_df.product_id:
            stock_rows.append({"day": day, "product_id": pid, "stock_units": int(level[pid]),
                               "units_sold": int(today.loc[pid, "units_sold"]), "restocked_units": 0})
            level[pid] += int(today.loc[pid, "units_sold"])
    stock_df = pd.DataFrame(stock_rows).sort_values(["day", "product_id"]).reset_index(drop=True)
    return stats_df, stock_df


def generate(cfg: dict | None = None, out: str | Path | None = None) -> dict:
    cfg = cfg or load_config()
    o = out_dir(out)
    products_df, entities_df, hidden = _build(cfg)
    stats_df, stock_df = _history(cfg, products_df, entities_df, hidden)
    paths = {
        "products": o / "products.csv",
        "entities": o / "entities.csv",
        "history": o / "history.parquet",
        "stock_history": o / "stock_history.parquet",
        "hidden": o / "_hidden_market.json",
    }
    products_df.to_csv(paths["products"], index=False)
    entities_df.to_csv(paths["entities"], index=False)
    stats_df.to_parquet(paths["history"], index=False)
    stock_df.to_parquet(paths["stock_history"], index=False)
    paths["hidden"].write_text(json.dumps(hidden, indent=1), encoding="utf-8")
    # public store facts (not hidden): where the catalogue came from and which calendar day is which
    meta = {"source": hidden["store_source"]}
    if "real_meta" in hidden:
        meta.update({"citation": hidden["real_meta"]["citation"], "dates": hidden["real_meta"]["dates"],
                     "note": "Real products, prices and daily demand pattern; demand scaled to a small shop. "
                             "Ad clicks, costs, conversions, margins and stock are simulated."})
    (o / "store_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # a fresh store invalidates any live platform state
    for stale in ["platform_state.json", "daily_stats.parquet", "stock.parquet", "changes.parquet"]:
        (o / stale).unlink(missing_ok=True)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t = time.time()
    paths = generate(out=args.out)
    print(f"generated store in {time.time() - t:.2f}s -> {paths['products'].parent}")


if __name__ == "__main__":
    main()
