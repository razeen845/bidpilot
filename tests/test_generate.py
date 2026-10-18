import json
import time

import pandas as pd

from analysis.metrics import stock_metrics
from data.generate import generate
from settings import load_config
from sim.platform import AdPlatform


def test_generation_fast_and_deterministic(cfg, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    t = time.time()
    generate(cfg, a)
    assert time.time() - t < 10
    generate(cfg, b)
    for f in ["products.csv", "entities.csv"]:
        assert (a / f).read_bytes() == (b / f).read_bytes()
    for f in ["history.parquet", "stock_history.parquet"]:
        pd.testing.assert_frame_equal(pd.read_parquet(a / f), pd.read_parquet(b / f))
    assert json.loads((a / "_hidden_market.json").read_text()) == json.loads((b / "_hidden_market.json").read_text())


def test_different_seed_differs(cfg, tmp_path):
    generate(cfg, tmp_path / "a")
    generate(load_config(overrides={"seed": 7}), tmp_path / "b")
    assert (tmp_path / "a" / "entities.csv").read_bytes() != (tmp_path / "b" / "entities.csv").read_bytes()


def test_store_shape(store):
    p = pd.read_csv(store / "products.csv")
    e = pd.read_csv(store / "entities.csv")
    assert len(p) == 40
    assert (e.entity_type == "keyword").sum() == 120
    assert (e.entity_type == "product").sum() == 40
    assert e.name.is_unique
    assert p.price.between(800, 6000).all()
    assert e.start_bid.between(5, 60).all()
    assert ((p.margin_pct >= 0.40) & (p.margin_pct <= 0.60)).sum() >= 20
    assert ((p.margin_pct >= 0.08) & (p.margin_pct <= 0.15)).sum() >= 5
    h = pd.read_parquet(store / "history.parquet")
    assert h.day.nunique() == 30


def test_junk_keywords_near_zero_and_patterned(store):
    hidden = json.loads((store / "_hidden_market.json").read_text())
    junk = set(hidden["junk_keywords"])
    assert 15 <= len(junk) <= 21  # ~15% of 120
    h = pd.read_parquet(store / "history.parquet")
    j = h[h.entity_id.isin(junk)]
    assert j.clicks.sum() > 500
    assert j.conversions.sum() / j.clicks.sum() < 0.005
    names = pd.read_csv(store / "entities.csv").set_index("entity_id").loc[sorted(junk), "name"]
    patterned = names.str.contains(r"\b(?:free|cheap|diy)\b")
    assert patterned.mean() > 0.6


def test_products_hit_low_stock_within_14_days(cfg, store):
    plat = AdPlatform(cfg, store)
    low = set()
    for _ in range(14):
        plat.step()
        sm = stock_metrics(plat.get_stock_history(), plat.day)
        low |= set(sm[sm.days_of_stock < cfg["guardrails"]["low_stock_days"]].index)
    assert len(low) >= 4


def test_best_category_converts_better(store):
    h = pd.read_parquet(store / "history.parquet")
    e = pd.read_csv(store / "entities.csv")
    p = pd.read_csv(store / "products.csv")
    df = h.merge(e, on="entity_id").merge(p[["product_id", "category", "margin_pct"]], on="product_id")
    df["gp"] = df.revenue * df.margin_pct
    g = df.groupby("category")[["gp", "cost"]].sum()
    assert ((g.gp - g.cost) / g.cost).idxmax() == "Bedding"
