"""Real data: store built from the UCI Online Retail dataset, and the read-only audit of real Facebook ads."""
import json

import numpy as np
import pandas as pd
import pytest

from agent.analyst import REAL_ADS_QUESTIONS, ask
from analysis.real_ads import audit, load_real_ads, with_real_ads
from data.generate import generate
from data.real_store import load_real_catalog
from sandbox.exec_runner import run_code
from settings import ROOT, load_config
from sim.platform import AdPlatform
from sim.runner import run, summarize

REAL = ROOT / "data" / "real"


@pytest.fixture(scope="module")
def real_cfg():
    return load_config(overrides={"llm": {"enabled": False}, "store": {"source": "uci_online_retail"}})


@pytest.fixture(scope="module")
def real_store(real_cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("real")
    generate(real_cfg, out)
    return out


def test_real_catalog_comes_from_the_dataset(real_cfg, real_store):
    meta = pd.read_csv(REAL / "uci_textiles.csv", dtype={"stock_code": str})
    p = pd.read_csv(real_store / "products.csv", dtype={"source_id": str})
    assert len(p) == 40 and set(p.source_id) == set(meta.stock_code)
    row = p.iloc[0]
    m = meta.set_index("stock_code").loc[row.source_id]
    assert row["name"] == m.description.title()
    assert row.price == round(m.price_gbp * real_cfg["real_data"]["gbp_to_rs"], -1)
    sm = json.loads((real_store / "store_meta.json").read_text())
    assert sm["source"] == "uci_online_retail" and "UCI" in sm["citation"] and sm["dates"]["0"] == "2011-10-14"


def test_organic_demand_follows_the_real_series(real_cfg, real_store):
    cat = load_real_catalog(real_cfg)
    hidden = json.loads((real_store / "_hidden_market.json").read_text())
    s = hidden["product_params"]["P01"]["organic_series"]
    assert s == cat["products"][0]["series"]
    # real demand pattern: rank correlation between real daily units (smoothed) and simulated organic sales
    daily = pd.read_csv(REAL / "uci_textiles_daily.csv", dtype={"stock_code": str})
    p = AdPlatform(real_cfg, real_store)
    for _ in range(14):
        p.step()
    sold = p.get_stock_history()
    tot = sold[sold.day > -30].groupby("product_id").units_sold.sum()
    real_tot = daily.groupby("stock_code").units.sum()
    ids = pd.read_csv(real_store / "products.csv", dtype={"source_id": str}).set_index("product_id").source_id
    corr = pd.Series(tot.values, index=ids[tot.index].values).corr(real_tot, method="spearman")
    assert corr > 0.6


def test_agent_wins_on_real_store(real_cfg, real_store):
    res = {s: summarize(run(s, cfg=real_cfg, out=real_store)) for s in ["fixed", "rules", "agent_nollm"]}
    a = res["agent_nollm"]
    assert a["total_contribution_profit"] > max(res["fixed"]["total_contribution_profit"],
                                                res["rules"]["total_contribution_profit"])
    assert a["profit"] > res["rules"]["profit"] and a["stockout_ad_spend"] <= res["rules"]["stockout_ad_spend"]


def test_real_ads_audit_matches_hand_computation():
    raw = pd.read_csv(REAL / "KAG_conversion_data.csv")
    a = audit(load_real_ads())
    assert a["ads"] == len(raw) == 1143
    assert a["total_spend"] == pytest.approx(raw.Spent.sum(), abs=0.05)
    assert a["total_purchases"] == raw.Approved_Conversion.sum()
    # every flagged ad: 0 purchases, enough spend/clicks, and zero sales unlikely at its campaign's rate
    camp = raw.groupby("xyz_campaign_id")[["Approved_Conversion", "Clicks"]].sum()
    rate = (camp.Approved_Conversion / camp.Clicks).to_dict()
    for r in a["waste_ads"].itertuples():
        assert r.purchases == 0 and r.spend >= 20 and r.clicks >= 20
        assert (1 - rate[int(r.campaign_id)]) ** r.clicks < 0.05
    # low-data protection: a zero-purchase ad with few clicks is never flagged
    few = raw[(raw.Approved_Conversion == 0) & (raw.Spent >= 20) & (raw.Clicks < 20)]
    assert not set(few.ad_id.astype(str)) & set(a["waste_ads"].ad_id)


def test_real_ads_presets_and_sandbox(cfg, store, tmp_path):
    data = with_real_ads(AdPlatform(cfg, store).public_data())
    assert "real_ads" in data
    src = ask(REAL_ADS_QUESTIONS[1], data, cfg, out=tmp_path)
    assert src["ok"]
    t = pd.DataFrame(src["result_table"])
    raw = pd.read_csv(REAL / "KAG_conversion_data.csv")
    g = raw.groupby(["age", "gender"])[["Spent", "Approved_Conversion"]].sum()
    best = (g.Spent / g.Approved_Conversion.replace(0, np.nan)).idxmin()
    assert (t.iloc[0].age, t.iloc[0].gender) == best
    # without the optional table, code that uses it fails cleanly inside the sandbox
    r = run_code("result = real_ads.head()", AdPlatform(cfg, store).public_data(), cfg)
    assert not r["ok"] and "real_ads" in r["error"]
