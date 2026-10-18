"""The MI300X speed estimate: sensible physics and honest labelling."""
import json
from pathlib import Path

from bench.estimate import EFFICIENCY, MI300X, QWEN32B, decode_rate, estimate, prefill_rate

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = {"gpu": "test", "results": [
    {"scenario": "bidding", "variant": "naive", "llm_calls": 160, "tokens_in": 37320, "tokens_out": 7612},
    {"scenario": "bidding", "variant": "bidpilot", "llm_calls": 92, "tokens_in": 25483, "tokens_out": 2701},
    {"scenario": "consultant", "variant": "sequential", "llm_calls": 3, "tokens_in": 881, "tokens_out": 34},
    {"scenario": "consultant", "variant": "parallel", "llm_calls": 3, "tokens_in": 881, "tokens_out": 34}]}


def test_physics_is_sensible():
    eff = EFFICIENCY["high"]
    one = decode_rate(MI300X, QWEN32B, eff, 1, 300)
    many = decode_rate(MI300X, QWEN32B, eff, 32, 300)
    assert 20 < one < 81            # can't beat bandwidth / model size (5300 GB/s / 65.6 GB ~ 81 tok/s)
    assert many < one and 32 * many > 10 * one   # batching: each slower, all together far faster
    assert prefill_rate(MI300X, QWEN32B, eff) > 1000


def test_estimate_is_labelled_and_consistent():
    est = estimate(WORKLOAD)
    assert est["is_estimate"] and "MI300X" in est["hardware"] and "how_to_confirm" in est
    lo, hi = est["range"]["low"], est["range"]["high"]
    for r in (lo, hi):
        assert r["bidding_bidpilot_s"] < r["bidding_naive_s"]
        assert r["consultant_parallel_s"] < r["consultant_sequential_s"]
    assert hi["bidding_naive_s"] < lo["bidding_naive_s"]          # better efficiency -> faster


def test_estimate_runs_on_the_real_measurement():
    res = ROOT / "bench" / "results.json"
    if res.exists():
        est = estimate(json.loads(res.read_text(encoding="utf-8")))
        assert est["range"]["high"]["speedup_bidding"] > 5
