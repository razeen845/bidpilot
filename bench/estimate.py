"""Estimate BidPilot's speed on an AMD Instinct MI300X with vLLM, from the workload measured on the laptop.

This is an ESTIMATE, not a measurement. Run `python -m cli bench` on the MI300X to replace it with real numbers.

Method (standard roofline model for LLM inference):
  * decode (writing tokens) is memory-bound: every step reads all model weights plus the KV cache of every
    request in the batch, so  step time = (weights + batch KV bytes) / (bandwidth x efficiency).
    One request gets 1 token per step; a batch of c requests gets c tokens per step for nearly the same cost.
    That is why parallel calls pay off on a big GPU and not on a laptop iGPU.
  * prefill (reading the prompt) is compute-bound: 2 x parameters FLOPs per prompt token.
  * the workload (calls, prompt and output tokens per call, CPU time for the optimizer and the sandbox) is
    taken from bench/results.json, measured on the laptop with the same prompts.
  * a low/high efficiency range brackets typical vLLM results (low = conservative, high = well-tuned).

Hardware: AMD Instinct MI300X datasheet - 192 GB HBM3, 5.3 TB/s, 1307.4 TFLOPS dense FP16.
Model: Qwen2.5-32B-Instruct in BF16 (32.8 B parameters, 64 layers, 8 KV heads x 128 dims).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MI300X = {"name": "AMD Instinct MI300X (vLLM, ROCm)", "bandwidth_gbs": 5300.0, "tflops": 1307.4}
QWEN32B = {"name": "Qwen2.5-32B-Instruct (BF16)", "params_b": 32.8, "bytes_per_param": 2.0, "layers": 64,
           "kv_heads": 8, "head_dim": 128}
EFFICIENCY = {   # share of the theoretical peak that vLLM typically reaches
    "low": {"bandwidth": 0.40, "compute": 0.30, "overhead_s": 0.05},
    "high": {"bandwidth": 0.65, "compute": 0.50, "overhead_s": 0.02},
}
CONCURRENCY = 32          # llm.max_concurrency in config.yaml
CPU_PLAN_S = 2.2          # optimizer + guardrails + workflows (CPU), measured on the laptop
SANDBOX_S = 2.54          # one sandboxed analysis (CPU), measured: (16.70 s - 9.08 s of LLM) / 3


def _model_bytes(m):
    return m["params_b"] * 1e9 * m["bytes_per_param"]


def _kv_bytes_per_token(m):
    return m["layers"] * m["kv_heads"] * m["head_dim"] * 2 * m["bytes_per_param"]   # K and V


def decode_rate(hw, m, eff, batch: int, context: float) -> float:
    """Tokens/second for ONE request when `batch` requests decode together."""
    step_bytes = _model_bytes(m) + batch * context * _kv_bytes_per_token(m)
    return hw["bandwidth_gbs"] * 1e9 * eff["bandwidth"] / step_bytes


def prefill_rate(hw, m, eff) -> float:
    return hw["tflops"] * 1e12 * eff["compute"] / (2 * m["params_b"] * 1e9)


def sequential_llm(hw, m, eff, calls, tin, tout) -> float:
    per_in, per_out = tin / calls, tout / calls
    one = per_in / prefill_rate(hw, m, eff) + per_out / decode_rate(hw, m, eff, 1, per_in + per_out / 2)
    return calls * (one + eff["overhead_s"])


def batched_llm(hw, m, eff, calls, tin, tout, concurrency=CONCURRENCY) -> float:
    per_in, per_out = tin / calls, tout / calls
    waves = math.ceil(calls / concurrency)
    t = 0.0
    for w in range(waves):
        c = min(concurrency, calls - w * concurrency)
        t += (c * per_in / prefill_rate(hw, m, eff)
              + per_out / decode_rate(hw, m, eff, c, per_in + per_out / 2) + eff["overhead_s"])
    return t


def estimate(results: dict, hw=MI300X, m=QWEN32B) -> dict:
    by = {(r["scenario"], r["variant"]): r for r in results["results"]}
    nv, bp, cs = by[("bidding", "naive")], by[("bidding", "bidpilot")], by[("consultant", "sequential")]
    out = {}
    for level, eff in EFFICIENCY.items():
        naive = sequential_llm(hw, m, eff, nv["llm_calls"], nv["tokens_in"], nv["tokens_out"])
        # BidPilot: CPU planning, 1 planner call, then the explanations in parallel batches
        n = bp["llm_calls"]
        per_in, per_out = bp["tokens_in"] / n, bp["tokens_out"] / n
        bidpilot = (CPU_PLAN_S + sequential_llm(hw, m, eff, 1, per_in, per_out)
                    + batched_llm(hw, m, eff, n - 1, per_in * (n - 1), per_out * (n - 1)))
        q = cs["llm_calls"]
        cons_seq = q * SANDBOX_S + sequential_llm(hw, m, eff, q, cs["tokens_in"], cs["tokens_out"])
        cons_par = SANDBOX_S * 1.15 + batched_llm(hw, m, eff, q, cs["tokens_in"], cs["tokens_out"])
        out[level] = {
            "bidding_naive_s": round(naive, 1), "bidding_bidpilot_s": round(bidpilot, 1),
            "consultant_sequential_s": round(cons_seq, 1), "consultant_parallel_s": round(cons_par, 1),
            "speedup_bidding": round(naive / bidpilot, 1), "speedup_consultant": round(cons_seq / cons_par, 1),
            "single_stream_tokens_per_s": round(decode_rate(hw, m, eff, 1, 300), 1),
            "batch32_total_tokens_per_s": round(32 * decode_rate(hw, m, eff, 32, 300), 0),
            "prefill_tokens_per_s": round(prefill_rate(hw, m, eff), 0),
        }
    return {"hardware": hw["name"], "model": m["name"], "is_estimate": True,
            "method": "roofline model (memory-bound decode, compute-bound prefill) on the workload measured on the "
                      "laptop; low/high = conservative/well-tuned vLLM efficiency",
            "measured_on": results.get("gpu"), "concurrency": CONCURRENCY, "range": out,
            "how_to_confirm": "python -m cli bench   (on the MI300X with LLM_BASE_URL pointing at vLLM)"}


def main() -> int:
    res_path = ROOT / "bench" / "results.json"
    if not res_path.exists():
        print("bench/results.json missing: run the laptop benchmark first (python -m cli bench)")
        return 1
    est = estimate(json.loads(res_path.read_text(encoding="utf-8")))
    (ROOT / "bench" / "estimate_mi300x.json").write_text(json.dumps(est, indent=1), encoding="utf-8")
    print(json.dumps(est, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
