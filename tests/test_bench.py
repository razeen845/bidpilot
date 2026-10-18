import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bench_dry_run_never_writes_real_results(tmp_path):
    real = ROOT / "bench" / "results.json"
    before = real.read_text() if real.exists() else None
    r = subprocess.run([sys.executable, str(ROOT / "bench" / "latency.py"), "--dry-run", "--runs", "1",
                        "--fake-latency", "0.01"], capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((ROOT / "bench" / "results_dryrun.json").read_text())
    assert res["dry_run"] and res["speedup_consultant"] > 1
    assert {(x["scenario"], x["variant"]) for x in res["results"]} == {
        ("bidding", "naive"), ("bidding", "bidpilot"), ("consultant", "sequential"), ("consultant", "parallel")}
    assert (real.read_text() if real.exists() else None) == before
