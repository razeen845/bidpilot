"""Mutation testing: plant one realistic bug at a time in a COPY of the project and check the tests catch it."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(r"C:\Users\razee\Documents\bidman")
WORK = Path(r"C:\Users\razee\AppData\Local\Temp\claude\c--Users-razee-Documents-bidman\57b1b0a0-a394-4e99-ab0b-7a24cc5f6a9c\scratchpad\mutant_repo")
PY = str(SRC / ".venv" / "Scripts" / "python.exe")

MUTANTS = [
    ("optimizer picks the WORST bid", "engine/optimizer.py", "best = int(np.argmax(profit))", "best = int(np.argmin(profit))"),
    ("+-20%/day safety limit removed", "engine/guardrails.py", "new = min(old * (1 + m), max(old * (1 - m), new))", "new = new"),
    ("stock guard cuts 10% instead of 50%", "engine/guardrails.py", "bids[eid] * 0.5, cfg, daily_limit=False", "bids[eid] * 0.9, cfg, daily_limit=False"),
    ("out-of-stock products not paused", "engine/guardrails.py", "        if units <= 0:\n            props[eid] = (\"pause\", bids[eid])", "        if units < 0:\n            props[eid] = (\"pause\", bids[eid])"),
    ("waste rule ignores statistical evidence", "engine/guardrails.py", "if p0 < 1.0 - conf:", "if p0 <= 1.0:"),
    ("big changes no longer need approval", "engine/guardrails.py", "abs(change.change_pct) > cfg[\"guardrails\"][\"approval_change_pct\"]", "abs(change.change_pct) > 5.0"),
    ("proven winners not protected", "agent/workflows.py", "                new = G.protect_proven(old, new, cfg)", "                pass"),
    ("daily budget not enforced", "sim/market.py", "if total_cost > budget > 0:", "if False:"),
    ("paused ads still get clicks", "sim/platform.py", "eff = {eid: (0.0 if eid in self._paused else b)", "eff = {eid: (b if eid in self._paused else b)"),
    ("number check accepts any number", "agent/explain.py", "                    return True\n    return False", "                    return True\n    return True"),
    ("explanations may talk about price", "agent/explain.py", "re.search(r\"\\bprices?\\b\", reason, re.I)", "re.search(r\"NEVERMATCH\", reason, re.I)"),
    ("citation check accepts unknown sources", "agent/citations.py", "unknown = sorted(i for i in ids if i not in self.sources)", "unknown = []"),
    ("web quotes not verified against the page", "agent/research.py", "    if q not in normalize(page_text):", "    if False:"),
    ("sandbox allows any import", "sandbox/policy.py", "if a.name.split(\".\")[0] not in ALLOWED_IMPORTS or \".\" in a.name:", "if False:"),
    ("sandbox allows dunder attributes", "sandbox/policy.py", "if a.startswith(\"_\"):", "if a.startswith(\"_____\"):"),
    ("hidden market columns reach the sandbox", "sandbox/exec_runner.py", "        if bad:\n            raise ValueError", "        if False:\n            raise ValueError"),
    ("incrementality estimate sign wrong", "agent/experiments.py", "est = (m_on - m_off) / a_on", "est = (m_on + m_off) / a_on"),
    ("rollback tests the wrong direction", "engine/rollback.py", "alternative=\"greater\"", "alternative=\"less\""),
    ("cautious raises disabled", "engine/optimizer.py", "profit = np.where(cand > cur + 1e-9, profit_low, profit)", "profit = profit"),
    ("REJECTED changes get applied", "agent/orchestrator.py", "elif decisions.get(eid) == \"approve\":", "elif decisions.get(eid) in (\"approve\", \"reject\"):"),
    ("unverified consultant bid actions applied", "agent/consultant.py", "if r[\"type\"] == \"bid_action\" and r.get(\"bid_action\") and r[\"status\"] == \"verified\":", "if r[\"type\"] == \"bid_action\" and r.get(\"bid_action\"):"),
    ("ACOS computed upside down", "analysis/metrics.py", "(s / rev.where(rev > 0, np.nan))", "(rev / s.where(s > 0, np.nan))"),
    ("restocked products never resume", "engine/guardrails.py", "if lw == (\"stock_guard\", \"pause\") and units > 0:", "if lw == (\"stock_guard\", \"pause\") and units > 10**9:"),
    ("junk keywords convert like normal ones", "data/generate.py", "cvr, story = 0.0003, \"junk\"", "cvr, story = 0.03, \"junk\""),
    ("reviewed analyses bypassed for AI code", "agent/analyst.py", "if llm is not None and question not in PRESET_CODE:", "if llm is not None:"),
]

TEST_ARGS = ["-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
             "--ignore=tests/test_dashboard.py", "--ignore=tests/test_bench.py", "-k", "not cli_end_to_end"]


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK, ignore=shutil.ignore_patterns(".venv", "out", ".git", "__pycache__", "*.xlsx", "*.zip",
                                                               "online_retail_raw.parquet", ".pytest_cache"))
    results = []
    for name, rel, old, new in MUTANTS:
        f = WORK / rel
        original = f.read_text(encoding="utf-8")
        if original.count(old) != 1:
            results.append({"mutant": name, "status": "SKIPPED (pattern not found)"})
            print(results[-1], flush=True)
            continue
        f.write_text(original.replace(old, new), encoding="utf-8")
        t = time.time()
        r = subprocess.run([PY, *TEST_ARGS], cwd=WORK, capture_output=True, text=True,
                           env={**__import__("os").environ, "BIDPILOT_LOG_LEVEL": "WARNING"}, timeout=1800)
        f.write_text(original, encoding="utf-8")
        failed = [ln.split(" - ")[0].replace("FAILED ", "") for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
        status = "CAUGHT" if r.returncode != 0 else "MISSED"
        results.append({"mutant": name, "status": status, "by": failed[:2], "seconds": round(time.time() - t)})
        print(results[-1], flush=True)
    caught = sum(r["status"] == "CAUGHT" for r in results)
    tried = sum(r["status"] in ("CAUGHT", "MISSED") for r in results)
    print(f"\nMUTATION SCORE: {caught}/{tried} planted bugs caught")
    (WORK.parent / "mutation_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
