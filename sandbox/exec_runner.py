"""Run untrusted (AI-written) analysis code outside the main process.

run_code(code, data) -> {ok, result_table, result_summary, error, ...}

* `sandbox/policy.py` must approve the code first.
* Only the allowed tables (stats, stock, products, entities, changes) are copied into a temp dir as
  parquet. Never the hidden market.
* docker backend:     docker run --rm --network none --read-only --memory {memory_mb}m, data mounted read-only.
* subprocess backend: separate `python -I` process, near-empty env, cwd = temp dir, timeout;
                      on Linux also RLIMIT_CPU / RLIMIT_AS (floor 1.5 GB so numpy/pandas can import).
                      Windows has no setrlimit: timeout only.
* Inside, a tiny harness preloads the DataFrames, gives the code a restricted set of builtins and an
  import hook that only allows pandas / numpy / math / statistics, then prints the result as JSON.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from sandbox.policy import check_code

ALLOWED_TABLES = ("stats", "stock", "products", "entities", "changes")
OPTIONAL_TABLES = ("real_ads",)          # real ad export, only when provided
FORBIDDEN_COLUMNS = {"click_cap", "k", "cvr", "cpc_ratio", "true_optimal_bid", "story"}
MARKER = "@@BIDPILOT_RESULT@@"
DOCKER_IMAGE = "bidpilot-sandbox:latest"
LINUX_MIN_AS_MB = 1536

HARNESS = r'''
import builtins as _b, json, math, statistics, sys, traceback
import numpy as np
import pandas as pd

MARKER = "@@BIDPILOT_RESULT@@"
TABLES = ("stats", "stock", "products", "entities", "changes", "real_ads")
SAFE = ["abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "int", "isinstance", "len", "list", "map", "max", "min", "pow", "print", "range", "reversed", "round",
        "set", "slice", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError", "KeyError",
        "IndexError", "TypeError", "ZeroDivisionError", "True", "False", "None"]
MODULES = {"pandas": pd, "numpy": np, "math": math, "statistics": statistics}


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in MODULES:
        raise ImportError("import not allowed in sandbox: " + str(name))
    return MODULES[name]


def jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if f != f or f in (float("inf"), float("-inf")) else round(f, 4)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (int, str)) or v is None:
        return v
    return str(v)


def to_frame(obj):
    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name=obj.name if obj.name is not None else "value")
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(x) for x in c if str(x)) for c in df.columns]
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index(drop=all(n is None for n in df.index.names))
        df.columns = [str(c) for c in df.columns]
        return df
    if isinstance(obj, (int, float, str, bool, np.generic)):
        return pd.DataFrame([{"value": obj}])
    raise TypeError("result must be a DataFrame, Series or scalar, got " + type(obj).__name__)


def main():
    d = sys.argv[1]
    max_rows = int(sys.argv[2])
    code = open(d + "/code.py", encoding="utf-8").read()
    import os
    fmt = "pkl" if os.path.exists(d + "/stats.pkl") else "parquet"
    # .pkl files are written by our own trusted parent process (same interpreter); parquet is used for Docker
    frames = {t: (pd.read_pickle if fmt == "pkl" else pd.read_parquet)(d + "/" + t + "." + fmt) for t in TABLES
              if os.path.exists(d + "/" + t + "." + fmt)}
    builtins = {n: getattr(_b, n) for n in SAFE}
    builtins["__import__"] = safe_import
    g = {"__builtins__": builtins, "pd": pd, "np": np, "math": math, "statistics": statistics}
    g.update(frames)
    try:
        exec(compile(code, "<analysis>", "exec"), g)
        if "result" not in g:
            raise NameError("the code must assign a variable named `result`")
        df = to_frame(g["result"])
        records = [{k: jsonable(v) for k, v in r.items()} for r in df.head(max_rows).to_dict(orient="records")]
        summary = g.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        summary = {str(k): jsonable(v) for k, v in list(summary.items())[:10]}
        if not summary and len(df) == 1 and list(df.columns) == ["value"]:
            summary = {"value": records[0]["value"]}
        out = {"ok": True, "result_table": records, "result_summary": summary,
               "total_rows": int(len(df)), "truncated": bool(len(df) > max_rows)}
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<analysis>"), None)
        out = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:400] + (" (line %s)" % line if line else "")}
    sys.stdout.write("\n" + MARKER + json.dumps(out) + "\n")


main()
'''


def _limits(cpu_s: int, mem_mb: int):  # pragma: no cover - Linux only
    def set_limits():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
    return set_limits


def _write_tables(data: dict[str, pd.DataFrame], d: Path, fmt: str = "parquet") -> None:
    for t in ALLOWED_TABLES + OPTIONAL_TABLES:
        df = data.get(t)
        if df is None:
            if t in OPTIONAL_TABLES:
                continue
            df = pd.DataFrame()
        bad = FORBIDDEN_COLUMNS & set(map(str, df.columns))
        if bad:
            raise ValueError(f"refusing to pass hidden-market columns to the sandbox: {sorted(bad)}")
        if fmt == "pkl":
            df.to_pickle(d / f"{t}.pkl")
        else:
            df.to_parquet(d / f"{t}.parquet", index=False)


def run_code(code: str, data: dict[str, pd.DataFrame], cfg: dict) -> dict:
    sb = cfg["sandbox"]
    backend = sb.get("backend", "subprocess")
    timeout = int(sb.get("timeout_s", 10))
    mem = int(sb.get("memory_mb", 512))
    max_rows = int(sb.get("max_result_rows", 50))

    pol = check_code(code)
    if not pol.ok:
        return {"ok": False, "error": "policy rejected: " + "; ".join(pol.reasons), "policy_reasons": pol.reasons,
                "sandbox_backend": backend}

    with tempfile.TemporaryDirectory(prefix="bidpilot_sb_") as td:
        d = Path(td)
        _write_tables(data, d, "parquet" if backend == "docker" else "pkl")
        (d / "code.py").write_text(code, encoding="utf-8")
        (d / "harness.py").write_text(HARNESS, encoding="utf-8")
        kwargs: dict = {"capture_output": True, "text": True, "timeout": timeout, "cwd": td}
        if backend == "docker":
            cmd = ["docker", "run", "--rm", "--network", "none", "--read-only", "--memory", f"{mem}m", "--cpus", "1",
                   "--pids-limit", "64", "-v", f"{td}:/work:ro", "-w", "/work", sb.get("docker_image", DOCKER_IMAGE),
                   "python", "-I", "/work/harness.py", "/work", str(max_rows)]
        else:
            cmd = [sys.executable, "-I", str(d / "harness.py"), td, str(max_rows)]
            env = {"PYTHONIOENCODING": "utf-8", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1"}
            if os.name == "nt":
                env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
            else:  # pragma: no cover
                # address-space cap: numpy/pandas need ~1 GB of virtual memory just to import
                kwargs["preexec_fn"] = _limits(timeout, max(mem, LINUX_MIN_AS_MB))
            kwargs["env"] = env
        try:
            proc = subprocess.run(cmd, **kwargs)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout: code ran longer than {timeout}s", "sandbox_backend": backend}
        except FileNotFoundError as e:
            return {"ok": False, "error": f"sandbox backend unavailable: {e}", "sandbox_backend": backend}

    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(MARKER)), None)
    if line is None:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"ok": False, "error": f"sandbox crashed (exit {proc.returncode}): {err[-1] if err else 'no output'}",
                "sandbox_backend": backend}
    res = json.loads(line[len(MARKER):])
    res["sandbox_backend"] = backend
    return res
