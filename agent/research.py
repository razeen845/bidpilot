"""Web research with verified citations.

research(queries) -> [W sources]
  live:  search API (Tavily or Brave) -> page text (API content, else httpx + trafilatura)
         -> LLM extracts <= 2 facts per page, each with an exact quote (<= 25 words)
         -> quote check against the page text -> keep only verified facts
         -> save out/research/run_{timestamp}.json (+ page texts) and number sources W1, W2, ...
  cache: load the latest run file (out/research, else the shipped data/research_cache) and re-run the
         quote check against the saved page text. No network.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import httpx

from agent import prompts as P
from agent.llm import run_sync
from settings import ROOT, out_dir, write_json

SHIPPED_CACHE = ROOT / "data" / "research_cache"
UA = {"User-Agent": "Mozilla/5.0 (BidPilot research bot; hackathon demo)"}

_QUOTE_MAP = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "“": '"', "”": '"',
              "„": '"', "″": '"', "–": "-", "—": "-", "−": "-", " ": " ", "…": "..."}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    for a, b in _QUOTE_MAP.items():
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip().casefold()


def quote_check(quote: str, page_text: str, max_words: int = 25) -> tuple[bool, str]:
    q = normalize(quote).strip(" .\"'")
    if not q:
        return False, "empty quote"
    if len(q.split()) > max_words:
        return False, f"quote longer than {max_words} words"
    if q not in normalize(page_text):
        return False, "quote not found in page text"
    return True, "ok"


# ---------------- search + fetch ----------------

def search(query: str, cfg: dict, n: int) -> list[dict]:
    r = cfg["research"]
    key = r.get("api_key", "")
    if not key:
        raise RuntimeError("SEARCH_API_KEY not set")
    if r.get("provider", "tavily") == "tavily":
        resp = httpx.post("https://api.tavily.com/search", timeout=30, json={
            "api_key": key, "query": query, "max_results": n, "include_raw_content": True})
        resp.raise_for_status()
        return [{"url": x["url"], "title": x.get("title", ""), "content": x.get("raw_content") or ""}
                for x in resp.json().get("results", [])[:n]]
    resp = httpx.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": n}, timeout=30,
                     headers={"X-Subscription-Token": key, "Accept": "application/json"})
    resp.raise_for_status()
    return [{"url": x["url"], "title": x.get("title", ""), "content": ""}
            for x in resp.json().get("web", {}).get("results", [])[:n]]


def fetch_text(url: str) -> str:
    import trafilatura
    resp = httpx.get(url, headers=UA, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return trafilatura.extract(resp.text, include_comments=False, include_tables=True, favor_recall=True) or ""


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _save_pages(run: dict, pages: dict[str, str], rdir: Path) -> None:
    pdir = rdir / "pages"
    pdir.mkdir(parents=True, exist_ok=True)
    for sha, text in pages.items():
        (pdir / f"{sha}.txt").write_text(text, encoding="utf-8")


def _number(sources: list[dict]) -> list[dict]:
    for i, s in enumerate(sources, 1):
        s["id"] = f"W{i}"
    return sources


async def _extract(llm, query: str, page: dict, max_words: int) -> list[dict]:
    try:
        out = await llm.chat_json("research", P.RESEARCH_SYSTEM,
                                  P.research_user(query, page["title"], page["url"], page["text"]), P.FactsOut,
                                  max_tokens=400, temperature=0.0)
    except Exception:  # noqa: BLE001
        return []
    found = []
    for f in out["facts"][:2]:
        ok, why = quote_check(f["quote"], page["text"], max_words)
        found.append({"query": query, "url": page["url"], "title": page["title"], "fact": f["fact"].strip(),
                      "quote": f["quote"].strip(), "verified": ok, "check": why})
    return found


async def research_live_async(queries: list[str], cfg: dict, llm, out=None) -> dict:
    r = cfg["research"]
    queries = queries[: int(r.get("max_queries", 4))]
    n = int(r.get("results_per_query", 3))
    max_words = int(r.get("max_quote_words", 25))
    pages, page_meta, tasks = {}, [], []
    for q in queries:
        try:
            results = await asyncio.to_thread(search, q, cfg, n)
        except Exception as e:  # noqa: BLE001
            page_meta.append({"query": q, "error": f"search failed: {e}"[:200]})
            continue
        for res in results:
            text = res.get("content") or ""
            if len(text) < 400:
                try:
                    text = await asyncio.to_thread(fetch_text, res["url"])
                except Exception as e:  # noqa: BLE001
                    page_meta.append({"query": q, "url": res["url"], "error": f"fetch failed: {e}"[:200]})
                    continue
            if len(text) < 200:
                continue
            sha = _sha(text)
            pages[sha] = text
            page = {"url": res["url"], "title": res["title"], "text": text}
            page_meta.append({"query": q, "url": res["url"], "title": res["title"], "text_sha": sha})
            tasks.append(_extract(llm, q, page, max_words))
    facts = [f for group in await asyncio.gather(*tasks) for f in group]
    retrieved = time.strftime("%Y-%m-%dT%H:%M:%S")
    sha_of = {m["url"]: m["text_sha"] for m in page_meta if "text_sha" in m}
    for f in facts:
        f["retrieved_at"] = retrieved
        f["text_sha"] = sha_of.get(f["url"])
    run = {"run_id": time.strftime("%Y%m%d_%H%M%S"), "created_at": retrieved, "mode": "live",
           "provider": r.get("provider"), "queries": queries, "pages": page_meta,
           "rejected": [f for f in facts if not f["verified"]],
           "sources": _number([f for f in facts if f["verified"]])}
    rdir = out_dir(out) / "research"
    rdir.mkdir(parents=True, exist_ok=True)
    _save_pages(run, pages, rdir)
    write_json((rdir / f"run_{run['run_id']}.json"), run)
    return run


def latest_run_file(out=None) -> Path | None:
    for d in (out_dir(out) / "research", SHIPPED_CACHE):
        runs = sorted(d.glob("run_*.json"))
        if runs:
            return runs[-1]
    return None


def load_cached(cfg: dict, out=None) -> dict:
    """Latest run file, with every source re-verified against its saved page text."""
    path = latest_run_file(out)
    if path is None:
        return {"run_id": None, "mode": "cache", "sources": [], "error": "no research cache found"}
    run = json.loads(path.read_text(encoding="utf-8"))
    max_words = int(cfg["research"].get("max_quote_words", 25))
    kept = []
    for s in run.get("sources", []):
        page = path.parent / "pages" / f"{s.get('text_sha')}.txt"
        if not page.exists():
            continue
        ok, why = quote_check(s["quote"], page.read_text(encoding="utf-8"), max_words)
        if ok:
            kept.append({**s, "verified": True, "check": why})
    run["sources"] = _number(kept)
    run["mode"] = "cache"
    run["cache_file"] = str(path)
    return run


async def research_async(queries: list[str], cfg: dict, llm=None, out=None) -> dict:
    if cfg["research"].get("mode", "cache") == "live" and llm is not None:
        return await research_live_async(queries, cfg, llm, out)
    return await asyncio.to_thread(load_cached, cfg, out)


def research(queries: list[str], cfg: dict, llm=None, out=None) -> dict:
    return run_sync(research_async(queries, cfg, llm, out))


def page_text_for(source: dict, out=None) -> str | None:
    for d in (out_dir(out) / "research", SHIPPED_CACHE):
        p = d / "pages" / f"{source.get('text_sha')}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None
