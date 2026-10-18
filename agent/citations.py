"""Citation registry and checks for one consultant run.

check_text(text, cited_ids):
  * every [X#] tag in the text (and every cited id) exists in the registry;
  * at least one citation;
  * every number in the text matches a value in a cited D source (result table, summary or question)
    or appears in a cited W source's fact/quote (Phase 4 number check, rounding allowed);
  * cited W sources must be verified (quote found in the retrieved page).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.explain import check_numbers, numeric_values

TAG_RE = re.compile(r"\[([DW]\d+)\]")


@dataclass
class CheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


class CitationRegistry:
    def __init__(self):
        self.sources: dict[str, dict] = {}

    def add(self, src: dict) -> str:
        sid = src["id"]
        if not re.fullmatch(r"[DW]\d+", sid or ""):
            raise ValueError(f"bad source id {sid!r}")
        self.sources[sid] = src
        return sid

    def add_many(self, sources: list[dict]) -> None:
        for s in sources:
            self.add(s)

    def get(self, sid: str) -> dict | None:
        return self.sources.get(sid)

    def ids(self) -> list[str]:
        return sorted(self.sources, key=lambda s: (s[0], int(s[1:])))

    def catalog(self, max_rows: int = 12) -> list[dict]:
        """Compact view of the sources for the LLM writer."""
        out = []
        for sid in self.ids():
            s = self.sources[sid]
            if sid.startswith("D"):
                out.append({"id": sid, "question": s["question"], "summary": s.get("result_summary", {}),
                            "table": s.get("result_table", [])[:max_rows]})
            else:
                out.append({"id": sid, "title": s.get("title"), "url": s.get("url"), "fact": s.get("fact"),
                            "quote": s.get("quote"), "verified": s.get("verified", False)})
        return out

    def allowed_values(self, ids: set[str]) -> list[float]:
        vals: list[float] = []
        for sid in ids:
            s = self.sources.get(sid)
            if not s:
                continue
            if sid.startswith("D"):
                vals += numeric_values([s.get("result_table", []), s.get("result_summary", {}), s.get("question", "")])
            else:
                vals += numeric_values([s.get("fact", ""), s.get("quote", "")])
        return vals

    def check_text(self, text: str, cited_ids: list[str] | None = None) -> CheckResult:
        errors = []
        tags = set(TAG_RE.findall(text or ""))
        ids = tags | set(cited_ids or [])
        if not ids:
            errors.append("no citations")
        unknown = sorted(i for i in ids if i not in self.sources)
        if unknown:
            errors.append(f"unknown source tags: {unknown}")
        for i in sorted(ids):
            s = self.sources.get(i)
            if s and i.startswith("W") and not s.get("verified"):
                errors.append(f"{i} is not a verified web source")
        bad = check_numbers(text or "", self.allowed_values(ids))
        if bad:
            errors.append(f"numbers not found in cited sources: {bad}")
        # the store trades in Rs; only the real ad export (cited D source about real ads) may use other currency
        cites_real = any("real" in (self.sources.get(i) or {}).get("question", "").lower() for i in ids)
        if re.search(r"\$|\bUSD\b|\bdollars?\b", text or "", re.I) and not cites_real:
            errors.append("wrong currency: the store's money is in Rs")
        return CheckResult(not errors, errors)
