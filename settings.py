"""Config loading, validation, logging and safe file writes shared by every module.

Environment overrides (so secrets never live in config.yaml):
  BIDPILOT_CONFIG     path to an alternative config file
  BIDPILOT_OUT        output directory (default: ./out)
  BIDPILOT_LOG_LEVEL  DEBUG | INFO | WARNING (default INFO)
  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL   OpenAI-compatible endpoint
  LLM_MAX_CONCURRENCY, LLM_TIMEOUT_S     match the server (e.g. 4 slots on a small local server)
  SEARCH_API_KEY      web search key (Tavily or Brave)
"""
from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parent


class ConfigError(ValueError):
    """Raised with a readable message when config.yaml is invalid."""


# ---------------- config schema ----------------
class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class StoreCfg(_Loose):
    source: Literal["synthetic", "uci_online_retail"] = "synthetic"
    n_products: int = Field(gt=0, le=1000)
    n_keywords: int = Field(gt=0)
    history_days: int = Field(ge=7, le=365)
    sim_days: int = Field(gt=0, le=365)
    daily_budget: float = Field(gt=0)


class BidsCfg(_Loose):
    min_bid: float = Field(gt=0)
    max_bid: float = Field(gt=0)
    max_daily_change: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _range(self):
        if self.min_bid >= self.max_bid:
            raise ValueError("bids.min_bid must be smaller than bids.max_bid")
        return self


class GuardrailsCfg(_Loose):
    approval_change_pct: float = Field(ge=0, le=1)
    approval_top_spenders: int = Field(ge=0)
    min_clicks_to_act: int = Field(ge=0)
    waste_min_spend_7d: float = Field(ge=0)
    waste_min_clicks_7d: int = Field(ge=0)
    low_stock_days: float = Field(ge=0)
    overstock_days: float = Field(gt=0)
    waste_window_days: int = Field(default=14, ge=1)
    conversion_lag_days: int = Field(default=1, ge=0)
    waste_confidence: float = Field(default=0.95, ge=0, lt=1)
    protect_top_profit: int = Field(default=10, ge=0)
    protected_max_cut: float = Field(default=0.10, ge=0, le=1)


class LLMCfg(_Loose):
    enabled: bool = True
    base_url: str
    model_planner: str
    model_explainer: str
    json_mode: Literal["schema", "prompt"] = "schema"
    max_concurrency: int = Field(default=32, ge=1, le=1024)
    timeout_s: float = Field(default=30, gt=0)


class SandboxCfg(_Loose):
    backend: Literal["docker", "subprocess"] = "subprocess"
    timeout_s: int = Field(default=10, ge=1, le=600)
    memory_mb: int = Field(default=512, ge=64)
    max_result_rows: int = Field(default=50, ge=1, le=10_000)
    max_fix_attempts: int = Field(default=1, ge=0, le=5)


class ResearchCfg(_Loose):
    mode: Literal["live", "cache"] = "cache"
    provider: Literal["tavily", "brave"] = "tavily"
    max_queries: int = Field(default=4, ge=0, le=20)
    results_per_query: int = Field(default=3, ge=1, le=10)
    max_quote_words: int = Field(default=25, ge=3, le=100)


class ConsultantCfg(_Loose):
    run_on_days: list[int] = Field(default_factory=lambda: [7, 14])
    max_recommendations: int = Field(default=5, ge=1, le=20)
    max_analysis_questions: int = Field(default=3, ge=0, le=10)


class ObjectiveCfg(_Loose):
    mode: Literal["profit", "growth"] = "profit"
    max_acos: float = Field(default=0.5, gt=0, le=5)


class RollbackCfg(_Loose):
    enabled: bool = True
    window_days: int = Field(default=4, ge=2, le=30)
    min_bid_move: float = Field(default=0.15, ge=0, le=5)
    min_drop_per_day: float = Field(default=300, ge=0)
    p_value: float = Field(default=0.05, gt=0, lt=1)
    cooldown_days: int = Field(default=3, ge=0)


class ExperimentsCfg(_Loose):
    days: int = Field(default=14, ge=4, le=60)
    max_ci_width: float = Field(default=0.6, gt=0)
    min_days_each: int = Field(default=4, ge=2)
    factor_floor: float = Field(default=0.1, ge=0, le=1)


class ConfigModel(_Loose):
    seed: int
    platform: Literal["generic", "amazon", "google"] = "generic"
    store: StoreCfg
    bids: BidsCfg
    guardrails: GuardrailsCfg
    objectives: dict[str, ObjectiveCfg] = Field(default_factory=dict)
    incrementality: dict[str, float] = Field(default_factory=dict)
    llm: LLMCfg
    sandbox: SandboxCfg
    research: ResearchCfg
    consultant: ConsultantCfg
    rollback: RollbackCfg = Field(default_factory=RollbackCfg)
    experiments: ExperimentsCfg = Field(default_factory=ExperimentsCfg)


def validate_config(cfg: dict) -> dict:
    try:
        ConfigModel.model_validate(cfg)
    except ValidationError as e:
        lines = [f"  {'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise ConfigError("invalid config:\n" + "\n".join(lines)) from None
    for pid, f in (cfg.get("incrementality") or {}).items():
        if not 0 < float(f) <= 1.5:
            raise ConfigError(f"invalid config:\n  incrementality.{pid}: must be in (0, 1.5]")
    return cfg


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    p = Path(path or os.environ.get("BIDPILOT_CONFIG", ROOT / "config.yaml"))
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {p}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"config file is not valid YAML: {p}: {e}") from None
    llm = cfg.setdefault("llm", {})
    if os.environ.get("LLM_BASE_URL"):
        llm["base_url"] = os.environ["LLM_BASE_URL"]
    if os.environ.get("LLM_API_KEY"):
        llm["api_key"] = os.environ["LLM_API_KEY"]
    if os.environ.get("LLM_MODEL"):
        llm["model_planner"] = llm["model_explainer"] = os.environ["LLM_MODEL"]
    if os.environ.get("LLM_MAX_CONCURRENCY"):
        llm["max_concurrency"] = int(os.environ["LLM_MAX_CONCURRENCY"])
    if os.environ.get("LLM_TIMEOUT_S"):
        llm["timeout_s"] = float(os.environ["LLM_TIMEOUT_S"])
    cfg.setdefault("research", {})["api_key"] = os.environ.get("SEARCH_API_KEY", "")
    cfg = _merge(cfg, overrides or {})
    cfg.setdefault("objectives", cfg.get("objectives") or {})
    cfg.setdefault("incrementality", cfg.get("incrementality") or {})
    validate_config(cfg)
    from engine.platform_rules import apply_profile  # local import: engine imports settings
    return apply_profile(cfg)


def out_dir(path: str | os.PathLike | None = None) -> Path:
    p = Path(path or os.environ.get("BIDPILOT_OUT", ROOT / "out"))
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------- safe writes ----------------
def _atomic(path: Path, write) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    """Write JSON atomically (temp file + rename), so a crash never leaves a half-written state file."""
    def w(tmp):
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=str)
    _atomic(Path(path), w)


def write_text(path: str | os.PathLike, text: str) -> None:
    _atomic(Path(path), lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))


def write_parquet(df, path: str | os.PathLike) -> None:
    _atomic(Path(path), lambda tmp: df.to_parquet(tmp, index=False))


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return copy.deepcopy(default)
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------- logging ----------------
_LOGGING_READY = False


def get_logger(name: str) -> logging.Logger:
    global _LOGGING_READY
    if not _LOGGING_READY:
        level = os.environ.get("BIDPILOT_LOG_LEVEL", "INFO").upper()
        root = logging.getLogger("bidpilot")
        if not root.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(h)
        root.setLevel(getattr(logging, level, logging.INFO))
        root.propagate = False
        _LOGGING_READY = True
    return logging.getLogger(f"bidpilot.{name}")


__all__ = ["ROOT", "ConfigError", "load_config", "validate_config", "out_dir", "write_json", "write_text",
           "write_parquet", "read_json", "get_logger"]
