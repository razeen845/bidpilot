"""Platform rule profiles: generic | amazon | google.

Platforms report and action things differently, so their rules stay separate instead of assuming every
marketplace uses the same signals. A profile changes the rules and the wording only; no real platform
API is ever called (the account is simulated).

  conversion_lag_days      how many of the newest days' clicks may still convert (waste rule ignores them)
  attribution_window_days  the platform's click-attribution window (shown to the user, used in reports)
  primary_kpi              acos (Amazon convention) or roas (Google convention) in explanations
  waste_action_*           what "stop wasting money" means on that platform
"""
from __future__ import annotations

import copy

PROFILES: dict[str, dict] = {
    "generic": {
        "label": "Generic practice account (simulated)",
        "primary_kpi": "roas",
        "conversion_lag_days": 1,
        "attribution_window_days": 7,
        "waste_action_keyword": "pause keyword",
        "waste_action_product": "pause product ad",
    },
    "amazon": {
        "label": "Amazon Sponsored Products (rules profile, simulated account)",
        "primary_kpi": "acos",
        "conversion_lag_days": 2,          # sponsored-ads sales can take ~48h to appear in reports
        "attribution_window_days": 7,      # Sponsored Products: 7-day click attribution
        "waste_action_keyword": "add as negative exact keyword",
        "waste_action_product": "pause product ad",
    },
    "google": {
        "label": "Google Ads Search + Shopping (rules profile, simulated account)",
        "primary_kpi": "roas",
        "conversion_lag_days": 3,          # conversions keep arriving for days after the click
        "attribution_window_days": 30,     # default click-through conversion window
        "waste_action_keyword": "add as negative phrase keyword",
        "waste_action_product": "exclude product from Shopping campaign",
    },
}


def get_profile(cfg: dict) -> dict:
    return PROFILES[cfg.get("platform", "generic")]


def apply_profile(cfg: dict) -> dict:
    """Copy of cfg with the profile's lag applied (unless the profile is generic, which keeps config values)."""
    cfg = copy.deepcopy(cfg)
    name = cfg.get("platform", "generic")
    prof = PROFILES[name]
    if name != "generic":
        cfg["guardrails"]["conversion_lag_days"] = prof["conversion_lag_days"]
    cfg["platform_profile"] = {"name": name, **prof}
    return cfg


def waste_action(cfg: dict, entity_type: str) -> str:
    prof = cfg.get("platform_profile") or get_profile(cfg)
    return prof["waste_action_keyword"] if entity_type == "keyword" else prof["waste_action_product"]


def primary_kpi(cfg: dict) -> str:
    return (cfg.get("platform_profile") or get_profile(cfg))["primary_kpi"]
