"""Baseline strategies the agent must beat."""
from __future__ import annotations

from analysis.metrics import entity_metrics


class FixedStrategy:
    """The lazy seller: never touches the bids."""
    name = "fixed"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def act(self, platform, day: int) -> None:
        return None


class RulesStrategy:
    """What basic tools do: 7-day ROAS below break-even -> x0.9, otherwise x1.05. No stock awareness."""
    name = "rules"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def act(self, platform, day: int) -> None:
        m = entity_metrics(platform.get_stats(), platform.day, 7)
        ents = platform.get_entities().set_index("entity_id")
        margin = platform.get_products().set_index("product_id").margin_pct
        bids = platform.get_bids()
        new = {}
        for eid, row in m.iterrows():
            if row.spend_7d <= 0:
                continue
            breakeven = 1.0 / float(margin[ents.loc[eid, "product_id"]])
            new[eid] = bids[eid] * (0.9 if row.roas_7d < breakeven else 1.05)
        platform.set_bids(new)
