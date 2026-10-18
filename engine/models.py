"""Shared records: the Change record every workflow produces."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Workflow = Literal["checkup", "bid_adjust", "waste", "stock_guard", "consultant", "experiment", "rollback"]
Action = Literal["raise", "lower", "pause", "hold", "resume"]
Role = Literal["discovery", "proven", "waste_candidate", "standard"]
Confidence = Literal["high", "medium", "low"]


class Evidence(BaseModel):
    spend_7d: float = 0.0
    clicks_7d: float = 0.0
    conversions_7d: float = 0.0
    revenue_7d: float = 0.0
    roas_7d: float = 0.0
    profit_7d: float = 0.0
    acos_7d: float = 0.0             # spend / attributed sales (fraction)
    cvr_7d: float = 0.0              # conversions / clicks (fraction)
    cpc_7d: float = 0.0
    days_of_stock: float = 0.0
    # comparison period: the 7 days before
    spend_prev_7d: float = 0.0
    conversions_prev_7d: float = 0.0
    roas_prev_7d: float = 0.0
    acos_prev_7d: float = 0.0


class Change(BaseModel):
    entity_id: str
    entity_type: Literal["product", "keyword"]
    name: str
    workflow: Workflow
    action: Action
    old_bid: float
    new_bid: float
    change_pct: float = 0.0          # fraction, e.g. -0.2 = -20%
    evidence: Evidence = Field(default_factory=Evidence)
    needs_approval: bool = False
    approval_reason: str | None = None
    reason: str | None = None
    reason_source: str | None = None   # llm | template
    role: Role = "standard"
    confidence: Confidence = "medium"
    objective: str = "profit"             # profit | growth (seller-chosen, per product)
    kpi: str = "roas"                     # primary KPI of the platform profile (acos | roas)
    platform_action: str | None = None  # how the action maps to the platform, e.g. "add as negative exact keyword"

    def model_post_init(self, __context) -> None:
        if self.old_bid > 0 and self.action in ("raise", "lower"):
            self.change_pct = round(self.new_bid / self.old_bid - 1.0, 4)
