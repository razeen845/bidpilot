"""Practice ad account: the only door from the agent to the (simulated) ad world.

No real ad platform API is ever called. The hidden market is loaded here and
never exposed through any public method.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from settings import load_config, out_dir, write_json, write_parquet
from sim.market import simulate_day

STATE_FILES = {
    "state": "platform_state.json",
    "stats": "daily_stats.parquet",
    "stock": "stock.parquet",
    "changes": "changes.parquet",
}


class AdPlatform:
    def __init__(self, cfg: dict | None = None, out: str | Path | None = None, persist: bool = False):
        self.cfg = cfg or load_config()
        self.out = out_dir(out)
        self.persist = persist
        self.min_bid = float(self.cfg["bids"]["min_bid"])
        self.max_bid = float(self.cfg["bids"]["max_bid"])
        self.budget = float(self.cfg["store"]["daily_budget"])
        self.seed = int(self.cfg["seed"])

        self._products = pd.read_csv(self.out / "products.csv")
        self._entities = pd.read_csv(self.out / "entities.csv")
        hidden = json.loads((self.out / "_hidden_market.json").read_text(encoding="utf-8"))
        self.__params = hidden["params"]
        self.__ents = {r.entity_id: {"product_id": r.product_id} for r in self._entities.itertuples()}
        self.__prods = {r.product_id: {"price": float(r.price), "margin_pct": float(r.margin_pct),
                                       "organic_sales_per_day": float(r.organic_sales_per_day),
                                       "restock_day": int(r.restock_day), "restock_units": int(r.restock_units),
                                       "incrementality": float(hidden.get("product_params", {})
                                                               .get(r.product_id, {}).get("incrementality", 1.0)),
                                       "organic_series": hidden.get("product_params", {})
                                       .get(r.product_id, {}).get("organic_series", {})}
                        for r in self._products.itertuples()}

        state_path = self.out / STATE_FILES["state"]
        if persist and state_path.exists():
            st = json.loads(state_path.read_text(encoding="utf-8"))
            self.day = int(st["day"])
            self._bids = {k: float(v) for k, v in st["bids"].items()}
            self._paused = set(st["paused"])
            self._stock = {k: int(v) for k, v in st["stock"].items()}
            self._stats = pd.read_parquet(self.out / STATE_FILES["stats"])
            self._stock_df = pd.read_parquet(self.out / STATE_FILES["stock"])
            cp = self.out / STATE_FILES["changes"]
            self._changes = pd.read_parquet(cp) if cp.exists() else pd.DataFrame()
        else:
            self.day = 0
            self._bids = dict(zip(self._entities.entity_id, self._entities.start_bid.astype(float)))
            self._paused = set()
            self._stock = dict(zip(self._products.product_id, self._products.stock_units.astype(int)))
            self._stats = pd.read_parquet(self.out / "history.parquet")
            self._stock_df = pd.read_parquet(self.out / "stock_history.parquet")
            self._changes = pd.DataFrame()

    # ---------------- read ----------------
    def get_report(self, day: int | None = None, lookback: int = 7) -> pd.DataFrame:
        day = self.day if day is None else day
        s = self._stats
        return s[(s.day > day - lookback) & (s.day <= day)].copy()

    def get_stats(self) -> pd.DataFrame:
        return self._stats.copy()

    def get_stock(self, day: int | None = None) -> pd.DataFrame:
        day = self.day if day is None else day
        s = self._stock_df
        return s[s.day == day][["product_id", "stock_units"]].reset_index(drop=True)

    def get_stock_history(self) -> pd.DataFrame:
        return self._stock_df.copy()

    def get_products(self) -> pd.DataFrame:
        return self._products.copy()

    def get_entities(self) -> pd.DataFrame:
        return self._entities.copy()

    def get_bids(self) -> dict:
        return dict(self._bids)

    def get_paused(self) -> set:
        return set(self._paused)

    def get_changes(self) -> pd.DataFrame:
        return self._changes.copy()

    def public_data(self) -> dict[str, pd.DataFrame]:
        """Everything the analyst sandbox may see. Never includes the hidden market."""
        return {
            "stats": self.get_stats(),
            "stock": self.get_stock_history(),
            "products": self.get_products(),
            "entities": self.get_entities(),
            "changes": self.get_changes(),
        }

    # ---------------- write ----------------
    def _clip(self, bid: float) -> float:
        return round(min(self.max_bid, max(self.min_bid, float(bid))), 2)

    def set_bids(self, bids: dict) -> None:
        for eid, b in bids.items():
            if eid not in self._bids:
                raise KeyError(eid)
            self._bids[eid] = self._clip(b)

    def pause(self, entity_id: str) -> None:
        if entity_id not in self._bids:
            raise KeyError(entity_id)
        self._paused.add(entity_id)

    def resume(self, entity_id: str) -> None:
        self._paused.discard(entity_id)

    def apply_changes(self, changes: list, approved_by: dict | str = "auto") -> None:
        """Apply Change records (pydantic objects or dicts) and log them."""
        rows = []
        for c in changes:
            c = c.model_dump() if hasattr(c, "model_dump") else dict(c)
            action = c["action"]
            if action == "pause":
                self.pause(c["entity_id"])
            elif action == "resume":
                self.resume(c["entity_id"])
                self.set_bids({c["entity_id"]: c["new_bid"]})
            elif action in ("raise", "lower"):
                self.set_bids({c["entity_id"]: c["new_bid"]})
            else:
                continue
            ev = c.pop("evidence", {}) or {}
            row = {"day": self.day + 1, **c, **{f"ev_{k}": v for k, v in ev.items()}}
            row["approved_by"] = approved_by.get(c["entity_id"], "auto") if isinstance(approved_by, dict) else approved_by
            rows.append(row)
        if rows:
            new = pd.DataFrame(rows)
            self._changes = new if self._changes.empty else pd.concat([self._changes, new], ignore_index=True)

    def step(self) -> dict:
        """Simulate the next day with the current bids. Returns the day's totals."""
        day = self.day + 1
        eff = {eid: (0.0 if eid in self._paused else b) for eid, b in self._bids.items()}
        rows, stock_rows, new_stock = simulate_day(
            day=day, seed=self.seed, entities=self.__ents, params=self.__params, products=self.__prods,
            bids=eff, stock=self._stock, budget=self.budget)
        stats = pd.DataFrame(rows)
        srows = pd.DataFrame(stock_rows)
        self._stats = pd.concat([self._stats, stats], ignore_index=True)
        self._stock_df = pd.concat([self._stock_df, srows], ignore_index=True)
        self._stock = new_stock
        self.day = day

        start_stock = dict(zip(srows.product_id, srows.stock_units + srows.units_sold))
        pid = stats.entity_id.map(dict(zip(self._entities.entity_id, self._entities.product_id)))
        out_of_stock = pid.map(start_stock) == 0
        totals = {
            "day": day,
            "spend": round(float(stats.cost.sum()), 2),
            "revenue": round(float(stats.revenue.sum()), 2),
            "profit": round(float(stats.profit.sum()), 2),
            "clicks": int(stats.clicks.sum()),
            "conversions": int(stats.conversions.sum()),
            "stockout_ad_spend": round(float(stats.cost[out_of_stock].sum()), 2),
        }
        # whole-business view: every unit sold (ads + organic) at contribution margin, minus ad spend
        price = srows.product_id.map({k: v["price"] for k, v in self.__prods.items()})
        margin = srows.product_id.map({k: v["margin_pct"] for k, v in self.__prods.items()})
        sales = srows.units_sold * price
        totals["total_sales"] = round(float(sales.sum()), 2)
        totals["organic_sales"] = round(totals["total_sales"] - totals["revenue"], 2)
        totals["total_contribution_profit"] = round(float((sales * margin).sum()) - totals["spend"], 2)
        totals["products_in_stock"] = int(sum(v > 0 for v in start_stock.values()))
        totals["products"] = len(start_stock)
        if self.persist:
            self.save()
        return totals

    def save(self) -> None:
        state = {"day": self.day, "bids": self._bids, "paused": sorted(self._paused), "stock": self._stock}
        write_json((self.out / STATE_FILES["state"]), state)
        write_parquet(self._stats, self.out / STATE_FILES["stats"])
        write_parquet(self._stock_df, self.out / STATE_FILES["stock"])
        if not self._changes.empty:
            write_parquet(self._changes, self.out / STATE_FILES["changes"])


def reset(cfg: dict | None = None, out: str | Path | None = None) -> None:
    """Throw away live state so the next persistent platform starts at day 0."""
    o = out_dir(out)
    for f in STATE_FILES.values():
        (o / f).unlink(missing_ok=True)
