"""Indian retail cost model for short index-option trades.

Held to expiry, each leg is sold once and then cash settled, so there is no
exit brokerage and no exercise STT — that falls on the option buyer, not the
writer. Costs are returned in index points per unit so they compare directly
with the premium, the only denominator the strategy itself supplies.

Rates default to discount-broker levels and are arguments, not constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# STT on selling options rose from 0.0625% to 0.1% of premium on 1 Oct 2024.
STT_HIKE_DATE = pd.Timestamp("2024-10-01")
STT_BEFORE = 0.000625
STT_AFTER = 0.001


@dataclass
class CostModel:
    """Per-order and percentage charges for writing one strangle."""
    brokerage_per_order: float = 20.0
    exchange_rate: float = 0.0005      # 0.05% of premium (NSE options)
    sebi_rate: float = 0.000001        # Rs 10 per crore
    gst_rate: float = 0.18
    stamp_rate: float = 0.0            # buy side only; a writer pays none
    slippage_points: float = 0.25      # per leg, half a typical bid-ask
    legs: int = 2

    def stt_rate(self, trade_date: pd.Timestamp) -> float:
        return STT_AFTER if pd.Timestamp(trade_date) >= STT_HIKE_DATE else STT_BEFORE

    def cost_points(self, premium: float, trade_date: pd.Timestamp,
                    lot_size: float) -> float:
        """Cost of one strangle in index points per unit.

        `premium` is both legs combined, in points.
        """
        if not lot_size or lot_size <= 0:
            return float("nan")

        premium_value = premium * lot_size

        brokerage = self.brokerage_per_order * self.legs
        pct_rates = self.stt_rate(trade_date) + self.exchange_rate + self.sebi_rate
        pct_charges = premium_value * pct_rates
        gst = self.gst_rate * (brokerage + premium_value * (self.exchange_rate + self.sebi_rate))

        rupee_cost = brokerage + pct_charges + gst
        slippage = self.slippage_points * self.legs

        return rupee_cost / lot_size + slippage


def apply_costs(trades: pd.DataFrame, model: CostModel | None = None) -> pd.DataFrame:
    """Add cost and net-P&L columns to a trade table, in index points."""
    model = model or CostModel()
    out = trades.copy()

    out["cost_points"] = [
        model.cost_points(r.premium, r.entry_date, r.lot_size)
        for r in out.itertuples()
    ]
    out["net_pnl"] = out["gross_pnl"] - out["cost_points"]
    out["cost_pct_of_premium"] = out["cost_points"] / out["premium"]
    return out


def breakeven_slippage(trades: pd.DataFrame, model: CostModel | None = None) -> float:
    """Slippage per leg, in points, that would reduce mean net P&L to zero.

    If a fraction of a point decides the sign, the result is a fill assumption
    rather than an edge.
    """
    model = model or CostModel()
    base = model.slippage_points
    priced = apply_costs(trades, model)
    mean_net = priced["net_pnl"].mean()
    return base + mean_net / model.legs


def breakeven_multiple(trades: pd.DataFrame, model: CostModel | None = None) -> float:
    """Multiple of current costs the edge could absorb before breaking even."""
    model = model or CostModel()
    priced = apply_costs(trades, model)
    gross, cost = priced["gross_pnl"].mean(), priced["cost_points"].mean()
    if cost <= 0:
        return float("nan")
    return float(gross / cost)
