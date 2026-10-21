"""Performance and distribution statistics for the strangle trades.

Two conventions, both of which change how risky the strategy looks:

* The unit of analysis is the position, not the leg — a strangle is one trade
  with one payoff. `skew_by_unit` reports both.
* Results are in index points and as a share of premium first, and only then on
  a capital base, since a percentage return needs a denominator the strategy
  does not supply. `MARGIN_PER_LOT` is that assumption, stated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

# Indicative SPAN + exposure margin for one short Nifty strangle lot. An
# assumption, not a measurement: real margin moves with volatility.
MARGIN_PER_LOT = 150_000.0

WEEKS_PER_YEAR = 52


def summarize(trades: pd.DataFrame, pnl_col: str = "net_pnl") -> dict:
    """Headline statistics for one P&L column, in index points."""
    pnl = trades[pnl_col].dropna()
    if pnl.empty:
        return {}

    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()

    return {
        "trades": int(len(pnl)),
        "mean_points": float(pnl.mean()),
        "median_points": float(pnl.median()),
        "std_points": float(pnl.std()),
        "total_points": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "skew": float(sps.skew(pnl.values)),
        "excess_kurtosis": float(sps.kurtosis(pnl.values)),
        "worst": float(pnl.min()),
        "best": float(pnl.max()),
        "worst_to_best": float(abs(pnl.min()) / pnl.max()) if pnl.max() > 0 else float("nan"),
        "max_drawdown_points": float(drawdown.min()),
        "mean_premium": float(trades["premium"].mean()),
        "edge_pct_of_premium": float(pnl.sum() / trades["premium"].sum()),
    }


def skew_by_unit(trades: pd.DataFrame) -> dict:
    """Skew per position versus per leg, because the two can disagree in sign.

    The position is the unit to score — it is what the account holds. Per-leg
    figures answer a different question: on one published intraday
    short-strangle dataset, per-leg skew is +0.37 against -0.33 per position,
    the difference between "thin tails" and the negative skew short volatility
    is known for. The direction of the gap depends on the payoff structure, so
    both numbers are reported.
    """
    position = (trades["gross_pnl"]).dropna()

    call_leg = trades["call_entry"] - trades["call_exit"]
    put_leg = trades["put_entry"] - trades["put_exit"]
    legs = pd.concat([call_leg, put_leg]).dropna()

    return {
        "skew_per_position": float(sps.skew(position.values)),
        "skew_per_leg": float(sps.skew(legs.values)),
        "kurtosis_per_position": float(sps.kurtosis(position.values)),
        "kurtosis_per_leg": float(sps.kurtosis(legs.values)),
        "n_positions": int(len(position)),
        "n_legs": int(len(legs)),
    }


def on_margin(trades: pd.DataFrame, pnl_col: str = "net_pnl",
              margin_per_lot: float = MARGIN_PER_LOT) -> dict:
    """Point P&L as returns on a stated margin base.

    Every figure scales inversely with `margin_per_lot`, so it is returned
    alongside the results.
    """
    df = trades.dropna(subset=[pnl_col, "lot_size"])
    if df.empty:
        return {}

    rupees = df[pnl_col] * df["lot_size"]
    ret = rupees / margin_per_lot

    equity = ret.cumsum()
    drawdown = equity - equity.cummax()
    years = len(ret) / WEEKS_PER_YEAR

    return {
        "margin_per_lot": margin_per_lot,
        "mean_return_per_trade": float(ret.mean()),
        "total_return": float(ret.sum()),
        "annualized_return": float(ret.sum() / years) if years else float("nan"),
        "return_std_per_trade": float(ret.std()),
        "sharpe_annualized": float(ret.mean() / ret.std() * np.sqrt(WEEKS_PER_YEAR))
            if ret.std() > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "total_rupees_per_lot": float(rupees.sum()),
    }


def worst_trades(trades: pd.DataFrame, n: int = 5, pnl_col: str = "net_pnl") -> pd.DataFrame:
    """The n largest losses, with the index move that caused them."""
    cols = ["entry_date", "expiry", "forward_entry", "settle_level", "move_pct",
            "premium", "gross_pnl", pnl_col]
    cols = [c for c in cols if c in trades.columns]
    return trades.nsmallest(n, pnl_col)[cols]


def yearly_breakdown(trades: pd.DataFrame, pnl_col: str = "net_pnl") -> pd.DataFrame:
    """Per-year trade count, total points and worst trade.

    An aggregate mean hides whether the record rests on a few calm years.
    """
    df = trades.dropna(subset=[pnl_col]).copy()
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year

    out = df.groupby("year").agg(
        trades=(pnl_col, "size"),
        total_points=(pnl_col, "sum"),
        mean_points=(pnl_col, "mean"),
        win_rate=(pnl_col, lambda s: (s > 0).mean()),
        worst=(pnl_col, "min"),
    )
    return out.round(3)


def cost_sensitivity(trades: pd.DataFrame, slippages: list[float] | None = None) -> pd.DataFrame:
    """Mean net P&L across a range of per-leg slippage assumptions.

    Entry uses the closing price, the one number a real execution would not
    have matched exactly.
    """
    from costs import CostModel, apply_costs

    slippages = slippages if slippages is not None else [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
    rows = []
    for s in slippages:
        priced = apply_costs(trades, CostModel(slippage_points=s))
        rows.append({
            "slippage_points_per_leg": s,
            "mean_net_points": priced["net_pnl"].mean(),
            "total_net_points": priced["net_pnl"].sum(),
            "win_rate": (priced["net_pnl"] > 0).mean(),
        })
    return pd.DataFrame(rows).round(4)
