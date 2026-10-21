"""Build the trades, price the costs, print the results.

    PYTHONPATH=src python src/study.py [start] [end]

Downloads are cached under data/raw/, so only the first run is slow.
"""

from __future__ import annotations

import argparse

import pandas as pd

from bhavcopy import DATA_DIR, discover_expiries
from costs import CostModel, apply_costs, breakeven_multiple, breakeven_slippage
from plots import plot_equity, plot_iv_vs_move, plot_pnl_distribution
from stats import (
    cost_sensitivity,
    on_margin,
    skew_by_unit,
    summarize,
    worst_trades,
    yearly_breakdown,
)
from strangle import build_trades

RULE = "=" * 72


def run(start: str, end: str, symbol: str, otm_pct: float, slippage: float,
        save_figures: bool) -> pd.DataFrame:
    print(RULE)
    print(f"Short {symbol} strangle, weekly expiries, {start} -> {end}")
    print(f"Strikes {otm_pct:.1%} out of the money, held to settlement")
    print(RULE)

    expiries = discover_expiries(start, end, symbol=symbol)
    print(f"\nExpiries discovered: {len(expiries)}")
    if len(expiries) < 2:
        raise SystemExit("Not enough expiries to form a trade.")

    trades = build_trades(expiries, symbol=symbol, otm_pct=otm_pct)
    if trades.empty:
        raise SystemExit("No trades could be built — check the cached data.")

    model = CostModel(slippage_points=slippage)
    trades = apply_costs(trades, model)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DATA_DIR / "trades.csv"
    trades.to_csv(out_csv, index=False)
    print(f"Trades built: {len(trades)}  (saved -> {out_csv})")

    gross = summarize(trades, "gross_pnl")
    net = summarize(trades, "net_pnl")

    print(f"\n{RULE}\n(1) Does the premium survive costs?\n{RULE}")
    print(f"  Mean premium sold          : {net['mean_premium']:8.2f} pts")
    print(f"  Mean gross P&L             : {gross['mean_points']:+8.2f} pts")
    print(f"  Mean cost                  : {trades['cost_points'].mean():8.2f} pts "
          f"({trades['cost_pct_of_premium'].mean():.1%} of premium)")
    print(f"  Mean net P&L               : {net['mean_points']:+8.2f} pts")
    print(f"  Total net P&L              : {net['total_points']:+8.1f} pts "
          f"over {net['trades']} trades")
    print(f"  Net edge as % of premium   : {net['edge_pct_of_premium']:+8.2%}")
    print(f"  Cost breakeven multiple    : {breakeven_multiple(trades, model):8.2f}x "
          f"(>1 means gross edge exceeds costs)")
    print(f"  Breakeven slippage per leg : {breakeven_slippage(trades, model):8.2f} pts")

    print(f"\n{RULE}\n(2) What does the distribution look like?\n{RULE}")
    print(f"  Win rate                   : {net['win_rate']:8.1%}")
    print(f"  Median net P&L             : {net['median_points']:+8.2f} pts")
    print(f"  Standard deviation         : {net['std_points']:8.2f} pts")
    print(f"  Skew                       : {net['skew']:+8.2f}")
    print(f"  Excess kurtosis            : {net['excess_kurtosis']:+8.2f}")
    print(f"  Best / worst trade         : {net['best']:+8.2f} / {net['worst']:+.2f} pts")
    print(f"  Worst loss : best gain     : {net['worst_to_best']:8.1f}x")
    print(f"  Max drawdown               : {net['max_drawdown_points']:+8.1f} pts")

    units = skew_by_unit(trades)
    print(f"\n  Unit of analysis matters:")
    print(f"    skew per position ({units['n_positions']:4d} obs) : "
          f"{units['skew_per_position']:+.2f}")
    print(f"    skew per leg      ({units['n_legs']:4d} obs) : "
          f"{units['skew_per_leg']:+.2f}")
    print(f"    -> a strangle is one position; the per-leg figure answers a")
    print(f"       different question and need not even share the same sign.")

    print(f"\n{RULE}\n(3) Returns on a stated capital base\n{RULE}")
    margin = on_margin(trades)
    print(f"  Margin assumed per lot     : Rs {margin['margin_per_lot']:,.0f} "
          f"(an assumption — every figure below scales with it)")
    print(f"  Mean return per trade      : {margin['mean_return_per_trade']:+8.3%}")
    print(f"  Annualized return          : {margin['annualized_return']:+8.2%}")
    print(f"  Annualized Sharpe          : {margin['sharpe_annualized']:8.2f}")
    print(f"  Max drawdown               : {margin['max_drawdown']:+8.2%}")
    print(f"  Total P&L per lot          : Rs {margin['total_rupees_per_lot']:+,.0f}")

    print(f"\n{RULE}\n(4) Sensitivity to the entry-fill assumption\n{RULE}")
    print(cost_sensitivity(trades).to_string(index=False))

    print(f"\n{RULE}\n(5) Year by year\n{RULE}")
    print(yearly_breakdown(trades).to_string())

    print(f"\n{RULE}\n(6) Worst five trades\n{RULE}")
    worst = worst_trades(trades, 5).copy()
    worst["entry_date"] = pd.to_datetime(worst["entry_date"]).dt.date
    worst["expiry"] = pd.to_datetime(worst["expiry"]).dt.date
    print(worst.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(RULE)

    if save_figures:
        figures = DATA_DIR.parent / "figures"
        figures.mkdir(exist_ok=True)
        plot_equity(trades, str(figures / "equity_curve.png"))
        plot_pnl_distribution(trades, str(figures / "pnl_distribution.png"))
        plot_iv_vs_move(trades, str(figures / "iv_vs_realized_move.png"))

    return trades


def main() -> None:
    p = argparse.ArgumentParser(description="Short Nifty strangle study")
    p.add_argument("start", nargs="?", default="2019-01-01")
    p.add_argument("end", nargs="?", default="2026-10-01")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--otm-pct", type=float, default=0.02,
                   help="strike distance from the forward (0.02 = 2%%)")
    p.add_argument("--slippage", type=float, default=0.25,
                   help="assumed slippage per leg, in index points")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    run(args.start, args.end, args.symbol, args.otm_pct, args.slippage,
        save_figures=not args.no_figures)


if __name__ == "__main__":
    main()
