"""Charts for the strangle study."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(trades: pd.DataFrame, save_path: str | None = None) -> None:
    """Cumulative net P&L in index points, with drawdown beneath."""
    pnl = trades.dropna(subset=["net_pnl"]).copy()
    equity = pnl["net_pnl"].cumsum()
    drawdown = equity - equity.cummax()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(pnl["entry_date"], equity, color="#2c3e50", linewidth=1.3)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Short Nifty strangle: cumulative net P&L",
                  fontsize=14, weight="bold")
    ax1.set_ylabel("Index points")
    ax1.grid(alpha=0.3)

    ax2.fill_between(pnl["entry_date"], drawdown, 0, color="#c0392b", alpha=0.55)
    ax2.set_ylabel("Drawdown (pts)")
    ax2.set_xlabel("Entry date")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    _finish(fig, save_path)


def plot_pnl_distribution(trades: pd.DataFrame, save_path: str | None = None) -> None:
    """Histogram of per-trade net P&L against the premium collected."""
    pnl = trades["net_pnl"].dropna()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(pnl, bins=60, color="#2c3e50", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.axvline(pnl.mean(), color="#27ae60", linewidth=1.6,
               label=f"mean {pnl.mean():+.1f} pts")

    ax.set_title("Per-trade net P&L: small steady gains, rare large losses",
                 fontsize=14, weight="bold")
    ax.set_xlabel("Net P&L per strangle (index points)")
    ax.set_ylabel("Trades")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _finish(fig, save_path)


def plot_iv_vs_move(trades: pd.DataFrame, save_path: str | None = None) -> None:
    """Volatility sold against the move that followed.

    Below the diagonal the index moved less than the options implied; above it
    are the weeks that pay for the rest.
    """
    df = trades.dropna(subset=["call_iv", "put_iv", "move_pct"]).copy()
    iv = (df["call_iv"] + df["put_iv"]) / 2
    weeks = df["days_held"] / 365.0
    implied_move = iv * weeks ** 0.5
    realized_move = df["move_pct"].abs()

    fig, ax = plt.subplots(figsize=(8, 8))
    breached = df["gross_pnl"] < 0
    ax.scatter(implied_move[~breached], realized_move[~breached], s=18,
               color="#27ae60", alpha=0.7, label="expired worthless")
    ax.scatter(implied_move[breached], realized_move[breached], s=22,
               color="#c0392b", alpha=0.8, label="strike breached")

    lim = max(implied_move.max(), realized_move.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linewidth=0.9, linestyle="--",
            label="realized = implied")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("Implied move sold vs. realized move", fontsize=14, weight="bold")
    ax.set_xlabel("Implied move over the week (sigma * sqrt(T))")
    ax.set_ylabel("Realized absolute move")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _finish(fig, save_path)


def _finish(fig, save_path: str | None) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure -> {save_path}")
        plt.close(fig)
    else:
        plt.show()
