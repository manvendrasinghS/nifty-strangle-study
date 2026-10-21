"""Construct short-strangle trades from the daily option chain.

One trade per weekly expiry: at the close of the previous expiry day, sell a
call and a put a given distance out of the money and hold to settlement. The
exit is then NSE's own settlement value, so no intraday fill is assumed.

The reference price is the forward from put-call parity, F = K + (C - P),
rather than the futures close: it matches the option's own expiry (futures are
monthly), needs no rate assumption, and works in both bhavcopy formats, one of
which has no underlying-price column.
"""

from __future__ import annotations

import pandas as pd
import requests

from bhavcopy import infer_lot_size, load_day
from blackscholes import black76_delta, implied_vol, intrinsic

DAYS_PER_YEAR = 365.0


def option_chain(date: pd.Timestamp, symbol: str = "NIFTY",
                 session: requests.Session | None = None) -> pd.DataFrame:
    """Traded index-option rows for one date, with stale quotes removed.

    The bhavcopy carries a previous close forward for contracts that never
    traded, so untraded strikes are dropped rather than treated as executable.
    """
    day = load_day(date, symbol=symbol, session=session)
    if day is None:
        return pd.DataFrame()

    chain = day[(day["instrument"] == "OPTIDX") & day["option_type"].notna()].copy()
    chain = chain[(chain["volume"] > 0) & (chain["oi"] > 0) & (chain["close"] > 0)]
    return chain.reset_index(drop=True)


def forward_from_parity(chain: pd.DataFrame, expiry: pd.Timestamp,
                        min_pairs: int = 3) -> float:
    """Forward price for `expiry` implied by put-call parity, or NaN.

    F = K + (C - P) for every strike quoted on both sides, rates ignored, taking
    the median near the money where both legs are liquid. At expiry the identity
    is exact, so this also recovers the settlement level.
    """
    legs = chain[chain["expiry"] == expiry]
    if legs.empty:
        return float("nan")

    pivot = legs.pivot_table(index="strike", columns="option_type",
                             values="close", aggfunc="last")
    if not {"CE", "PE"}.issubset(pivot.columns):
        return float("nan")

    pairs = pivot.dropna(subset=["CE", "PE"])
    if len(pairs) < min_pairs:
        return float("nan")

    implied = pairs.index + (pairs["CE"] - pairs["PE"])
    # Closest to the money, judged by the smallest call-put spread.
    near = (pairs["CE"] - pairs["PE"]).abs().nsmallest(max(min_pairs, 5)).index
    return float(implied.loc[near].median())


def settlement_level(chain: pd.DataFrame, expiry: pd.Timestamp) -> float:
    """NSE's settlement level for options expiring on `expiry`.

    On expiry day the settlement-price field of every expiring row holds the
    settlement value of the *index*, not the option — the same number repeated
    down the column. That is what cash settlement is computed against, so it is
    read directly, falling back to parity only if the field is missing.
    """
    expiring = chain[(chain["expiry"] == expiry) & chain["settle"].notna()]
    if not expiring.empty:
        level = expiring["settle"].median()
        if pd.notna(level) and level > 0:
            return float(level)
    return forward_from_parity(chain, expiry)


def _pick_leg(legs: pd.DataFrame, target_strike: float, option_type: str) -> pd.Series | None:
    """The traded strike closest to `target_strike` on the requested side."""
    side = legs[legs["option_type"] == option_type]
    if side.empty:
        return None
    return side.iloc[(side["strike"] - target_strike).abs().argmin()]


def select_by_otm_pct(chain: pd.DataFrame, expiry: pd.Timestamp, forward: float,
                      otm_pct: float) -> tuple[pd.Series, pd.Series] | None:
    """Pick the call and put nearest `otm_pct` away from the forward."""
    legs = chain[chain["expiry"] == expiry]
    call = _pick_leg(legs, forward * (1 + otm_pct), "CE")
    put = _pick_leg(legs, forward * (1 - otm_pct), "PE")
    if call is None or put is None:
        return None
    return call, put


def select_by_delta(chain: pd.DataFrame, expiry: pd.Timestamp, forward: float,
                    target_delta: float, T: float) -> tuple[pd.Series, pd.Series] | None:
    """Pick the call and put whose absolute delta is nearest `target_delta`.

    Delta needs a volatility, so each candidate's implied vol is solved first;
    strikes whose quotes admit none are skipped.
    """
    legs = chain[chain["expiry"] == expiry].copy()
    if legs.empty:
        return None

    legs["iv"] = [
        implied_vol(r.close, forward, r.strike, T, r.option_type)
        for r in legs.itertuples()
    ]
    legs = legs[legs["iv"].notna()]
    if legs.empty:
        return None

    legs["delta"] = [
        black76_delta(forward, r.strike, T, r.iv, r.option_type)
        for r in legs.itertuples()
    ]

    picks = []
    for side in ("CE", "PE"):
        s = legs[legs["option_type"] == side]
        if s.empty:
            return None
        picks.append(s.iloc[(s["delta"].abs() - target_delta).abs().argmin()])
    return picks[0], picks[1]


def build_trades(
    expiries: list[pd.Timestamp],
    symbol: str = "NIFTY",
    otm_pct: float = 0.02,
    target_delta: float | None = None,
    max_days_held: int = 10,
    session: requests.Session | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """One short-strangle trade per consecutive pair of expiries.

    Entry is the close of expiry `i`, selling the options expiring at `i + 1`.
    Set `target_delta` to select strikes by delta rather than distance from the
    forward.

    `max_days_held` keeps the horizon homogeneous: Nifty weeklies only began in
    February 2019, so the first gap in the sample is a month, not a week.

    Returns index points per unit, not per lot, plus the day's lot size so
    rupee costs can be applied later.
    """
    sess = session or requests.Session()
    rows = []

    for entry_date, expiry in zip(expiries[:-1], expiries[1:]):
        if (expiry - entry_date).days > max_days_held:
            continue

        entry_chain = option_chain(entry_date, symbol, session=sess)
        if entry_chain.empty:
            continue

        forward = forward_from_parity(entry_chain, expiry)
        if not pd.notna(forward):
            continue

        T = max((expiry - entry_date).days, 0) / DAYS_PER_YEAR
        if target_delta is not None:
            picked = select_by_delta(entry_chain, expiry, forward, target_delta, T)
        else:
            picked = select_by_otm_pct(entry_chain, expiry, forward, otm_pct)
        if picked is None:
            continue
        call, put = picked

        exit_chain = option_chain(expiry, symbol, session=sess)
        if exit_chain.empty:
            continue

        level = settlement_level(exit_chain, expiry)
        if not pd.notna(level):
            continue

        # Cash settled, so each leg pays intrinsic against the settlement
        # level. Out-of-the-money options still print a few paise at the
        # expiry-day close but settle at zero.
        exits = {
            "call": intrinsic(level, call["strike"], "CE"),
            "put": intrinsic(level, put["strike"], "PE"),
        }

        lot = infer_lot_size(entry_chain, forward)

        rows.append({
            "entry_date": entry_date,
            "expiry": expiry,
            "days_held": (expiry - entry_date).days,
            "forward_entry": forward,
            "settle_level": level,
            "call_strike": call["strike"],
            "put_strike": put["strike"],
            "call_entry": call["close"],
            "put_entry": put["close"],
            "call_exit": exits["call"],
            "put_exit": exits["put"],
            "call_iv": implied_vol(call["close"], forward, call["strike"], T, "CE"),
            "put_iv": implied_vol(put["close"], forward, put["strike"], T, "PE"),
            "lot_size": lot,
        })

        if verbose:
            print(f"  {entry_date.date()} -> {expiry.date()}  F={forward:8.1f}  "
                  f"{call['strike']:.0f}CE/{put['strike']:.0f}PE  "
                  f"premium={call['close'] + put['close']:7.2f}")

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades

    trades["premium"] = trades["call_entry"] + trades["put_entry"]
    trades["payout"] = trades["call_exit"] + trades["put_exit"]
    # Short both legs: collect the premium, pay the settlement value.
    trades["gross_pnl"] = trades["premium"] - trades["payout"]
    trades["move_pct"] = trades["settle_level"] / trades["forward_entry"] - 1.0
    return trades
