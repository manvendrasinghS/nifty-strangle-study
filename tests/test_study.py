"""Correctness tests on synthetic data: no network, no cached files needed.

The failures that matter in an options backtest are quiet ones — a settlement
value read from the wrong column, a stale quote treated as a fill, a lot size
guessed wrong. Each has a case here with an answer worked out by hand.

Run from the project root:   PYTHONPATH=src pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from bhavcopy import LOT_CANDIDATES, infer_lot_size, normalize
from blackscholes import black76_delta, black76_price, implied_vol, intrinsic
from costs import STT_HIKE_DATE, CostModel, apply_costs
from stats import skew_by_unit, summarize
from strangle import forward_from_parity, option_chain, select_by_otm_pct, settlement_level


def _chain(expiry="2023-01-12", trade_date="2023-01-05", level=18000.0,
           strikes=range(17000, 19001, 100), traded=True):
    """A synthetic chain priced by put-call parity around `level`."""
    rows = []
    for k in strikes:
        # Intrinsic plus a token time value, so parity holds exactly.
        time_value = 50.0
        rows.append({"trade_date": pd.Timestamp(trade_date), "symbol": "NIFTY",
                     "instrument": "OPTIDX", "expiry": pd.Timestamp(expiry),
                     "strike": float(k), "option_type": "CE",
                     "close": max(level - k, 0) + time_value, "settle": np.nan,
                     "oi": 1000, "volume": 100 if traded else 0, "turnover": 1e7,
                     "lot_size": 50, "underlying": level,
                     "open": 0.0, "high": 0.0, "low": 0.0})
        rows.append({**rows[-1], "option_type": "PE",
                     "close": max(k - level, 0) + time_value})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pricing


def test_black76_put_call_parity():
    """C - P must equal F - K, discounted. The pricer is wrong if it doesn't."""
    F, K, T, sigma = 18000.0, 17800.0, 7 / 365, 0.14
    call = black76_price(F, K, T, sigma, "CE")
    put = black76_price(F, K, T, sigma, "PE")
    assert call - put == pytest.approx(F - K, abs=1e-6)


def test_black76_expired_is_intrinsic():
    assert black76_price(18000, 17500, 0.0, 0.2, "CE") == pytest.approx(500.0)
    assert black76_price(18000, 17500, 0.0, 0.2, "PE") == pytest.approx(0.0)


def test_implied_vol_round_trip():
    """A price generated at a known sigma must solve back to that sigma."""
    F, K, T = 18000.0, 18400.0, 7 / 365
    for sigma in (0.08, 0.15, 0.40, 0.90):
        price = black76_price(F, K, T, sigma, "CE")
        assert implied_vol(price, F, K, T, "CE") == pytest.approx(sigma, abs=1e-4)


def test_implied_vol_rejects_impossible_prices():
    """Below intrinsic or above the ceiling there is no solution: expect NaN."""
    F, K, T = 18000.0, 17000.0, 7 / 365
    assert np.isnan(implied_vol(500.0, F, K, T, "CE"))      # below intrinsic (1000)
    assert np.isnan(implied_vol(19000.0, F, K, T, "CE"))    # above F
    assert np.isnan(implied_vol(0.0, F, K, T, "CE"))


def test_delta_signs_and_bounds():
    F, T, sigma = 18000.0, 7 / 365, 0.15
    assert 0 < black76_delta(F, 18400, T, sigma, "CE") < 1
    assert -1 < black76_delta(F, 17600, T, sigma, "PE") < 0
    # Deep in the money, a call's delta approaches one.
    assert black76_delta(F, 10000, T, sigma, "CE") == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------- chain handling


def test_forward_from_parity_recovers_level():
    """Parity must recover the level the synthetic chain was built around."""
    chain = _chain(level=18123.45)
    assert forward_from_parity(chain, pd.Timestamp("2023-01-12")) == pytest.approx(18123.45)


def test_forward_needs_both_sides():
    """With calls only there is no parity estimate; expect NaN, not a guess."""
    chain = _chain()
    calls_only = chain[chain["option_type"] == "CE"]
    assert np.isnan(forward_from_parity(calls_only, pd.Timestamp("2023-01-12")))


def test_settlement_level_prefers_official_field():
    """The index settlement value NSE writes into every expiring option's
    settlement field must win over the parity estimate."""
    expiry = pd.Timestamp("2023-01-12")
    chain = _chain(expiry=expiry, trade_date=expiry, level=18000.0)
    chain["settle"] = 17858.2
    assert settlement_level(chain, expiry) == pytest.approx(17858.2)


def test_settlement_level_falls_back_to_parity():
    expiry = pd.Timestamp("2023-01-12")
    chain = _chain(expiry=expiry, trade_date=expiry, level=17950.0)
    chain["settle"] = np.nan
    assert settlement_level(chain, expiry) == pytest.approx(17950.0)


def test_select_by_otm_pct_brackets_the_forward():
    """The call must sit above the forward and the put below, each near target."""
    chain = _chain(level=18000.0)
    call, put = select_by_otm_pct(chain, pd.Timestamp("2023-01-12"), 18000.0, 0.02)
    assert call["option_type"] == "CE" and put["option_type"] == "PE"
    assert call["strike"] == 18360.0 or call["strike"] == 18400.0
    assert put["strike"] == 17640.0 or put["strike"] == 17600.0
    assert call["strike"] > 18000.0 > put["strike"]


def test_option_chain_drops_untraded_strikes(monkeypatch):
    """Zero-volume rows carry a stale previous close and must not be tradable."""
    import strangle

    chain = _chain()
    chain.loc[chain["strike"] == 18400.0, "volume"] = 0
    monkeypatch.setattr(strangle, "load_day", lambda *a, **k: chain)

    kept = option_chain(pd.Timestamp("2023-01-05"))
    assert 18400.0 not in set(kept["strike"])
    assert 18300.0 in set(kept["strike"])


# ---------------------------------------------------------------- bhavcopy


def test_normalize_handles_both_formats():
    """Legacy and UDiFF layouts must land on the same schema and dtypes."""
    legacy = pd.DataFrame({
        "INSTRUMENT": ["OPTIDX"], "SYMBOL": ["NIFTY"], "EXPIRY_DT": ["12-Jan-2023"],
        "STRIKE_PR": [18000.0], "OPTION_TYP": ["CE"], "OPEN": [10.0], "HIGH": [12.0],
        "LOW": [9.0], "CLOSE": [11.0], "SETTLE_PR": [11.0], "CONTRACTS": [500],
        "VAL_INLAKH": [100.0], "OPEN_INT": [1000], "CHG_IN_OI": [0],
        "TIMESTAMP": ["05-JAN-2023"],
    })
    udiff = pd.DataFrame({
        "TckrSymb": ["NIFTY"], "FinInstrmTp": ["IDO"], "XpryDt": ["2023-01-12"],
        "StrkPric": [18000.0], "OptnTp": ["CE"], "OpnPric": [10.0], "HghPric": [12.0],
        "LwPric": [9.0], "ClsPric": [11.0], "SttlmPric": [11.0],
        "TtlTradgVol": [500], "TtlTrfVal": [1e7], "OpnIntrst": [1000],
        "NewBrdLotQty": [50], "UndrlygPric": [18050.0],
    })

    a = normalize(legacy, pd.Timestamp("2023-01-05"))
    b = normalize(udiff, pd.Timestamp("2023-01-05"))

    assert list(a.columns) == list(b.columns)
    for df in (a, b):
        assert df["instrument"].iloc[0] == "OPTIDX"
        assert df["expiry"].iloc[0] == pd.Timestamp("2023-01-12")
        assert df["close"].iloc[0] == 11.0
    # Legacy turnover is in lakhs and must be scaled to rupees.
    assert a["turnover"].iloc[0] == pytest.approx(1e7)


def test_infer_lot_size_from_notional_turnover():
    """Legacy files omit lot size, so it is recovered from notional turnover."""
    level = 18000.0
    lot = 50
    day = pd.DataFrame({
        "lot_size": [np.nan] * 3,
        "volume": [10_000, 20_000, 30_000],
        "turnover": [10_000 * lot * level, 20_000 * lot * level, 30_000 * lot * level],
    })
    assert infer_lot_size(day, level) == lot


def test_infer_lot_size_prefers_stated_value():
    day = pd.DataFrame({"lot_size": [25, 25], "volume": [1e6, 1e6],
                        "turnover": [1e9, 1e9]})
    assert infer_lot_size(day, 24000.0) == 25


def test_infer_lot_size_snaps_to_real_lot_sizes():
    """A closing price is not the day's average level, so the noisy estimate is
    snapped to a lot size NSE actually used."""
    level = 18000.0
    day = pd.DataFrame({
        "lot_size": [np.nan],
        "volume": [10_000],
        "turnover": [10_000 * 51.4 * level],   # noisy 50
    })
    assert infer_lot_size(day, level) == 50
    assert 50 in LOT_CANDIDATES


# ---------------------------------------------------------------- costs


def _trades(premium=50.0, gross=10.0, n=4, date="2023-01-05"):
    return pd.DataFrame({
        "entry_date": [pd.Timestamp(date)] * n,
        "premium": [premium] * n,
        "gross_pnl": [gross] * n,
        "lot_size": [50] * n,
        "call_entry": [premium / 2] * n, "put_entry": [premium / 2] * n,
        "call_exit": [0.0] * n, "put_exit": [0.0] * n,
    })


def test_costs_reduce_pnl_and_are_positive():
    priced = apply_costs(_trades(), CostModel())
    assert (priced["cost_points"] > 0).all()
    assert (priced["net_pnl"] < priced["gross_pnl"]).all()


def test_cost_components_by_hand():
    """Worked example: premium 50 pts, lot 50, so premium value is Rs 2,500.

    brokerage        2 x 20                       = 40.000
    STT              2500 x 0.000625              =  1.5625
    exchange         2500 x 0.0005                =  1.250
    SEBI             2500 x 0.000001              =  0.0025
    GST              0.18 x (40 + 1.250 + 0.0025) =  7.42545
    total rupees                                  = 50.24045
    per unit         50.24045 / 50                =  1.0048
    slippage         0.25 x 2 legs                =  0.500
    total points                                  =  1.5048
    """
    model = CostModel(slippage_points=0.25)
    got = model.cost_points(premium=50.0, trade_date=pd.Timestamp("2023-01-05"),
                            lot_size=50)
    assert got == pytest.approx(1.5048, abs=1e-4)


def test_stt_hike_applies_from_october_2024():
    """STT on selling options rose from 0.0625% to 0.1% on 1 Oct 2024."""
    model = CostModel()
    before = model.stt_rate(STT_HIKE_DATE - pd.Timedelta(days=1))
    after = model.stt_rate(STT_HIKE_DATE)
    assert before == 0.000625
    assert after == 0.001
    cheap = model.cost_points(100.0, STT_HIKE_DATE - pd.Timedelta(days=1), 50)
    dear = model.cost_points(100.0, STT_HIKE_DATE, 50)
    assert dear > cheap


def test_cost_scales_down_with_bigger_lot():
    """Flat brokerage is diluted per unit as the lot size grows."""
    model = CostModel(slippage_points=0.0)
    assert model.cost_points(50.0, pd.Timestamp("2023-01-05"), 25) > \
           model.cost_points(50.0, pd.Timestamp("2023-01-05"), 75)


# ---------------------------------------------------------------- statistics


def test_summarize_basic_fields():
    t = apply_costs(_trades(gross=10.0, n=5), CostModel())
    t["gross_pnl"] = [8.0, 9.0, 10.0, 11.0, 12.0]
    s = summarize(t, "gross_pnl")
    assert s["trades"] == 5
    assert s["mean_points"] == pytest.approx(10.0)
    assert s["win_rate"] == 1.0
    assert s["worst"] == 8.0 and s["best"] == 12.0


def test_skew_by_unit_reports_both_units():
    """Position-level skew must see a breach as one loss, not two half-losses.

    No ordering between the two units is asserted: which is more negative
    depends on the payoff structure.
    """
    n = 20
    premium = 50.0
    call_exit = [0.0] * n
    call_exit[-1] = 400.0            # the breach

    t = pd.DataFrame({
        "entry_date": pd.date_range("2023-01-05", periods=n, freq="7D"),
        "premium": [premium] * n,
        "call_entry": [premium / 2] * n, "put_entry": [premium / 2] * n,
        "call_exit": call_exit, "put_exit": [0.0] * n,
        "lot_size": [50] * n,
    })
    t["gross_pnl"] = t["premium"] - (t["call_exit"] + t["put_exit"])

    units = skew_by_unit(t)
    assert units["n_legs"] == 2 * units["n_positions"] == 40
    assert units["skew_per_position"] < 0
    assert units["skew_per_leg"] < 0
    # The breach is one position-level loss of -350, not two of -175.
    assert t["gross_pnl"].min() == pytest.approx(-350.0)


def test_intrinsic_matches_cash_settlement():
    """A short leg pays intrinsic against the settlement level, never the
    expiry-day traded price."""
    assert intrinsic(17858.2, 18400.0, "CE") == 0.0
    assert intrinsic(17858.2, 17650.0, "PE") == 0.0
    assert intrinsic(17511.25, 17700.0, "PE") == pytest.approx(188.75)
