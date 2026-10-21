"""Black-76 pricing and an implied-volatility solver.

Black-76 takes a forward rather than spot, which avoids needing a dividend
yield. `implied_vol` is Newton-Raphson with a bisection fallback, since Newton
diverges on far out-of-the-money options where vega is tiny.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

MIN_T = 1e-6


def black76_price(F: float, K: float, T: float, sigma: float, option_type: str,
                  r: float = 0.0) -> float:
    """Price of a European option on a futures price F, discounted at r."""
    if T <= MIN_T or sigma <= 0:
        return intrinsic(F, K, option_type)

    sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / sqrt_t
    d2 = d1 - sqrt_t
    disc = np.exp(-r * T)

    if option_type == "CE":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def intrinsic(F: float, K: float, option_type: str) -> float:
    return max(F - K, 0.0) if option_type == "CE" else max(K - F, 0.0)


def black76_vega(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Sensitivity of price to a 1.0 change in sigma, not per 1%."""
    if T <= MIN_T or sigma <= 0:
        return 0.0
    sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / sqrt_t
    return np.exp(-r * T) * F * norm.pdf(d1) * np.sqrt(T)


def black76_delta(F: float, K: float, T: float, sigma: float, option_type: str,
                  r: float = 0.0) -> float:
    """Delta with respect to the futures price. Calls positive, puts negative."""
    if T <= MIN_T or sigma <= 0:
        itm = (F > K) if option_type == "CE" else (F < K)
        return (1.0 if option_type == "CE" else -1.0) * (1.0 if itm else 0.0)

    sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / sqrt_t
    disc = np.exp(-r * T)
    return disc * (norm.cdf(d1) if option_type == "CE" else norm.cdf(d1) - 1.0)


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    option_type: str,
    r: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> float:
    """Volatility that reproduces `price`, or NaN if there isn't one.

    Quotes below intrinsic or above the no-arbitrage ceiling have no solution
    and return NaN. Stale bhavcopy closes do produce them, and clamping instead
    would invent data.
    """
    if T <= MIN_T or price <= 0 or F <= 0 or K <= 0:
        return float("nan")

    floor = intrinsic(F, K, option_type) * np.exp(-r * T)
    ceiling = (F if option_type == "CE" else K) * np.exp(-r * T)
    if price < floor - tol or price > ceiling:
        return float("nan")

    sigma = 0.25
    for _ in range(max_iter):
        diff = black76_price(F, K, T, sigma, option_type, r) - price
        if abs(diff) < tol:
            return sigma
        vega = black76_vega(F, K, T, sigma, r)
        if vega < 1e-10:
            break
        step = diff / vega
        sigma -= step
        if sigma <= 0 or sigma > 10:
            break
    else:
        return sigma

    # Newton stalled or left the valid range; bisect instead.
    lo, hi = 1e-4, 10.0
    if black76_price(F, K, T, hi, option_type, r) < price:
        return float("nan")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if black76_price(F, K, T, mid, option_type, r) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
