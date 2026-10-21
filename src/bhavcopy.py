"""Download and normalize NSE's daily F&O bhavcopy.

The archive layout changed in 2024, so two URL patterns and two column mappings
are handled and normalized to the schema in `COLUMNS`. Files are cached under
data/raw/.

`volume` counts contracts in the legacy files and units in the newer ones, so
it is only used as a liquidity filter, never compared across the boundary.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

BASE = "https://nsearchives.nseindia.com"
HEADERS = {
    # NSE's archive rejects requests without a browser-ish User-Agent.
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

# The UDiFF format starts mid-2024; before that only the legacy layout exists.
UDIFF_FROM = pd.Timestamp("2024-07-01")

COLUMNS = [
    "trade_date", "symbol", "instrument", "expiry", "strike", "option_type",
    "open", "high", "low", "close", "settle", "oi", "volume", "turnover",
    "lot_size", "underlying",
]

# Lot sizes NSE has used for index options; `infer_lot_size` snaps to these.
LOT_CANDIDATES = (15, 20, 25, 40, 50, 65, 75, 100)

_LEGACY_MAP = {
    "SYMBOL": "symbol",
    "INSTRUMENT": "instrument",
    "EXPIRY_DT": "expiry",
    "STRIKE_PR": "strike",
    "OPTION_TYP": "option_type",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "SETTLE_PR": "settle",
    "OPEN_INT": "oi",
    "CONTRACTS": "volume",
    "VAL_INLAKH": "turnover",
}

_UDIFF_MAP = {
    "TckrSymb": "symbol",
    "FinInstrmTp": "instrument",
    "XpryDt": "expiry",
    "StrkPric": "strike",
    "OptnTp": "option_type",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "SttlmPric": "settle",
    "OpnIntrst": "oi",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover",
    "NewBrdLotQty": "lot_size",
    "UndrlygPric": "underlying",
}

_INSTRUMENT_MAP = {"IDO": "OPTIDX", "IDF": "FUTIDX", "STO": "OPTSTK", "STF": "FUTSTK"}


def _url(date: pd.Timestamp) -> str:
    if date >= UDIFF_FROM:
        return f"{BASE}/content/fo/BhavCopy_NSE_FO_0_0_0_{date:%Y%m%d}_F_0000.csv.zip"
    mon = f"{date:%b}".upper()
    return (f"{BASE}/content/historical/DERIVATIVES/{date:%Y}/{mon}/"
            f"fo{date:%d}{mon}{date:%Y}bhav.csv.zip")


def _cache_path(date: pd.Timestamp) -> Path:
    return RAW_DIR / f"fo_{date:%Y%m%d}.csv.gz"


def fetch_raw(
    date: pd.Timestamp,
    session: requests.Session | None = None,
    pause: float = 0.4,
) -> pd.DataFrame | None:
    """One day's bhavcopy, or None when NSE has no file (holiday or weekend)."""
    date = pd.Timestamp(date).normalize()
    cache = _cache_path(date)
    if cache.exists():
        return pd.read_csv(cache)

    sess = session or requests.Session()
    resp = sess.get(_url(date), headers=HEADERS, timeout=60)
    time.sleep(pause)

    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(zf.open(name))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, compression="gzip")
    return df


def normalize(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Map either bhavcopy layout onto the common `COLUMNS` schema."""
    legacy = "SYMBOL" in df.columns
    out = df.rename(columns=_LEGACY_MAP if legacy else _UDIFF_MAP).copy()

    for col in COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["trade_date"] = pd.Timestamp(date).normalize()
    # Legacy dates look like 05-Jan-2023; UDiFF uses ISO.
    out["expiry"] = pd.to_datetime(out["expiry"], format="mixed", dayfirst=legacy)

    if not legacy:
        out["instrument"] = out["instrument"].map(_INSTRUMENT_MAP).fillna(out["instrument"])

    numeric = ["strike", "open", "high", "low", "close", "settle", "oi", "volume",
               "turnover", "lot_size", "underlying"]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if legacy:
        # Legacy turnover is quoted in lakhs of rupees; UDiFF already uses rupees.
        out["turnover"] = out["turnover"] * 1e5

    out["option_type"] = out["option_type"].astype(str).str.strip().str.upper()
    out.loc[~out["option_type"].isin(["CE", "PE"]), "option_type"] = pd.NA

    return out[COLUMNS]


def load_day(
    date: pd.Timestamp,
    symbol: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame | None:
    """Normalized bhavcopy for one date, optionally filtered to one symbol."""
    raw = fetch_raw(date, session=session)
    if raw is None:
        return None
    df = normalize(raw, date)
    if symbol is not None:
        df = df[df["symbol"] == symbol]
    return df.reset_index(drop=True)


def infer_lot_size(day: pd.DataFrame, reference_price: float) -> int | None:
    """Contract multiplier for one day's option rows.

    Legacy files omit it but report turnover as notional, so dividing turnover
    by contracts and the index level recovers it. The median over liquid
    contracts is snapped to `LOT_CANDIDATES` to absorb the error from using a
    closing price as the day's average level.
    """
    stated = day["lot_size"].dropna()
    if not stated.empty:
        return int(stated.mode().iloc[0])

    liquid = day[(day["volume"] > 1000) & day["turnover"].notna()]
    if liquid.empty or not reference_price:
        return None

    implied = (liquid["turnover"] / (liquid["volume"] * reference_price)).median()
    if not pd.notna(implied):
        return None
    return min(LOT_CANDIDATES, key=lambda c: abs(c - implied))


def expiries_on(date: pd.Timestamp, symbol: str = "NIFTY", **kw) -> list[pd.Timestamp]:
    """Option expiry dates listed in `date`'s file, ascending.

    Any single file advertises every live expiry. Reading them beats assuming a
    weekday: Nifty's weekly expiry moved from Thursday to Wednesday to Tuesday
    over the sample.
    """
    df = load_day(date, symbol=symbol, **kw)
    if df is None:
        return []
    opts = df[df["instrument"] == "OPTIDX"]
    return sorted(pd.to_datetime(opts["expiry"].dropna().unique()))


def discover_expiries(
    start: str,
    end: str,
    symbol: str = "NIFTY",
    session: requests.Session | None = None,
) -> list[pd.Timestamp]:
    """Expiry calendar between two dates.

    One file reveals a whole month's expiries, so this probes a few days per
    month rather than downloading every day.
    """
    sess = session or requests.Session()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    found: set[pd.Timestamp] = set()

    for month_start in pd.date_range(start_ts, end_ts, freq="MS"):
        for offset in range(0, 10):
            probe = month_start + pd.Timedelta(days=offset)
            if probe > end_ts:
                break
            got = expiries_on(probe, symbol=symbol, session=sess)
            if got:
                found.update(got)
                break

    return sorted(d for d in found if start_ts <= d <= end_ts)


def download_dates(
    dates: list[pd.Timestamp],
    session: requests.Session | None = None,
) -> list[pd.Timestamp]:
    """Cache the bhavcopy for each date; return those NSE actually served."""
    sess = session or requests.Session()
    ok = []
    for d in dates:
        if fetch_raw(d, session=sess) is not None:
            ok.append(pd.Timestamp(d).normalize())
    return ok


if __name__ == "__main__":
    import sys

    start = sys.argv[1] if len(sys.argv) > 1 else "2019-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-10-01"

    print(f"Discovering NIFTY option expiries {start} -> {end} ...")
    exp = discover_expiries(start, end)
    print(f"Found {len(exp)} expiries. First: {exp[0].date()}  Last: {exp[-1].date()}")

    print(f"\nDownloading bhavcopy for each expiry date ...")
    got = download_dates(exp)
    print(f"Cached {len(got)} of {len(exp)} files under {RAW_DIR}")
