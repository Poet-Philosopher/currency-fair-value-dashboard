"""
data_engine.py
==============
Data ingestion and relative-PPP engine for the Currency Fair Value Dashboard.

Pipeline
--------
1. fetch_spot()     daily spot FX from Yahoo Finance (yfinance)
2. fetch_cpi()      monthly (occasionally quarterly) CPI index levels from FRED
3. compute_ppp()    align the low-frequency CPI onto the daily spot index, then
                    build the relative-PPP fair-value line and % deviation
4. build_dataset()  orchestrates 1-3 for a pair / base year / date range
5. compute_signal() turns the latest deviation into a human-readable signal

Quote convention
----------------
A pair "BASE/QUOTE" is quoted as *units of QUOTE per 1 unit of BASE*.
EUR/USD = 1.10 means 1 EUR buys 1.10 USD. This matches Yahoo's "EURUSD=X".

    spot up   -> BASE currency strengthens vs QUOTE
    deviation > 0 -> spot is ABOVE fair value -> BASE is OVERVALUED
    deviation < 0 -> spot is BELOW fair value -> BASE is UNDERVALUED

The module has no Streamlit dependency, so it can be used from a notebook or
tested on its own:  `python data_engine.py --check`
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("data_engine")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# |deviation| below this is reported as "fairly valued".
FAIR_BAND_PCT = 2.0
# |deviation| above this is reported as a strong misvaluation (also the band
# width drawn on the charts).
STRONG_PCT = 10.0
# Yahoo Finance FX history starts around 2003, so the first complete year is 2004.
MIN_BASE_YEAR = 2004

# A CPI series older than this (relative to the end of the window) counts as stale.
STALE_CPI_DAYS = 200
# Gaps shorter than this are normal publication lag and are simply held flat.
LAG_TOLERANCE_DAYS = 45
# When extending stale CPI with its recent trend, stop after this long and hold flat.
MAX_EXTRAPOLATION_DAYS = 3 * 365

# FRED CPI *index level* series, in preference order. fetch_cpi() tries every
# candidate and keeps the one with the freshest observation that also has data
# in the chosen base year (only index levels work, not growth rates).
#
# Known issue (checked Sep 2026): the OECD-sourced "...MINMEI" series on FRED
# stopped updating in 2024-25, so only the US and euro-area series are current.
# When the best candidate is stale, fetch_cpi() searches FRED for a fresher
# series (needs an API key), and compute_ppp() extends whatever is left using
# the trailing inflation trend. `python data_engine.py --discover` shows what
# the search finds for each currency.
CPI_SERIES: dict[str, list[str]] = {
    "USD": ["CPIAUCSL", "CPIAUCNS"],
    "EUR": ["CP0000EZ19M086NEST"],
    "GBP": ["GBRCPIALLMINMEI"],
    "JPY": ["JPNCPIALLMINMEI"],
    "CHF": ["CP0000CHM086NEST", "CHECPIALLMINMEI"],
    "CAD": ["CANCPIALLMINMEI"],
    "AUD": ["AUSCPIALLQINMEI"],   # quarterly
    "NZD": ["NZLCPIALLQINMEI"],   # quarterly
    "INR": ["INDCPIALLMINMEI"],
    "CNY": ["CHNCPIALLMINMEI"],
    "MXN": ["MEXCPIALLMINMEI"],
    "SEK": ["CP0000SEM086NEST", "SWECPIALLMINMEI"],
    "NOK": ["CP0000NOM086NEST", "NORCPIALLMINMEI"],
}

# Words used to find each country's CPI series in FRED's title search.
COUNTRY_SEARCH_NAMES: dict[str, str] = {
    "USD": "U.S.", "EUR": "Euro Area", "GBP": "United Kingdom", "JPY": "Japan",
    "CHF": "Switzerland", "CAD": "Canada", "AUD": "Australia", "NZD": "New Zealand",
    "INR": "India", "CNY": "China", "MXN": "Mexico", "SEK": "Sweden", "NOK": "Norway",
}

CURRENCY_NAMES: dict[str, str] = {
    "USD": "US dollar", "EUR": "Euro", "GBP": "British pound", "JPY": "Japanese yen",
    "CHF": "Swiss franc", "CAD": "Canadian dollar", "AUD": "Australian dollar",
    "NZD": "New Zealand dollar", "INR": "Indian rupee", "CNY": "Chinese yuan",
    "MXN": "Mexican peso", "SEK": "Swedish krona", "NOK": "Norwegian krone",
}

PRESET_PAIRS: list[str] = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "NZD/USD", "USD/CAD",
    "USD/CHF", "USD/INR", "EUR/GBP", "EUR/JPY", "USD/CNY", "USD/MXN",
]

FILL_METHODS = ("ffill", "interpolate")


_SECRETS_PATH = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"


class DataUnavailableError(RuntimeError):
    """Raised when spot or CPI data cannot be fetched or is unusable."""


def _redact(text) -> str:
    """Strip API keys from error text. requests puts the full URL, key included, in its errors."""
    return re.sub(r"(api_key=)[A-Za-z0-9]+", r"\1***", str(text))


def _get_fred_api_key() -> Optional[str]:
    """FRED API key from the FRED_API_KEY env var, else .streamlit/secrets.toml.

    Reading the TOML file directly means the same key works for the Streamlit
    app and for the CLI (`python data_engine.py --check`).
    """
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        return key
    try:
        import tomllib  # Python 3.11+

        with open(_SECRETS_PATH, "rb") as fh:
            key = str(tomllib.load(fh).get("FRED_API_KEY", "")).strip()
        return key or None
    except (ImportError, OSError, ValueError):
        return None


def fred_key_source() -> Optional[str]:
    """Where the FRED API key was found (for display), or None."""
    if os.environ.get("FRED_API_KEY", "").strip():
        return "FRED_API_KEY environment variable"
    return ".streamlit/secrets.toml" if _get_fred_api_key() else None


# ----------------------------------------------------------------------------
# Pair helpers
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PairSpec:
    base: str
    quote: str

    @property
    def label(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def ticker(self) -> str:
        # Yahoo Finance FX tickers: EURUSD=X, USDJPY=X, ...
        return f"{self.base}{self.quote}=X"


def parse_pair(label: str) -> PairSpec:
    """'EUR/USD' -> PairSpec('EUR', 'USD'), validating that CPI data is mapped."""
    try:
        base, quote = label.replace(" ", "").upper().split("/")
    except ValueError as exc:
        raise ValueError(f"Pair must look like 'EUR/USD', got {label!r}") from exc
    if base == quote:
        raise ValueError("Base and quote currency must differ.")
    for ccy in (base, quote):
        if ccy not in CPI_SERIES:
            raise ValueError(f"No CPI series configured for {ccy}. Add it to CPI_SERIES.")
    return PairSpec(base, quote)


# ----------------------------------------------------------------------------
# Spot FX (daily) from yfinance
# ----------------------------------------------------------------------------
def _extract_close(raw: Optional[pd.DataFrame]) -> pd.Series:
    """Pull a clean, tz-naive, daily 'Close' series out of a yfinance frame.

    Recent yfinance versions return MultiIndex columns (field, ticker) even for
    a single ticker, older ones return flat columns. `raw["Close"]` works for
    both; in the MultiIndex case it yields a one-column DataFrame.
    """
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float, name="spot")
    if "Close" not in raw.columns.get_level_values(0):
        return pd.Series(dtype=float, name="spot")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[close > 0]  # drop bad ticks (FX rates are strictly positive)

    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.name = "spot"
    return close


def fetch_spot(ticker: str, start: date, end: date) -> pd.Series:
    """Daily FX close for `ticker` between `start` and `end` (inclusive)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailableError("yfinance is not installed (pip install yfinance).") from exc

    end_exclusive = end + timedelta(days=1)  # yfinance treats `end` as exclusive
    errors: list[str] = []
    close = pd.Series(dtype=float)

    try:
        raw = yf.download(
            ticker, start=start.isoformat(), end=end_exclusive.isoformat(),
            interval="1d", auto_adjust=False, progress=False, threads=False,
        )
        close = _extract_close(raw)
    except Exception as exc:  # network errors, rate limits, parsing issues
        errors.append(f"download: {exc}")

    if close.empty:  # second attempt through a different yfinance code path
        try:
            raw = yf.Ticker(ticker).history(
                start=start.isoformat(), end=end_exclusive.isoformat(),
                interval="1d", auto_adjust=False,
            )
            close = _extract_close(raw)
        except Exception as exc:
            errors.append(f"history: {exc}")

    if close.empty:
        detail = f" ({'; '.join(errors)})" if errors else ""
        raise DataUnavailableError(
            f"Yahoo Finance returned no data for {ticker} between {start} and {end}{detail}."
        )
    return close


# ----------------------------------------------------------------------------
# CPI (monthly / quarterly) from FRED
# ----------------------------------------------------------------------------
def _clean_fred_series(s: pd.Series, series_id: str) -> pd.Series:
    """Validate that `s` is a positive index-level series and normalise its index."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        raise DataUnavailableError(f"FRED series {series_id} returned no observations.")
    if (s <= 0).any():
        # Growth-rate series (e.g. ...M659N) contain zeros/negatives. PPP needs price LEVELS.
        raise DataUnavailableError(
            f"FRED series {series_id} contains values <= 0, so it looks like a growth "
            "rate. Use a CPI index-level series instead."
        )
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = series_id
    return s


def _fred_via_rest(series_id: str, start: date, end: date, api_key: str) -> pd.Series:
    import requests

    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id, "api_key": api_key, "file_type": "json",
            "observation_start": start.isoformat(), "observation_end": end.isoformat(),
        },
        timeout=30,
    )
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    return pd.Series(
        {pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in obs},
        dtype=float,
    )


def _fred_via_datareader(series_id: str, start: date, end: date) -> pd.Series:
    from pandas_datareader import data as pdr

    df = pdr.DataReader(series_id, "fred", start, end)
    return df.iloc[:, 0]


def _fred_via_csv(series_id: str, start: date, end: date) -> pd.Series:
    """Direct download of FRED's public CSV endpoint (no key, no extra dependency)."""
    import requests

    resp = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id, "cosd": start.isoformat(), "coed": end.isoformat()},
        headers={"User-Agent": "Mozilla/5.0 (currency-fair-value-dashboard)"},
        timeout=30,
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    dates = pd.to_datetime(df.iloc[:, 0])
    return pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=dates)


def _fetch_fred_series(series_id: str, start: date, end: date) -> pd.Series:
    """Fetch one FRED series, trying (1) the official REST API if an API key is
    configured, (2) pandas_datareader, (3) FRED's public CSV endpoint."""
    attempts = []
    api_key = _get_fred_api_key()
    if api_key:
        attempts.append(("FRED API", lambda: _fred_via_rest(series_id, start, end, api_key)))
    attempts.append(("pandas_datareader", lambda: _fred_via_datareader(series_id, start, end)))
    attempts.append(("FRED CSV", lambda: _fred_via_csv(series_id, start, end)))

    errors: list[str] = []
    for name, fn in attempts:
        try:
            return _clean_fred_series(fn(), series_id)
        except DataUnavailableError:
            raise  # series exists but is unusable; another transport won't fix it
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {_redact(exc)}")
    raise DataUnavailableError(f"Could not download FRED series {series_id}. " + " | ".join(errors))


_EXCLUDE_WORDS = (
    "excluding", "less ", "core", "food", "energy", "housing", "shelter", "services", "goods",
    "clothing", "transport", "health", "education", "communication", "recreation", "alcohol",
    "tobacco", "furnish", "restaurant", "rent", "fuel", "utilities", "growth rate", "percent",
    "metropolitan", "county", "regional",
)


def _rank_discovered(rows: list[dict], country: str, base_year: int, end: date) -> list[dict]:
    """Keep FRED search hits that look like a headline, index-level, fresh CPI series."""
    cutoff = pd.Timestamp(end) - pd.Timedelta(days=STALE_CPI_DAYS)
    keep = []
    for r in rows:
        title = str(r.get("title", "")).lower()
        if country.lower() not in title or "all items" not in title:
            continue
        if any(w in title for w in _EXCLUDE_WORDS):
            continue
        if not str(r.get("units", "")).lower().startswith("index"):
            continue  # growth rates and percent changes are useless for PPP
        if r.get("frequency_short") not in ("M", "Q"):
            continue
        try:
            obs_start = pd.Timestamp(r["observation_start"])
            obs_end = pd.Timestamp(r["observation_end"])
        except (KeyError, ValueError):
            continue
        if obs_end < cutoff or obs_start > pd.Timestamp(base_year, 1, 1):
            continue
        keep.append(r)
    # Monthly before quarterly, then most popular first.
    keep.sort(key=lambda r: (r.get("frequency_short") != "M", -int(r.get("popularity") or 0)))
    return keep


def _search_fred(query: str, api_key: str) -> list[dict]:
    """One FRED series search (up to 1000 hits, most popular first), retried once."""
    import requests

    last_exc: Optional[Exception] = None
    for _ in range(2):
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/search",
                params={
                    "search_text": query, "api_key": api_key, "file_type": "json",
                    "limit": 1000, "order_by": "popularity", "sort_order": "desc",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("seriess", [])
        except Exception as exc:
            last_exc = exc
    log.warning("FRED series search failed for %r: %s", query, _redact(last_exc))
    return []


def _search_country_cpi(currency: str, api_key: str) -> list[dict]:
    """Search both title styles FRED uses: OECD/national 'Consumer Price Index ...' and
    Eurostat 'Harmonized Index of Consumer Prices ...'. A single query misses one of them,
    because FRED matches whole words and 'Prices' is not 'Price'."""
    country = COUNTRY_SEARCH_NAMES[currency]
    hits: dict[str, dict] = {}
    for query in (f"consumer price index {country}", f"harmonized index of consumer prices {country}"):
        for row in _search_fred(query, api_key):
            hits[row["id"]] = row
    return list(hits.values())


def discover_cpi_series(currency: str, base_year: int, end: date, max_results: int = 3) -> list[dict]:
    """Search FRED for fresh CPI index series for `currency`. Needs an API key."""
    api_key = _get_fred_api_key()
    if not api_key:
        return []
    rows = _search_country_cpi(currency, api_key)
    return _rank_discovered(rows, COUNTRY_SEARCH_NAMES[currency], base_year, end)[:max_results]


def _consider(sid, best, notes, start, end, base_year):
    """Fetch `sid` and return it if it is usable and fresher than `best`."""
    try:
        s = _fetch_fred_series(sid, start, end)
    except DataUnavailableError as exc:
        notes.append(str(exc))
        return best
    if not (s.index.year == base_year).any():
        notes.append(f"{sid} has no observations in {base_year}.")
        return best
    if best is None or s.index.max() > best[1].index.max():
        return (sid, s)
    return best


def fetch_cpi(
    currency: str, start: date, end: date, base_year: int, override_id: Optional[str] = None
) -> tuple[pd.Series, str]:
    """Return (CPI index series, series id used) for a currency's home economy.

    Every configured candidate is fetched and the freshest one that has data in
    `base_year` wins (ties go to the earlier candidate). If even the best one is
    stale, FRED's search is asked for something fresher. A user-supplied
    `override_id` skips both steps.
    """
    override = (override_id or "").strip()
    candidates = [override] if override else list(CPI_SERIES[currency])
    best: Optional[tuple[str, pd.Series]] = None
    notes: list[str] = []

    for sid in candidates:
        best = _consider(sid, best, notes, start, end, base_year)

    stale_before = pd.Timestamp(end) - pd.Timedelta(days=STALE_CPI_DAYS)
    if not override and (best is None or best[1].index.max() < stale_before):
        for row in discover_cpi_series(currency, base_year, end):
            if row["id"] not in candidates:
                best = _consider(row["id"], best, notes, start, end, base_year)

    if best is None:
        raise DataUnavailableError(
            f"No usable CPI series for {currency} (tried {', '.join(candidates)}). " + " ".join(notes)
        )
    return best[1], best[0]


def extend_cpi(cpi: pd.Series, through: pd.Timestamp, lookback_days: int = 365):
    """Extend a stale CPI series to `through` using its trailing inflation trend.

    Returns (series, info). `info` is None when nothing was added. The trend is
    the compound daily growth over the last `lookback_days` of published data:

        CPI(t) = CPI_last x exp(rate x days since last print)

    Monthly points are added until `through`, or for at most
    MAX_EXTRAPOLATION_DAYS, after which the level is held flat. This is an
    estimate, not data, and the UI says so.
    """
    last_date = cpi.index.max()
    gap_days = (through - last_date).days
    if gap_days <= LAG_TOLERANCE_DAYS:
        return cpi, None
    prior = cpi[cpi.index <= last_date - pd.Timedelta(days=lookback_days)]
    if prior.empty:
        return cpi, None  # not enough history to estimate a trend
    ref_date, ref_val = prior.index[-1], float(prior.iloc[-1])
    span = (last_date - ref_date).days
    daily_rate = float(np.log(float(cpi.iloc[-1]) / ref_val) / span)

    horizon = min(gap_days, MAX_EXTRAPOLATION_DAYS)
    new_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1), last_date + pd.Timedelta(days=horizon), freq="MS"
    )
    if len(new_dates) == 0:
        return cpi, None
    days_ahead = np.asarray((new_dates - last_date).days, dtype=float)
    ext = pd.Series(float(cpi.iloc[-1]) * np.exp(daily_rate * days_ahead), index=new_dates, name=cpi.name)
    info = {
        "from": last_date,
        "months": int(len(new_dates)),
        "annual_rate_pct": float((np.exp(daily_rate * 365) - 1) * 100),
        "capped": bool(gap_days > MAX_EXTRAPOLATION_DAYS),
    }
    return pd.concat([cpi, ext]), info


# ----------------------------------------------------------------------------
# PPP maths + frequency alignment
# ----------------------------------------------------------------------------
def _rebase(cpi: pd.Series, base_year: int, label: str) -> tuple[pd.Series, int]:
    """Rebase a CPI series so its base-year average equals 1.0.

    CPI_t / mean(CPI in base year) removes the arbitrary index base (2010=100,
    2015=100, ...) so US and Eurozone indices become directly comparable.
    """
    in_year = cpi[cpi.index.year == base_year]
    if in_year.empty:
        raise DataUnavailableError(f"{label} CPI has no observations in {base_year}.")
    return cpi / in_year.mean(), int(len(in_year))


def _to_daily(series: pd.Series, daily_index: pd.DatetimeIndex, method: str) -> pd.Series:
    """Project a monthly/quarterly series onto the (business-)daily FX index.

    FRED stamps each CPI print on the FIRST day of its period. Two options:
      * "ffill":       step function. The latest print stays in force until the
                       next one. Simple, and there is no interpolation across the
                       gaps between prints.
      * "interpolate": straight line (in calendar time) between consecutive
                       prints, which gives a smooth fair-value line. Beyond the
                       last print there is nothing to interpolate to, so the
                       last value is held flat.
    Days before the first observation stay NaN and are dropped later.
    """
    union = series.index.union(daily_index)
    s = series.reindex(union)
    if method == "interpolate":
        s = s.interpolate(method="time", limit_area="inside")
    s = s.ffill()
    return s.reindex(daily_index)


def compute_ppp(
    spot: pd.Series,
    cpi_base: pd.Series,
    cpi_quote: pd.Series,
    base_year: int,
    fill_method: str = "ffill",
    extend_stale: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Relative purchasing power parity fair value and deviation.

    Relative PPP says the exchange rate should drift by the inflation
    differential. With S quoted as QUOTE per BASE:

        S*_t = S_0  x  (P_quote_t / P_quote_0)  /  (P_base_t / P_base_0)

    * S_0  : average spot during the base year. The base year is assumed to be
             "in equilibrium", so the fair-value line starts there. A whole-year
             average is used instead of a single day so that one noisy quote
             does not shift the line for the entire history.
    * P_0  : average CPI during the base year (see _rebase).
    * If the quote country inflates faster than the base country, the ratio
      rises above 1 and fair value (QUOTE per BASE) rises: the quote currency
      "should" depreciate.

    Deviation (%) = (Spot / S* - 1) x 100.
        > 0  BASE trades above fair value (overvalued vs QUOTE)
        < 0  BASE trades below fair value (undervalued vs QUOTE)
    """
    if fill_method not in FILL_METHODS:
        raise ValueError(f"fill_method must be one of {FILL_METHODS}")

    spot_in_base_year = spot[spot.index.year == base_year]
    if spot_in_base_year.empty:
        raise DataUnavailableError(f"No spot data in base year {base_year}.")
    s0 = float(spot_in_base_year.mean())

    # CPI is published with a lag (and some series are frozen for years). Optionally
    # extend it to the last spot date with its own recent trend instead of
    # holding the last level flat, which would understate the inflation differential.
    extension: dict = {}
    if extend_stale:
        through = spot.index.max()
        cpi_base, info_b = extend_cpi(cpi_base, through)
        cpi_quote, info_q = extend_cpi(cpi_quote, through)
        extension = {k: v for k, v in (("base", info_b), ("quote", info_q)) if v}

    base_idx, n_base = _rebase(cpi_base, base_year, "Base-currency")
    quote_idx, n_quote = _rebase(cpi_quote, base_year, "Quote-currency")

    # Frequency alignment: monthly/quarterly -> daily, each series on its own,
    # BEFORE combining them (they may have different release frequencies).
    base_daily = _to_daily(base_idx, spot.index, fill_method)
    quote_daily = _to_daily(quote_idx, spot.index, fill_method)

    fair_value = s0 * (quote_daily / base_daily)

    df = pd.DataFrame(
        {"spot": spot, "cpi_base": base_daily, "cpi_quote": quote_daily, "fair_value": fair_value}
    ).dropna()
    df["deviation_pct"] = (df["spot"] / df["fair_value"] - 1.0) * 100.0

    meta = {
        "spot_base_year_avg": s0,
        "cpi_obs_in_base_year": {"base": n_base, "quote": n_quote},
        "fill_method": fill_method,
        "cpi_extended": extension,
    }
    return df, meta


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
@dataclass
class PPPResult:
    pair: PairSpec
    base_year: int
    data: pd.DataFrame
    meta: dict = field(default_factory=dict)


def build_dataset(
    pair: str | PairSpec,
    start: date,
    end: date,
    base_year: int,
    fill_method: str = "ffill",
    base_cpi_id: Optional[str] = None,
    quote_cpi_id: Optional[str] = None,
    extend_stale_cpi: bool = True,
) -> PPPResult:
    """Fetch everything and return the aligned dataset for [start, end]."""
    spec = parse_pair(pair) if isinstance(pair, str) else pair
    if start >= end:
        raise ValueError("Start date must be before end date.")
    if base_year < MIN_BASE_YEAR:
        raise ValueError(f"Base year must be {MIN_BASE_YEAR} or later (Yahoo FX history starts ~2003).")
    if base_year > end.year:
        raise ValueError("Base year cannot be after the end of the date range.")

    # The base year must be fetched even if the user's display window starts later.
    fetch_start = min(start, date(base_year, 1, 1))
    # CPI is stamped at the start of each period, so pull extra history before
    # the window; otherwise the first days would have nothing to forward-fill from.
    cpi_start = fetch_start - timedelta(days=180)

    spot = fetch_spot(spec.ticker, fetch_start, end)
    cpi_base, base_id = fetch_cpi(spec.base, cpi_start, end, base_year, base_cpi_id)
    cpi_quote, quote_id = fetch_cpi(spec.quote, cpi_start, end, base_year, quote_cpi_id)

    df, meta = compute_ppp(spot, cpi_base, cpi_quote, base_year, fill_method, extend_stale_cpi)
    df = df.loc[pd.Timestamp(start): pd.Timestamp(end)]
    if df.empty:
        raise DataUnavailableError("No overlapping spot and CPI data in the selected date range.")

    meta.update(
        cpi_series={"base": base_id, "quote": quote_id},
        cpi_last_obs={"base": cpi_base.index.max(), "quote": cpi_quote.index.max()},
    )
    return PPPResult(pair=spec, base_year=base_year, data=df, meta=meta)


# ----------------------------------------------------------------------------
# Signal
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Signal:
    label: str          # "Undervalued" | "Overvalued" | "Fairly valued"
    severity: str       # "fair" | "moderate" | "strong"
    deviation_pct: float
    percentile: float   # share of days in the window with a LOWER deviation (0-100)
    headline: str
    color: str
    as_of: pd.Timestamp


def compute_signal(result: PPPResult) -> Signal:
    df = result.data
    dev = float(df["deviation_pct"].iloc[-1])
    percentile = float((df["deviation_pct"] < dev).mean() * 100.0)
    base, quote = result.pair.base, result.pair.quote

    if abs(dev) < FAIR_BAND_PCT:
        label, severity, color = "Fairly valued", "fair", "#64748b"
        headline = f"{base} is trading close to PPP fair value vs {quote} ({dev:+.1f}%)"
    else:
        severity = "strong" if abs(dev) >= STRONG_PCT else "moderate"
        if dev > 0:
            label, color = "Overvalued", "#dc2626"
        else:
            label, color = "Undervalued", "#16a34a"
        headline = f"{base} is currently {abs(dev):.1f}% {label.lower()} vs {quote}"

    return Signal(label, severity, dev, percentile, headline, color, df.index[-1])


# ----------------------------------------------------------------------------
# CLI: sanity checks that need network access
# ----------------------------------------------------------------------------
def _check_series() -> None:
    """Print the latest observation for every configured CPI series."""
    end = date.today()
    start = end - timedelta(days=3650)
    for ccy, ids in CPI_SERIES.items():
        for sid in ids:
            try:
                s = _fetch_fred_series(sid, start, end)
                age = (pd.Timestamp(end) - s.index.max()).days
                tag = "OK   " if age <= STALE_CPI_DAYS else "STALE"
                print(f"{tag} {ccy}  {sid:<22} last obs {s.index.max():%Y-%m-%d} ({age} days old)")
            except Exception as exc:
                print(f"FAIL  {ccy}  {sid:<22} {str(exc)[:110]}")


def _discover(currency: Optional[str] = None, raw: bool = False) -> None:
    """Show fresh CPI series that FRED's search finds (or, with raw=True, everything it returns)."""
    api_key = _get_fred_api_key()
    if not api_key:
        print("No FRED API key found (env FRED_API_KEY or .streamlit/secrets.toml).")
        return
    end = date.today()
    currencies = [currency.upper()] if currency else list(CPI_SERIES)
    for ccy in currencies:
        if ccy not in CPI_SERIES:
            print(f"Unknown currency {ccy}")
            continue
        if raw:
            country = COUNTRY_SEARCH_NAMES[ccy].lower()
            rows = [r for r in _search_country_cpi(ccy, api_key) if country in str(r.get("title", "")).lower()]
            rows.sort(key=lambda r: str(r.get("observation_end", "")), reverse=True)
            print(f"{ccy}: {len(rows)} series mention {COUNTRY_SEARCH_NAMES[ccy]} (freshest first)")
            for r in rows[:25]:
                print(f"  {r['id']:<26} {r.get('frequency_short')}  {str(r.get('units', ''))[:16]:<16} "
                      f"ends {r.get('observation_end')}  {str(r.get('title', ''))[:70]}")
            continue
        rows = discover_cpi_series(ccy, 2015, end, max_results=5)
        if not rows:
            print(f"{ccy}: nothing fresh found")
        for r in rows:
            print(f"{ccy}  {r['id']:<26} {r.get('frequency_short')}  ends {r.get('observation_end')}  {r.get('title')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Currency fair value data engine")
    parser.add_argument("--check", action="store_true", help="test every configured FRED CPI series")
    parser.add_argument("--discover", action="store_true", help="search FRED for fresh CPI series")
    parser.add_argument("--currency", help="limit --discover to one currency, e.g. JPY")
    parser.add_argument("--raw", action="store_true", help="with --discover: show every hit, unfiltered")
    parser.add_argument("--pair", default="EUR/USD")
    parser.add_argument("--base-year", type=int, default=2015)
    parser.add_argument("--years", type=int, default=10, help="length of the window ending today")
    args = parser.parse_args()

    if args.check:
        _check_series()
        return
    if args.discover:
        _discover(args.currency, args.raw)
        return

    end = date.today()
    start = date(end.year - args.years, 1, 1)
    result = build_dataset(args.pair, start, end, args.base_year)
    print(result.data.tail(10).round(4))
    print(compute_signal(result).headline)
    print("meta:", result.meta)


if __name__ == "__main__":
    main()
