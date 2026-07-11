"""Thin wrapper around yfinance so all external data access lives in one place."""

import logging
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)


def get_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol.upper())


def get_info(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        logger.warning("Failed to fetch ticker info for %s", t.ticker, exc_info=True)
        return {}


def get_interest_expense(t: yf.Ticker) -> Optional[float]:
    """Look up interest expense, falling back from `info` to the income statement.

    `info['interestExpense']` is unreliable/missing for most tickers in
    current yfinance; the income statement's "Interest Expense" row is a
    more dependable source.
    """
    info = get_info(t)
    value = info.get("interestExpense")
    if value:
        return abs(float(value))

    try:
        financials = t.financials
        if financials is not None and not financials.empty and "Interest Expense" in financials.index:
            row = financials.loc["Interest Expense"].dropna()
            if not row.empty:
                return abs(float(row.iloc[0]))
    except Exception:
        logger.warning("Failed to fetch interest expense from financials for %s", t.ticker, exc_info=True)

    return None


def get_country(t: yf.Ticker) -> str:
    """Classify a company into a region code ('US', 'EU', 'UK', 'CN', 'JP')
    based on exchange and country metadata, defaulting to 'US'.
    """
    info = get_info(t)
    exchange = (info.get("exchange") or "").upper()
    country_code = (info.get("country") or "").upper()

    if any(x in exchange for x in ("FRA", "PAR", "GER", "XETRA")):
        return "EU"
    if "LON" in exchange or "LSE" in exchange:
        return "UK"
    if country_code in ("GERMANY", "FRANCE", "ITALY", "SPAIN", "NETHERLANDS"):
        return "EU"
    if country_code in ("UNITED KINGDOM", "UK"):
        return "UK"
    if country_code in ("CHINA", "HONG KONG"):
        return "CN"
    if country_code == "JAPAN":
        return "JP"
    return "US"
