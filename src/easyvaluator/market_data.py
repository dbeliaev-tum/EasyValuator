"""Market data access.

The valuation engine depends only on the `MarketDataProvider` protocol;
`YahooFinanceProvider` is the production implementation and the only place
in the package that imports yfinance. Tests substitute an in-memory provider.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

import pandas as pd

from .exceptions import DataUnavailableError
from .fx import normalize_currency
from .models import CompanySnapshot

logger = logging.getLogger(__name__)


class MarketDataProvider(Protocol):
    def get_company(self, symbol: str) -> CompanySnapshot:
        """Return a snapshot of the company's market and financial data."""
        ...

    def get_fx_rate(self, from_currency: str, to_currency: str) -> float | None:
        """Spot rate for one unit of `from_currency` in `to_currency`, or None."""
        ...

    def get_us_treasury_yield(self) -> float | None:
        """Current US 10Y Treasury yield as a decimal (0.042 == 4.2%), or None."""
        ...


# yfinance exposes a ready-made "Free Cash Flow" line item on current data;
# prefer it when present since it's the source-of-truth figure.
FCF_FIELDS = ["Free Cash Flow"]

OP_CASH_FIELDS = [
    "Operating Cash Flow",
    "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
    "Cash From Operating Activities",
]

CAPEX_FIELDS = [
    "Capital Expenditure",
    "Capital Expenditures",
    "Purchase Of PPE",
    "Purchase Of Property Plant And Equipment",
]

US_TREASURY_TICKER = "^TNX"


def _first_available(frame: pd.DataFrame, fields: list[str]) -> pd.Series | None:
    for name in fields:
        if name in frame.columns:
            return frame[name]
    return None


def extract_fcf(cashflow: pd.DataFrame | None) -> pd.Series:
    """Extract historical Free Cash Flow from a yfinance cash flow statement
    (line items as rows, periods as columns), sorted oldest first.

    Prefers yfinance's own `Free Cash Flow` row. Falls back to
    Operating Cash Flow + Capital Expenditure — yfinance reports CapEx as a
    negative cash outflow, so it must be *added*, not subtracted.
    """
    if cashflow is None or cashflow.empty:
        raise DataUnavailableError("No cash flow statement available")

    by_period = cashflow.T
    fcf = _first_available(by_period, FCF_FIELDS)
    if fcf is None:
        op_cash = _first_available(by_period, OP_CASH_FIELDS)
        capex = _first_available(by_period, CAPEX_FIELDS)
        if op_cash is None or capex is None:
            raise DataUnavailableError(
                "Could not locate cash flow fields required to compute FCF. "
                "Available line items: " + ", ".join(map(str, by_period.columns))
            )
        fcf = op_cash + capex

    fcf = pd.to_numeric(fcf, errors="coerce").dropna()
    fcf.name = "free_cash_flow"
    return fcf.sort_index()


def _latest(frame: pd.DataFrame | None, row: str) -> float | None:
    """Most recent non-null value of a line item in a yfinance statement."""
    if frame is None or frame.empty or row not in frame.index:
        return None
    values = pd.to_numeric(frame.loc[row], errors="coerce").dropna()
    if values.empty:
        return None
    # Columns are period end dates; newest first in yfinance, but don't rely on it.
    return float(values.sort_index().iloc[-1])


def _number(info: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = info.get(key)
        if isinstance(value, (int, float)) and not math.isnan(value):
            return float(value)
    return None


class YahooFinanceProvider:
    """`MarketDataProvider` backed by Yahoo Finance via yfinance."""

    def __init__(self) -> None:
        import yfinance  # imported lazily so the pure modules don't need it

        self._yf = yfinance

    def get_company(self, symbol: str) -> CompanySnapshot:
        ticker = self._yf.Ticker(symbol)
        try:
            info: dict[str, Any] = ticker.info or {}
        except Exception as exc:  # yfinance raises a variety of HTTP/parse errors
            raise DataUnavailableError(f"Could not fetch data for {symbol}: {exc}") from exc

        quoted_price = _number(info, "currentPrice", "regularMarketPrice")
        if quoted_price is None or not info.get("currency"):
            raise DataUnavailableError(f"No market price for {symbol!r} — is the ticker valid?")

        price_currency, unit = normalize_currency(info["currency"])
        price = quoted_price * unit
        financial_currency, _ = normalize_currency(info.get("financialCurrency") or info["currency"])

        shares = _number(info, "sharesOutstanding", "impliedSharesOutstanding")
        market_cap = _number(info, "marketCap")
        if not shares and market_cap:
            shares = market_cap / price
        if not shares:
            raise DataUnavailableError(f"Shares outstanding not available for {symbol}")
        if not market_cap:
            market_cap = price * shares

        try:
            financials = ticker.financials
        except Exception:
            logger.warning("Income statement unavailable for %s", symbol, exc_info=True)
            financials = None

        interest = _number(info, "interestExpense")
        if interest is None:
            interest = _latest(financials, "Interest Expense")

        return CompanySnapshot(
            symbol=symbol,
            name=info.get("longName") or info.get("shortName") or symbol,
            country=info.get("country"),
            exchange=info.get("exchange"),
            price_currency=price_currency,
            financial_currency=financial_currency,
            price=price,
            shares_outstanding=shares,
            market_cap=market_cap,
            total_debt=_number(info, "totalDebt") or 0.0,
            total_cash=_number(info, "totalCash") or 0.0,
            historical_fcf=extract_fcf(ticker.cashflow),
            beta=_number(info, "beta"),
            interest_expense=abs(interest) if interest is not None else None,
            effective_tax_rate=_latest(financials, "Tax Rate For Calcs"),
        )

    def get_fx_rate(self, from_currency: str, to_currency: str) -> float | None:
        pair = self._yf.Ticker(f"{from_currency}{to_currency}=X")
        try:
            hist = pair.history(period="5d")
        except Exception:
            logger.warning("FX lookup failed for %s->%s", from_currency, to_currency, exc_info=True)
            return None
        if hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])

    def get_us_treasury_yield(self) -> float | None:
        try:
            hist = self._yf.Ticker(US_TREASURY_TICKER).history(period="5d")
        except Exception:
            logger.warning("Treasury yield lookup failed", exc_info=True)
            return None
        if hist.empty:
            return None
        # ^TNX is quoted in percent (4.25 == 4.25%).
        return float(hist["Close"].dropna().iloc[-1]) / 100.0
