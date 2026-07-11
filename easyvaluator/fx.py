"""Foreign exchange utilities: live rates and currency normalization."""

import logging
from functools import lru_cache
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_exchange_rate(currency_from: str, currency_to: str) -> Optional[float]:
    """Look up the current exchange rate between two currencies via Yahoo Finance.

    Returns None if the rate cannot be retrieved. Results are cached for the
    lifetime of the process to avoid redundant API calls.
    """
    if not currency_from or not currency_to:
        return None

    if currency_from.upper() == currency_to.upper():
        return 1.0

    ticker = f"{currency_from.upper()}{currency_to.upper()}=X"

    try:
        data = yf.Ticker(ticker)

        rate = data.info.get("regularMarketPrice")
        if rate:
            return float(rate)

        hist = data.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])

    except Exception:
        logger.warning("Failed to fetch FX rate for %s->%s", currency_from, currency_to, exc_info=True)

    return None


def convert(amount: Optional[float], from_currency: Optional[str], to_currency: str) -> Optional[float]:
    """Convert `amount` from `from_currency` to `to_currency`.

    Falls back to the original (unconverted) amount if no rate is available,
    logging a warning so silent unit mismatches don't slip into a report.
    """
    if amount is None:
        return None

    if from_currency is None:
        return amount

    if from_currency.upper() == to_currency.upper():
        return amount

    rate = get_exchange_rate(from_currency, to_currency)
    if rate:
        return amount * rate

    logger.warning(
        "No FX rate available for %s->%s; value left unconverted (still denominated in %s)",
        from_currency, to_currency, from_currency,
    )
    return amount
