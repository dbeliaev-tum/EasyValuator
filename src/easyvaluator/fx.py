"""Currency normalization and conversion."""

from __future__ import annotations

from collections.abc import Callable

from .exceptions import FXRateUnavailableError

# Some exchanges quote prices in a minor unit. Yahoo marks these with a
# lower-case suffix ("GBp" for pence) or a dedicated code ("ILA" for agorot).
# Treating "GBp" as "GBP" would overstate every LSE share price 100x.
MINOR_UNITS: dict[str, tuple[str, float]] = {
    "GBp": ("GBP", 0.01),
    "GBX": ("GBP", 0.01),
    "ZAc": ("ZAR", 0.01),
    "ZAC": ("ZAR", 0.01),
    "ILA": ("ILS", 0.01),
}


def normalize_currency(code: str) -> tuple[str, float]:
    """Map a quoted currency code to (ISO code in major units, multiplier).

    >>> normalize_currency("GBp")
    ('GBP', 0.01)
    >>> normalize_currency("usd")
    ('USD', 1.0)
    """
    if code in MINOR_UNITS:
        return MINOR_UNITS[code]
    return code.upper(), 1.0


RateFetcher = Callable[[str, str], "float | None"]


class FXConverter:
    """Converts amounts between ISO currencies, caching each rate it fetches.

    Unlike a "best effort" converter, a missing rate raises instead of
    silently returning the unconverted amount — mixing currencies in a
    valuation produces numbers that look plausible and are wrong.
    """

    def __init__(self, fetch_rate: RateFetcher) -> None:
        self._fetch_rate = fetch_rate
        self._cache: dict[tuple[str, str], float] = {}

    def rate(self, from_currency: str, to_currency: str) -> float:
        src, dst = from_currency.upper(), to_currency.upper()
        if src == dst:
            return 1.0

        key = (src, dst)
        if key not in self._cache:
            rate = self._fetch_rate(src, dst)
            if rate is None or rate <= 0:
                # Some crosses are only listed one way round.
                inverse = self._fetch_rate(dst, src)
                if inverse is None or inverse <= 0:
                    raise FXRateUnavailableError(f"No exchange rate available for {src}->{dst}")
                rate = 1.0 / inverse
            self._cache[key] = rate
        return self._cache[key]

    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        return amount * self.rate(from_currency, to_currency)
