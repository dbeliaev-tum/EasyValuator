"""Region classification and region/currency-dependent market parameters.

Two different things are looked up here, on purpose with different keys:

* The **risk-free rate** must be in the same currency as the cash flows being
  discounted, so it is keyed by the *financial-statement currency*.
* The **equity risk premium** reflects where the business operates, so it is
  keyed by the company's *region*.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_REGION = "US"

# Implied equity risk premium for a mature market, in line with
# Damodaran's forward-looking estimates for the S&P 500 (~4-4.5% in recent
# years). Historical-average premiums (5.5%+) stacked on today's higher
# risk-free rates double-count risk and push mega-cap WACCs above 10%.
MATURE_MARKET_ERP = 0.045

# Additional country risk premium on top of the mature-market ERP.
COUNTRY_RISK_PREMIUMS: dict[str, float] = {
    "US": 0.0,
    "EU": 0.0,
    "UK": 0.0,
    "JP": 0.0,
    "CN": 0.010,
}

EQUITY_RISK_PREMIUMS: dict[str, float] = {
    region: MATURE_MARKET_ERP + premium for region, premium in COUNTRY_RISK_PREMIUMS.items()
}

# 10Y government yield spread vs. the US Treasury, by currency.
RISK_FREE_SPREADS: dict[str, float] = {
    "USD": 0.0,
    "HKD": 0.0,  # pegged to USD
    "EUR": -0.015,
    "GBP": -0.005,
    "CHF": -0.035,
    "CNY": -0.020,
    "JPY": -0.025,
}

# Used when the live Treasury yield can't be fetched or looks implausible.
FALLBACK_US_TREASURY_YIELD = 0.042

_COUNTRY_TO_REGION = {
    "UNITED STATES": "US",
    "UNITED KINGDOM": "UK",
    "CHINA": "CN",
    "HONG KONG": "CN",
    "JAPAN": "JP",
    **dict.fromkeys(
        (
            "AUSTRIA",
            "BELGIUM",
            "FINLAND",
            "FRANCE",
            "GERMANY",
            "IRELAND",
            "ITALY",
            "LUXEMBOURG",
            "NETHERLANDS",
            "PORTUGAL",
            "SPAIN",
            "DENMARK",
            "SWEDEN",
        ),
        "EU",
    ),
}

_SUFFIX_TO_REGION = {
    "L": "UK",
    "DE": "EU",
    "F": "EU",
    "PA": "EU",
    "AS": "EU",
    "MI": "EU",
    "MC": "EU",
    "BR": "EU",
    "LS": "EU",
    "VI": "EU",
    "HE": "EU",
    "IR": "EU",
    "T": "JP",
    "HK": "CN",
    "SS": "CN",
    "SZ": "CN",
}


@dataclass(frozen=True)
class RiskFreeRate:
    rate: float
    source: str
    is_live: bool
    currency_supported: bool


def classify_region(symbol: str, country: str | None) -> str | None:
    """Classify a company as 'US', 'EU', 'UK', 'CN' or 'JP'.

    The country of domicile wins over the listing venue: an ADR of a Japanese
    company trades in New York but carries Japanese business risk. The ticker
    suffix is the fallback when the country is unknown. Returns None when
    neither identifies a supported region.
    """
    if country:
        region = _COUNTRY_TO_REGION.get(country.strip().upper())
        if region:
            return region
    if "." in symbol:
        suffix = symbol.rsplit(".", 1)[1].upper()
        if suffix in _SUFFIX_TO_REGION:
            return _SUFFIX_TO_REGION[suffix]
    if not country and "." not in symbol:
        return DEFAULT_REGION  # plain US-style ticker with no metadata
    return None


def equity_risk_premium(region: str) -> float:
    return EQUITY_RISK_PREMIUMS.get(region, EQUITY_RISK_PREMIUMS[DEFAULT_REGION])


def risk_free_rate(currency: str, us_treasury_yield: float | None) -> RiskFreeRate:
    """Risk-free rate for cash flows in `currency`, as a spread over the US 10Y.

    Unsupported currencies get the US rate; the caller should flag it, since
    discounting e.g. INR cash flows at a USD rate understates the risk.
    """
    if us_treasury_yield is not None and 0.001 < us_treasury_yield < 0.15:
        base, is_live = us_treasury_yield, True
    else:
        base, is_live = FALLBACK_US_TREASURY_YIELD, False
    source = "US 10Y Treasury" if is_live else "fallback US 10Y Treasury"

    spread = RISK_FREE_SPREADS.get(currency.upper())
    if spread is None:
        return RiskFreeRate(base, source, is_live, currency_supported=False)
    if spread == 0.0:
        return RiskFreeRate(base, source, is_live, currency_supported=True)
    # Floor at 0.5%: a near-zero or negative rate makes CAPM meaningless.
    return RiskFreeRate(max(base + spread, 0.005), f"{source} {spread:+.2%} {currency.upper()} spread", is_live, True)
