import pytest

from easyvaluator import regions


@pytest.mark.parametrize(
    ("symbol", "country", "expected"),
    [
        ("AAPL", "United States", "US"),
        ("SAP.DE", "Germany", "EU"),
        ("SHEL.L", "United Kingdom", "UK"),
        ("7203.T", "Japan", "JP"),
        ("0700.HK", "Hong Kong", "CN"),
        ("TM", "Japan", "JP"),  # ADR: domicile beats listing venue
        ("MC.PA", None, "EU"),  # suffix fallback
        ("AAPL", None, "US"),  # bare ticker, no metadata
        ("TSM", "Taiwan", None),  # not modelled
    ],
)
def test_classify_region(symbol, country, expected):
    assert regions.classify_region(symbol, country) == expected


def test_equity_risk_premium():
    assert regions.equity_risk_premium("US") == pytest.approx(regions.MATURE_MARKET_ERP)
    assert regions.equity_risk_premium("CN") == pytest.approx(regions.MATURE_MARKET_ERP + 0.01)
    assert regions.equity_risk_premium("BR") == regions.equity_risk_premium("US")


def test_risk_free_rate_live_with_spread():
    rf = regions.risk_free_rate("EUR", 0.045)
    assert rf.rate == pytest.approx(0.03)
    assert rf.is_live
    assert rf.currency_supported


def test_risk_free_rate_falls_back_on_implausible_yield():
    # e.g. ^TNX quoted x10 by mistake would read 42%
    rf = regions.risk_free_rate("USD", 0.42)
    assert rf.rate == regions.FALLBACK_US_TREASURY_YIELD
    assert not rf.is_live


def test_risk_free_rate_unsupported_currency_flags_it():
    rf = regions.risk_free_rate("TWD", 0.04)
    assert rf.rate == pytest.approx(0.04)
    assert not rf.currency_supported


def test_risk_free_rate_is_floored():
    assert regions.risk_free_rate("CHF", 0.02).rate == pytest.approx(0.005)
