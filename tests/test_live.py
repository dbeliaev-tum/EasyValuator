"""Smoke tests against the real Yahoo Finance API.

Excluded from CI (`pytest -m "not network"`) since they depend on a third-party
service; run them locally to catch upstream schema changes.
"""

import pytest

from easyvaluator import Assumptions, YahooFinanceProvider, price_stock

pytestmark = pytest.mark.network


@pytest.mark.parametrize("symbol", ["AAPL", "SAP.DE", "SHEL.L"])
def test_live_valuation(symbol):
    result = price_stock(symbol, Assumptions(target_currency="USD"))
    assert result.fair_value == result.fair_value  # not NaN
    assert result.current_price > 0
    assert 0.06 <= result.wacc.wacc <= 0.20


def test_pence_listing_is_normalized():
    snapshot = YahooFinanceProvider().get_company("SHEL.L")
    assert snapshot.price_currency == "GBP"
    # Shell trades around GBP 20-40; in pence it would be in the thousands.
    assert snapshot.price < 200
