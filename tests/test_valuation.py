import pytest

from easyvaluator.valuation import calculate_dcf, get_market_risk_premium


def test_calculate_dcf_matches_manual_calculation():
    forecasts = [100.0, 110.0, 121.0]
    wacc = 0.10
    terminal_growth = 0.02
    shares_outstanding = 1000.0
    net_debt = 200.0

    result = calculate_dcf(forecasts, wacc, terminal_growth, shares_outstanding, net_debt)

    expected_pv_fcf = 100 / 1.10 + 110 / 1.10**2 + 121 / 1.10**3
    expected_terminal_value = 121 * 1.02 / (0.10 - 0.02)
    expected_pv_terminal = expected_terminal_value / 1.10**3
    expected_enterprise_value = expected_pv_fcf + expected_pv_terminal
    expected_equity_value = expected_enterprise_value - net_debt

    assert result["pv_fcf"] == pytest.approx(expected_pv_fcf)
    assert result["pv_terminal"] == pytest.approx(expected_pv_terminal)
    assert result["enterprise_value"] == pytest.approx(expected_enterprise_value)
    assert result["equity_value"] == pytest.approx(expected_equity_value)
    assert result["price_per_share"] == pytest.approx(expected_equity_value / shares_outstanding)


def test_calculate_dcf_handles_zero_shares():
    result = calculate_dcf([100.0], 0.10, 0.02, shares_outstanding=0, net_debt=0)
    assert result["price_per_share"] == 0.0


def test_market_risk_premium_known_region():
    assert get_market_risk_premium("CN") == pytest.approx(0.070)


def test_market_risk_premium_unknown_region_defaults():
    assert get_market_risk_premium("BR") == pytest.approx(0.055)
