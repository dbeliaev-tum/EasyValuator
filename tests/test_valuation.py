"""End-to-end tests of the valuation pipeline against an in-memory provider."""

import pytest

from easyvaluator import Assumptions, Valuator
from easyvaluator.dcf import discount_cash_flows
from easyvaluator.exceptions import DataUnavailableError, FXRateUnavailableError, ValuationError


def test_full_pipeline_in_reporting_currency(provider, make_snapshot):
    provider.companies["TEST"] = make_snapshot()
    result = Valuator(provider, Assumptions(target_currency="EUR")).value("test")

    assert result.symbol == "TEST"
    assert result.region == "US"
    assert result.target_currency == "EUR"
    assert result.current_price == pytest.approx(50.0 * 0.9)
    assert result.historical_fcf.iloc[-1] == pytest.approx(2_662_000 * 0.9)
    assert len(result.forecast_fcf) == 5
    assert result.cagr == pytest.approx(0.10, rel=1e-2)
    assert result.wacc.tax_rate == pytest.approx(0.21)  # reported effective rate
    assert result.wacc.cost_of_debt == pytest.approx(0.05)  # 500k / 10M
    assert result.sensitivity is not None
    assert result.sensitivity.values[2][2] == pytest.approx(result.fair_value)
    assert result.warnings == []


def test_fair_value_is_currency_invariant(provider, make_snapshot):
    """Converting at the end must give the same answer as valuing in USD."""
    provider.companies["TEST"] = make_snapshot()
    usd = Valuator(provider, Assumptions(target_currency="USD")).value("TEST")
    eur = Valuator(provider, Assumptions(target_currency="EUR")).value("TEST")

    assert eur.fair_value == pytest.approx(usd.fair_value * 0.9)
    assert eur.upside == pytest.approx(usd.upside)
    assert eur.wacc == usd.wacc


def test_wacc_weights_use_one_currency_for_adrs(provider, make_snapshot):
    """ADR case (e.g. TSM): price/market cap in USD, statements in TWD.

    Before the fix, USD market cap was compared to TWD debt directly,
    inflating the debt weight ~30x.
    """
    usd_twd = 32.0
    provider.fx_rates[("USD", "TWD")] = usd_twd
    provider.fx_rates[("TWD", "EUR")] = 0.9 / usd_twd
    twd = make_snapshot(
        financial_currency="TWD",
        country="Taiwan",
        market_cap=50_000_000.0,  # USD
        total_debt=10_000_000.0 * usd_twd,  # TWD, same real amount as the USD fixture
        total_cash=4_000_000.0 * usd_twd,
        interest_expense=500_000.0 * usd_twd,
        historical_fcf=make_snapshot().historical_fcf * usd_twd,
    )
    provider.companies["TWD"] = twd
    provider.companies["USD"] = make_snapshot()

    in_twd = Valuator(provider).value("TWD")
    in_usd = Valuator(provider).value("USD")

    assert in_twd.wacc.debt_weight == pytest.approx(in_usd.wacc.debt_weight)
    assert in_twd.wacc.debt_weight == pytest.approx(10 / 60)
    assert any("TWD" in w for w in in_twd.warnings)  # no TWD risk-free spread
    assert any("Taiwan" in w for w in in_twd.warnings)  # region not modelled


def test_pence_quoted_listing(provider, make_snapshot):
    """LSE prices arrive in GBp; the snapshot carries them in GBP already."""
    provider.fx_rates[("GBP", "EUR")] = 1.15
    provider.fx_rates[("USD", "GBP")] = 0.78
    provider.companies["SHEL.L"] = make_snapshot(
        symbol="SHEL.L", country="United Kingdom", price_currency="GBP", price=35.97, market_cap=35.97 * 1_000_000
    )
    result = Valuator(provider).value("SHEL.L")
    assert result.current_price == pytest.approx(35.97 * 1.15)
    assert result.region == "UK"


def test_fallbacks_are_reported_as_warnings(provider, make_snapshot):
    provider.treasury_yield = None
    provider.companies["TEST"] = make_snapshot(beta=None, interest_expense=None, effective_tax_rate=None)

    result = Valuator(provider).value("TEST")

    assert result.wacc.beta == 1.0
    assert result.wacc.cost_of_debt == pytest.approx(0.05)
    assert result.wacc.tax_rate == pytest.approx(0.21)
    joined = " ".join(result.warnings)
    for fragment in ("Beta", "Cost of debt", "tax rate", "Treasury"):
        assert fragment in joined


def test_tax_rate_override(provider, make_snapshot):
    provider.companies["TEST"] = make_snapshot()
    result = Valuator(provider, Assumptions(tax_rate=0.3)).value("TEST")
    assert result.wacc.tax_rate == pytest.approx(0.3)


def test_negative_latest_fcf_is_rejected(provider, make_snapshot, fcf_series):
    provider.companies["TEST"] = make_snapshot(historical_fcf=fcf_series(100.0, -50.0))
    with pytest.raises(ValuationError, match="negative"):
        Valuator(provider).value("TEST")


def test_too_little_history_is_rejected(provider, make_snapshot, fcf_series):
    provider.companies["TEST"] = make_snapshot(historical_fcf=fcf_series(100.0))
    with pytest.raises(ValuationError):
        Valuator(provider).value("TEST")


def test_missing_fx_rate_fails_loudly(provider, make_snapshot):
    provider.companies["TEST"] = make_snapshot()
    with pytest.raises(FXRateUnavailableError):
        Valuator(provider, Assumptions(target_currency="JPY")).value("TEST")


def test_unknown_ticker(provider):
    with pytest.raises(DataUnavailableError):
        Valuator(provider).value("NOPE")


def test_sensitivity_can_be_skipped(provider, make_snapshot):
    provider.companies["TEST"] = make_snapshot()
    assert Valuator(provider).value("TEST", sensitivity=False).sensitivity is None


def test_result_is_consistent_with_dcf_module(provider, make_snapshot):
    provider.companies["TEST"] = make_snapshot()
    result = Valuator(provider, Assumptions(target_currency="USD")).value("TEST")
    direct = discount_cash_flows(
        result.forecast_fcf, result.wacc.wacc, result.terminal_growth, 1_000_000.0, 6_000_000.0
    )
    assert result.fair_value == pytest.approx(direct.value_per_share)
