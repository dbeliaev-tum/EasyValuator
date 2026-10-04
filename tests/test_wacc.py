import pytest

from easyvaluator.assumptions import Assumptions
from easyvaluator.wacc import calculate_wacc

BASE = {
    "risk_free_rate": 0.04,
    "market_risk_premium": 0.055,
    "beta": 1.2,
    "market_cap": 800.0,
    "total_debt": 200.0,
    "cost_of_debt": 0.06,
    "tax_rate": 0.25,
}


def test_wacc_matches_manual_calculation():
    w = calculate_wacc(**BASE)
    cost_of_equity = 0.04 + 1.2 * 0.055
    expected = 0.8 * cost_of_equity + 0.2 * 0.06 * (1 - 0.25)

    assert w.cost_of_equity == pytest.approx(cost_of_equity)
    assert w.equity_weight == pytest.approx(0.8)
    assert w.debt_weight == pytest.approx(0.2)
    assert w.wacc == pytest.approx(expected)


def test_all_equity_firm():
    w = calculate_wacc(**{**BASE, "total_debt": 0.0})
    assert w.debt_weight == 0
    assert w.wacc == pytest.approx(w.cost_of_equity)


@pytest.mark.parametrize(("raw", "used"), [(-0.2, 0.5), (3.0, 2.0)])
def test_beta_is_clamped_and_reported(raw, used):
    assert calculate_wacc(**{**BASE, "beta": raw}).beta == used


def test_cost_of_debt_is_clamped():
    a = Assumptions(cost_of_debt_max=0.15)
    assert calculate_wacc(**{**BASE, "cost_of_debt": 0.9}, assumptions=a).cost_of_debt == 0.15


def test_wacc_is_clamped_to_bounds():
    a = Assumptions(wacc_min=0.06, wacc_max=0.20)
    low = calculate_wacc(**{**BASE, "risk_free_rate": 0.0, "beta": 0.5}, assumptions=a)
    high = calculate_wacc(**{**BASE, "market_risk_premium": 0.5, "beta": 2.0}, assumptions=a)
    assert low.wacc == 0.06
    assert high.wacc == 0.20
