import pytest

from easyvaluator import Assumptions, InvalidAssumptionError


def test_defaults_are_valid():
    a = Assumptions()
    assert a.terminal_growth < a.wacc_min


def test_currency_is_normalized():
    assert Assumptions(target_currency=" usd ").target_currency == "USD"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_currency": "EURO"},
        {"forecast_years": 0},
        {"forecast_years": 31},
        {"tax_rate": 1.0},
        {"tax_rate": -0.1},
        {"wacc_min": 0.3, "wacc_max": 0.2},
        {"beta_min": 3.0},
        # g >= WACC makes the Gordon Growth model diverge.
        {"terminal_growth": 0.06},
        {"terminal_growth": 0.07, "wacc_min": 0.05},
    ],
)
def test_invalid_assumptions_rejected(kwargs):
    with pytest.raises(InvalidAssumptionError):
        Assumptions(**kwargs)


def test_invalid_assumption_is_also_a_value_error():
    with pytest.raises(ValueError, match="forecast_years"):
        Assumptions(forecast_years=-1)
