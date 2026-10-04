import pytest

from easyvaluator.dcf import discount_cash_flows, sensitivity_table
from easyvaluator.exceptions import ValuationError


def test_dcf_matches_manual_calculation():
    forecasts = [100.0, 110.0, 121.0]
    result = discount_cash_flows(forecasts, wacc=0.10, terminal_growth=0.02, shares_outstanding=1000.0, net_debt=200.0)

    pv_forecast = 100 / 1.10 + 110 / 1.10**2 + 121 / 1.10**3
    terminal_value = 121 * 1.02 / (0.10 - 0.02)
    pv_terminal = terminal_value / 1.10**3
    enterprise_value = pv_forecast + pv_terminal

    assert result.pv_forecast == pytest.approx(pv_forecast)
    assert result.terminal_value == pytest.approx(terminal_value)
    assert result.pv_terminal == pytest.approx(pv_terminal)
    assert result.enterprise_value == pytest.approx(enterprise_value)
    assert result.equity_value == pytest.approx(enterprise_value - 200)
    assert result.value_per_share == pytest.approx((enterprise_value - 200) / 1000)
    assert result.terminal_value_share == pytest.approx(pv_terminal / enterprise_value)


def test_net_cash_increases_equity_value():
    with_cash = discount_cash_flows([100.0], 0.1, 0.02, 1.0, net_debt=-50.0)
    assert with_cash.equity_value == pytest.approx(with_cash.enterprise_value + 50)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forecasts": []},
        {"wacc": 0.02},  # == g
        {"wacc": 0.01},  # < g
        {"shares_outstanding": 0.0},
    ],
)
def test_invalid_inputs_raise(kwargs):
    args = {"forecasts": [100.0], "wacc": 0.1, "terminal_growth": 0.02, "shares_outstanding": 10.0, "net_debt": 0.0}
    with pytest.raises(ValuationError):
        discount_cash_flows(**{**args, **kwargs})


def test_mid_year_convention_shifts_every_cash_flow_half_a_year():
    forecasts = [100.0, 110.0]
    end = discount_cash_flows(forecasts, 0.10, 0.02, 1.0, 0.0)
    mid = discount_cash_flows(forecasts, 0.10, 0.02, 1.0, 0.0, mid_year=True)

    assert mid.pv_forecast == pytest.approx(end.pv_forecast * 1.10**0.5)
    assert mid.pv_terminal == pytest.approx(end.pv_terminal * 1.10**0.5)
    assert mid.terminal_value == pytest.approx(end.terminal_value)  # undiscounted TV is unchanged


def _constant(forecasts):
    return lambda g: forecasts


def test_sensitivity_centre_equals_base_case():
    forecasts = [100.0, 105.0, 110.0]
    base = discount_cash_flows(forecasts, 0.09, 0.025, 10.0, 50.0, mid_year=True)
    table = sensitivity_table(_constant(forecasts), 0.09, 0.025, 10.0, 50.0, mid_year=True, steps=2)

    assert len(table.values) == 5
    assert all(len(row) == 5 for row in table.values)
    assert table.values[2][2] == pytest.approx(base.value_per_share)


def test_sensitivity_rebuilds_forecast_for_each_growth_rate():
    """Each column's forecast must end at that column's g, like the base case."""
    requested = []

    def forecast_for(g):
        requested.append(g)
        return [100.0 * (1 + g)]

    table = sensitivity_table(forecast_for, 0.09, 0.02, 1.0, 0.0)

    assert sorted(requested) == pytest.approx(table.growth_values)
    expected = discount_cash_flows([100.0 * 1.03], 0.09, 0.03, 1.0, 0.0).value_per_share
    assert table.values[2][4] == pytest.approx(expected)


def test_sensitivity_is_monotonic():
    table = sensitivity_table(lambda g: [100.0, 100.0 * (1 + g)], 0.09, 0.02, 10.0, 0.0)
    centre_row = table.values[2]
    centre_col = [row[2] for row in table.values]
    assert centre_row == sorted(centre_row)  # higher g -> higher value
    assert centre_col == sorted(centre_col, reverse=True)  # higher WACC -> lower value


def test_sensitivity_marks_undefined_cells():
    table = sensitivity_table(
        _constant([100.0]), wacc=0.03, terminal_growth=0.025, shares_outstanding=1.0, net_debt=0.0
    )
    assert table.values[0][-1] is None  # WACC 1% vs g 3.5%


def test_sensitivity_scale_is_applied():
    plain = sensitivity_table(_constant([100.0]), 0.09, 0.02, 1.0, 0.0)
    scaled = sensitivity_table(_constant([100.0]), 0.09, 0.02, 1.0, 0.0, scale=0.5)
    assert scaled.values[2][2] == pytest.approx(plain.values[2][2] * 0.5)
    assert scaled.to_frame().shape == (5, 5)
