import pandas as pd
import pytest

from easyvaluator.assumptions import Assumptions
from easyvaluator.exceptions import ValuationError
from easyvaluator.fcf import estimate_growth, forecast_fcf, growth_path, normalized_base_fcf


def test_cagr_from_endpoints(fcf_series):
    growth = estimate_growth(fcf_series(100.0, 105.0, 110.25))
    # Year-end dates two years apart; CAGR uses elapsed days, not point count.
    assert growth.cagr == pytest.approx(0.05, rel=1e-3)
    assert growth.note is None


def test_cagr_uses_actual_dates_when_a_year_is_missing():
    # 2019 -> 2023 with 2020-2022 missing: four years elapsed, not one.
    series = pd.Series([100.0, 146.41], index=pd.to_datetime(["2019-12-31", "2023-12-31"]))
    growth = estimate_growth(series, Assumptions(cagr_max=0.5))
    assert growth.cagr == pytest.approx(0.10, rel=1e-3)


def test_cagr_is_clamped_and_reported(fcf_series):
    growth = estimate_growth(fcf_series(10.0, 100.0), Assumptions(cagr_max=0.15))
    assert growth.cagr == pytest.approx(0.15)
    assert growth.raw_cagr == pytest.approx(9.0, rel=1e-2)
    assert "clamped" in growth.note


@pytest.mark.parametrize("values", [(-50.0, 100.0), (100.0, -50.0), (0.0, 100.0)])
def test_cagr_undefined_for_non_positive_endpoints(fcf_series, values):
    a = Assumptions(terminal_growth=0.02)
    growth = estimate_growth(fcf_series(*values), a)
    assert growth.cagr == a.terminal_growth
    assert growth.raw_cagr is None
    assert growth.note


def test_estimate_growth_requires_two_points(fcf_series):
    with pytest.raises(ValuationError):
        estimate_growth(fcf_series(100.0))


def test_growth_path_decays_linearly_to_terminal():
    rates = growth_path(cagr=0.10, terminal_growth=0.02, years=4)
    assert rates == pytest.approx([0.08, 0.06, 0.04, 0.02])


def test_growth_path_single_year_is_terminal():
    assert growth_path(0.10, 0.02, 1) == pytest.approx([0.02])


def test_forecast_compounds():
    assert forecast_fcf(100.0, [0.10, 0.05]) == pytest.approx([110.0, 115.5])


def test_normalized_base_fcf_averages_recent_years(fcf_series):
    series = fcf_series(10.0, 100.0, 130.0, 70.0)
    assert normalized_base_fcf(series, 3) == pytest.approx(100.0)
    assert normalized_base_fcf(series, 1) == pytest.approx(70.0)
    assert normalized_base_fcf(series, 10) == pytest.approx(77.5)  # window longer than history


def test_normalized_base_fcf_requires_data(fcf_series):
    with pytest.raises(ValuationError):
        normalized_base_fcf(fcf_series(), 3)
