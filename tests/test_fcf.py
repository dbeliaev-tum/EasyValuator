import pandas as pd
import pytest

from easyvaluator.assumptions import Assumptions
from easyvaluator.fcf import forecast_fcf, get_historical_fcf


class FakeTicker:
    """Minimal stand-in for yf.Ticker exposing only what get_historical_fcf needs."""

    def __init__(self, cashflow: pd.DataFrame):
        self.cashflow = cashflow


def test_get_historical_fcf_prefers_direct_fcf_field():
    cashflow = pd.DataFrame(
        {"2023-12-31": [50.0], "2022-12-31": [40.0]},
        index=["Free Cash Flow"],
    )
    fcf = get_historical_fcf(FakeTicker(cashflow))
    assert list(fcf.values) == [40.0, 50.0]


def test_get_historical_fcf_adds_negative_capex():
    # yfinance reports CapEx as a negative cash outflow, so FCF = OpCash + CapEx.
    cashflow = pd.DataFrame(
        {"2023-12-31": [120.0, -20.0]},
        index=["Operating Cash Flow", "Capital Expenditure"],
    )
    fcf = get_historical_fcf(FakeTicker(cashflow))
    assert fcf.iloc[0] == pytest.approx(100.0)


def test_get_historical_fcf_raises_on_empty_cashflow():
    with pytest.raises(ValueError):
        get_historical_fcf(FakeTicker(pd.DataFrame()))


def test_get_historical_fcf_raises_when_fields_missing():
    cashflow = pd.DataFrame({"2023-12-31": [1.0]}, index=["Some Unrelated Field"])
    with pytest.raises(ValueError):
        get_historical_fcf(FakeTicker(cashflow))


def test_forecast_fcf_decays_towards_terminal_growth():
    historical = pd.Series([100.0, 110.0], index=pd.to_datetime(["2022-01-01", "2023-01-01"]))
    assumptions = Assumptions(terminal_growth=0.02)

    forecasts, cagr = forecast_fcf(historical, years=4, assumptions=assumptions)

    assert cagr == pytest.approx(0.10, abs=1e-9)
    assert len(forecasts) == 4

    first_year_growth = forecasts[0] / 110.0 - 1
    last_year_growth = forecasts[-1] / forecasts[-2] - 1
    assert first_year_growth == pytest.approx(0.10 * 3 / 4 + 0.02 * 1 / 4)
    assert last_year_growth == pytest.approx(0.02, abs=1e-9)


def test_forecast_fcf_requires_two_years():
    with pytest.raises(ValueError):
        forecast_fcf(pd.Series([100.0]))


def test_forecast_fcf_clips_extreme_cagr():
    historical = pd.Series([10.0, 100.0], index=pd.to_datetime(["2022-01-01", "2023-01-01"]))
    assumptions = Assumptions(cagr_max=0.15)

    _, cagr = forecast_fcf(historical, years=3, assumptions=assumptions)

    assert cagr == pytest.approx(0.15)
