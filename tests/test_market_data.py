import pandas as pd
import pytest

from easyvaluator.exceptions import DataUnavailableError
from easyvaluator.market_data import _latest, extract_fcf


def test_extract_fcf_prefers_direct_fcf_field():
    cashflow = pd.DataFrame(
        {"2023-12-31": [50.0, 999.0], "2022-12-31": [40.0, 999.0]},
        index=["Free Cash Flow", "Operating Cash Flow"],
    )
    fcf = extract_fcf(cashflow)
    assert list(fcf.values) == [40.0, 50.0]  # sorted oldest first


def test_extract_fcf_adds_negative_capex():
    # yfinance reports CapEx as a negative cash outflow, so FCF = OpCash + CapEx.
    cashflow = pd.DataFrame(
        {"2023-12-31": [120.0, -20.0]},
        index=["Operating Cash Flow", "Capital Expenditure"],
    )
    assert extract_fcf(cashflow).iloc[0] == pytest.approx(100.0)


def test_extract_fcf_drops_missing_periods():
    cashflow = pd.DataFrame(
        {"2023-12-31": [50.0], "2022-12-31": [None], "2021-12-31": [30.0]},
        index=["Free Cash Flow"],
    )
    assert list(extract_fcf(cashflow).values) == [30.0, 50.0]


@pytest.mark.parametrize("cashflow", [None, pd.DataFrame()])
def test_extract_fcf_raises_on_empty(cashflow):
    with pytest.raises(DataUnavailableError):
        extract_fcf(cashflow)


def test_extract_fcf_raises_when_fields_missing():
    cashflow = pd.DataFrame({"2023-12-31": [1.0]}, index=["Some Unrelated Field"])
    with pytest.raises(DataUnavailableError, match="Some Unrelated Field"):
        extract_fcf(cashflow)


def test_latest_picks_most_recent_period_regardless_of_column_order():
    statement = pd.DataFrame(
        {pd.Timestamp("2022-12-31"): [0.19], pd.Timestamp("2024-12-31"): [0.16], pd.Timestamp("2023-12-31"): [0.2]},
        index=["Tax Rate For Calcs"],
    )
    assert _latest(statement, "Tax Rate For Calcs") == pytest.approx(0.16)
    assert _latest(statement, "Missing Row") is None
    assert _latest(None, "Tax Rate For Calcs") is None


class _FakeTicker:
    def __init__(self, info, cashflow, financials=None, history=None):
        self.info = info
        self.cashflow = cashflow
        self.financials = financials
        self._history = history if history is not None else pd.DataFrame()

    def history(self, period):
        return self._history


class _FakeYF:
    """Stands in for the yfinance module inside YahooFinanceProvider."""

    def __init__(self, tickers):
        self._tickers = tickers

    def Ticker(self, symbol):  # mirrors yfinance's API
        return self._tickers[symbol]


def _provider(**tickers):
    from easyvaluator.market_data import YahooFinanceProvider

    provider = YahooFinanceProvider()
    provider._yf = _FakeYF(tickers)
    return provider


CASHFLOW = pd.DataFrame(
    {pd.Timestamp("2024-12-31"): [120.0], pd.Timestamp("2023-12-31"): [100.0]},
    index=["Free Cash Flow"],
)


def test_yahoo_snapshot_normalizes_pence_and_reads_statements():
    financials = pd.DataFrame(
        {pd.Timestamp("2024-12-31"): [-30.0, 0.25], pd.Timestamp("2023-12-31"): [-20.0, 0.2]},
        index=["Interest Expense", "Tax Rate For Calcs"],
    )
    info = {
        "longName": "Shell plc",
        "currency": "GBp",
        "financialCurrency": "USD",
        "country": "United Kingdom",
        "exchange": "LSE",
        "currentPrice": 3596.5,
        "sharesOutstanding": 1000.0,
        "totalDebt": 500.0,
        "totalCash": 200.0,
        "beta": 0.8,
    }
    snap = _provider(**{"SHEL.L": _FakeTicker(info, CASHFLOW, financials)}).get_company("SHEL.L")

    assert snap.price_currency == "GBP"
    assert snap.price == pytest.approx(35.965)
    assert snap.market_cap == pytest.approx(35.965 * 1000)  # derived when missing
    assert snap.financial_currency == "USD"
    assert snap.interest_expense == pytest.approx(30.0)  # latest, made positive
    assert snap.effective_tax_rate == pytest.approx(0.25)
    assert list(snap.historical_fcf) == [100.0, 120.0]


def test_yahoo_snapshot_derives_shares_from_market_cap():
    info = {"currency": "USD", "regularMarketPrice": 10.0, "marketCap": 5000.0}
    snap = _provider(X=_FakeTicker(info, CASHFLOW)).get_company("X")
    assert snap.shares_outstanding == pytest.approx(500.0)
    assert snap.name == "X"


@pytest.mark.parametrize(
    "info",
    [
        {},  # invalid ticker: yfinance returns an empty info dict
        {"currency": "USD", "currentPrice": 10.0},  # no shares, no market cap
    ],
)
def test_yahoo_snapshot_missing_essentials(info):
    with pytest.raises(DataUnavailableError):
        _provider(X=_FakeTicker(info, CASHFLOW)).get_company("X")


def test_yahoo_rates():
    tnx = _FakeTicker({}, None, history=pd.DataFrame({"Close": [4.1, 4.25]}))
    fx = _FakeTicker({}, None, history=pd.DataFrame({"Close": [0.91, 0.92]}))
    empty = _FakeTicker({}, None)
    provider = _provider(**{"^TNX": tnx, "USDEUR=X": fx, "USDXYZ=X": empty})

    assert provider.get_us_treasury_yield() == pytest.approx(0.0425)
    assert provider.get_fx_rate("USD", "EUR") == pytest.approx(0.92)
    assert provider.get_fx_rate("USD", "XYZ") is None
