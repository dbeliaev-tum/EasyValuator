"""WACC calculation and two-stage DCF valuation engine."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf

from . import fx, market_data
from .assumptions import Assumptions, DEFAULT_ASSUMPTIONS
from .fcf import forecast_fcf, get_historical_fcf

# Regional adjustments to the US Treasury baseline, reflecting historical
# interest-rate differentials between regions.
REGIONAL_ADJUSTMENTS = {
    "US": 0.0,
    "EU": -0.015,
    "UK": -0.005,
    "CN": -0.005,
    "JP": -0.035,
}

# Conservative fallback risk-free rates, used when live Treasury data is
# unavailable.
DEFAULT_RISK_FREE_RATES = {
    "US": 0.041,
    "EU": 0.026,
    "UK": 0.036,
    "CN": 0.028,
    "JP": 0.006,
}

# Long-term equity risk premiums by region (Damodaran-style estimates).
MARKET_RISK_PREMIUMS = {
    "US": 0.055,
    "EU": 0.055,
    "UK": 0.055,
    "CN": 0.070,
    "JP": 0.055,
}

US_TREASURY_TICKER = "^TNX"


def get_risk_free_rate(country: str = "US") -> float:
    """Fetch the current US 10Y Treasury yield and apply a regional spread.

    Falls back to a conservative historical average if live data is
    unavailable or out of plausible bounds.
    """
    try:
        treasury = yf.Ticker(US_TREASURY_TICKER)
        hist = treasury.history(period="5d")

        if not hist.empty:
            rate = float(hist["Close"].iloc[-1])
            if rate > 1.0:
                rate = rate / 100.0

            rate += REGIONAL_ADJUSTMENTS.get(country, 0.0)

            if 0.001 < rate < 0.10:
                return rate

    except Exception:
        pass

    return DEFAULT_RISK_FREE_RATES.get(country, 0.04)


def get_market_risk_premium(country: str = "US") -> float:
    return MARKET_RISK_PREMIUMS.get(country, 0.055)


def calculate_wacc(
    t: yf.Ticker,
    country: str = "US",
    risk_free_rate: Optional[float] = None,
    market_risk_premium: Optional[float] = None,
    assumptions: Assumptions = DEFAULT_ASSUMPTIONS,
) -> Tuple[float, float]:
    """Calculate WACC via CAPM cost of equity and market-implied capital
    structure weights.

    Returns (wacc, beta_used) — the beta is clipped to
    [assumptions.beta_min, assumptions.beta_max], and returning it lets
    callers report the exact figure that fed into the calculation instead
    of the raw, unclipped `info['beta']`.
    """
    info = market_data.get_info(t)

    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate(country)
    if market_risk_premium is None:
        market_risk_premium = get_market_risk_premium(country)

    beta = info.get("beta", 1.0)
    if beta is None or pd.isna(beta):
        beta = 1.0
    beta = max(assumptions.beta_min, min(beta, assumptions.beta_max))

    cost_of_equity = risk_free_rate + beta * market_risk_premium

    market_cap = info.get("marketCap", 0)
    total_debt = info.get("totalDebt", 0) or 0

    if not market_cap:
        current_price = info.get("currentPrice", info.get("regularMarketPrice", 100))
        shares = info.get("sharesOutstanding", 1)
        market_cap = current_price * shares

    interest_expense = market_data.get_interest_expense(t)
    if interest_expense and total_debt > 0:
        cost_of_debt = interest_expense / total_debt
    else:
        cost_of_debt = assumptions.fallback_cost_of_debt

    total_value = market_cap + total_debt
    if total_value > 0:
        equity_weight = market_cap / total_value
        debt_weight = total_debt / total_value
        wacc = (
            equity_weight * cost_of_equity
            + debt_weight * cost_of_debt * (1 - assumptions.tax_rate)
        )
    else:
        wacc = cost_of_equity

    wacc = max(assumptions.wacc_min, min(wacc, assumptions.wacc_max))

    return wacc, beta


def calculate_dcf(
    forecasts: List[float],
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
) -> dict:
    """Two-stage DCF: present value of explicit forecasts + Gordon Growth
    terminal value, converted to enterprise/equity/per-share value.
    """
    pv_fcf = 0.0
    for year, fcf in enumerate(forecasts, start=1):
        pv_fcf += fcf / ((1 + wacc) ** year)

    last_fcf = forecasts[-1]
    terminal_value = last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** len(forecasts))

    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value - net_debt
    price_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0.0

    return {
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "price_per_share": price_per_share,
        "pv_fcf": pv_fcf,
        "pv_terminal": pv_terminal,
    }


@dataclass
class ValuationResult:
    ticker: str
    company_name: str
    country: str
    price_currency: str
    financial_currency: str
    target_currency: str

    current_price_original: Optional[float]
    current_price: Optional[float]

    historical_fcf: pd.Series
    historical_fcf_converted: pd.Series
    forecasts: List[float]
    forecasts_converted: List[float]
    cagr: float

    risk_free_rate: float
    market_risk_premium: float
    wacc: float
    beta: float

    shares_outstanding: float
    total_debt_converted: float
    total_cash_converted: float
    net_debt_converted: float

    pv_fcf: float
    pv_terminal: float
    enterprise_value: float
    equity_value: float
    fair_price: float


def price_stock(ticker_symbol: str, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> ValuationResult:
    """Run the full DCF valuation pipeline for a ticker and return the
    result as data — no printing. Use `easyvaluator.report.print_report`
    to render it.
    """
    ticker_symbol = ticker_symbol.upper()
    t = market_data.get_ticker(ticker_symbol)
    info = market_data.get_info(t)

    company_name = info.get("longName", ticker_symbol)
    # Market price and financial statements are not always denominated in
    # the same currency (e.g. ADRs) — convert each using its own currency.
    price_currency = info.get("currency", "USD")
    financial_currency = info.get("financialCurrency", price_currency)
    current_price_original = info.get("currentPrice", info.get("regularMarketPrice"))

    country = market_data.get_country(t)
    current_price = fx.convert(current_price_original, price_currency, assumptions.target_currency)

    historical_fcf = get_historical_fcf(t)
    historical_fcf_converted = historical_fcf.apply(
        lambda x: fx.convert(x, financial_currency, assumptions.target_currency)
    )

    forecasts, cagr = forecast_fcf(historical_fcf, years=assumptions.forecast_years, assumptions=assumptions)
    forecasts_converted = [
        fx.convert(value, financial_currency, assumptions.target_currency) for value in forecasts
    ]

    risk_free = get_risk_free_rate(country)
    mrp = get_market_risk_premium(country)
    wacc, beta = calculate_wacc(t, country, risk_free, mrp, assumptions)

    shares_outstanding = info.get("sharesOutstanding", 1)
    total_debt = info.get("totalDebt", 0) or 0
    total_cash = info.get("totalCash", 0) or 0
    net_debt = total_debt - total_cash

    total_debt_converted = fx.convert(total_debt, financial_currency, assumptions.target_currency)
    total_cash_converted = fx.convert(total_cash, financial_currency, assumptions.target_currency)
    net_debt_converted = fx.convert(net_debt, financial_currency, assumptions.target_currency)

    dcf_result = calculate_dcf(forecasts, wacc, assumptions.terminal_growth, shares_outstanding, net_debt)

    pv_fcf_converted = fx.convert(dcf_result["pv_fcf"], financial_currency, assumptions.target_currency)
    pv_terminal_converted = fx.convert(dcf_result["pv_terminal"], financial_currency, assumptions.target_currency)
    enterprise_value_converted = fx.convert(
        dcf_result["enterprise_value"], financial_currency, assumptions.target_currency
    )
    equity_value_converted = fx.convert(dcf_result["equity_value"], financial_currency, assumptions.target_currency)
    fair_price = fx.convert(dcf_result["price_per_share"], financial_currency, assumptions.target_currency)

    return ValuationResult(
        ticker=ticker_symbol,
        company_name=company_name,
        country=country,
        price_currency=price_currency,
        financial_currency=financial_currency,
        target_currency=assumptions.target_currency,
        current_price_original=current_price_original,
        current_price=current_price,
        historical_fcf=historical_fcf,
        historical_fcf_converted=historical_fcf_converted,
        forecasts=forecasts,
        forecasts_converted=forecasts_converted,
        cagr=cagr,
        risk_free_rate=risk_free,
        market_risk_premium=mrp,
        wacc=wacc,
        beta=beta,
        shares_outstanding=shares_outstanding,
        total_debt_converted=total_debt_converted,
        total_cash_converted=total_cash_converted,
        net_debt_converted=net_debt_converted,
        pv_fcf=pv_fcf_converted,
        pv_terminal=pv_terminal_converted,
        enterprise_value=enterprise_value_converted,
        equity_value=equity_value_converted,
        fair_price=fair_price,
    )
