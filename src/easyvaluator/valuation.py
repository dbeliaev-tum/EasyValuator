"""Valuation pipeline: turns a `CompanySnapshot` into a `ValuationResult`.

All arithmetic happens in the company's *financial-statement currency*,
the currency its cash flows are earned in. Only the market capitalisation
(needed for WACC weights) is converted *into* that currency, and only the
final outputs are converted *out of* it to the target currency. Because
every conversion is a single multiplication at one spot rate, the result
doesn't depend on the order of operations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

from . import regions
from .assumptions import DEFAULT_ASSUMPTIONS, Assumptions
from .dcf import discount_cash_flows, sensitivity_table
from .exceptions import ValuationError
from .fcf import estimate_growth, forecast_fcf, growth_path
from .fx import FXConverter
from .market_data import MarketDataProvider, YahooFinanceProvider
from .models import CompanySnapshot, DCFBreakdown, ValuationResult
from .wacc import calculate_wacc

logger = logging.getLogger(__name__)

# A terminal value above this share of EV means the valuation is driven
# almost entirely by the perpetuity assumption — worth flagging.
TERMINAL_VALUE_WARNING_SHARE = 0.80

MAX_PLAUSIBLE_TAX_RATE = 0.50


class Valuator:
    """Runs DCF valuations against a market data provider.

    >>> valuator = Valuator()                     # Yahoo Finance
    >>> result = valuator.value("AAPL")           # doctest: +SKIP
    """

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
        assumptions: Assumptions = DEFAULT_ASSUMPTIONS,
    ) -> None:
        self.provider: MarketDataProvider = provider if provider is not None else YahooFinanceProvider()
        self.assumptions = assumptions
        self.fx = FXConverter(self.provider.get_fx_rate)

    def value(self, symbol: str, *, sensitivity: bool = True) -> ValuationResult:
        snapshot = self.provider.get_company(symbol.strip().upper())
        return self.value_snapshot(snapshot, sensitivity=sensitivity)

    def value_snapshot(self, company: CompanySnapshot, *, sensitivity: bool = True) -> ValuationResult:
        a = self.assumptions
        warnings: list[str] = []

        def warn(message: str) -> None:
            logger.warning("%s: %s", company.symbol, message)
            warnings.append(message)

        fin_ccy = company.financial_currency

        region = regions.classify_region(company.symbol, company.country)
        if region is None:
            warn(f"Region for country {company.country!r} not modelled; using US equity risk premium")
            region = regions.DEFAULT_REGION

        # --- Free cash flow -------------------------------------------------
        historical = company.historical_fcf
        if len(historical) < 2:
            raise ValuationError("At least 2 years of historical FCF are required")
        base_fcf = float(historical.iloc[-1])
        if base_fcf <= 0:
            raise ValuationError(
                f"Latest free cash flow is negative ({base_fcf:,.0f} {fin_ccy}); "
                "an FCF-based DCF is not meaningful for this company"
            )

        growth = estimate_growth(historical, a)
        if growth.note:
            warn(growth.note)
        forecasts = forecast_fcf(base_fcf, growth_path(growth.cagr, a.terminal_growth, a.forecast_years))

        # --- Discount rate --------------------------------------------------
        rf = regions.risk_free_rate(fin_ccy, self.provider.get_us_treasury_yield())
        if not rf.currency_supported:
            warn(f"No risk-free spread modelled for {fin_ccy}; discounting at the US rate")
        if not rf.is_live:
            warn("Live Treasury yield unavailable; using fallback risk-free rate")

        beta = company.beta
        if beta is None:
            warn("Beta unavailable; assuming 1.0")
            beta = 1.0
        elif not a.beta_min <= beta <= a.beta_max:
            warn(f"Beta {beta:.2f} clamped to [{a.beta_min}, {a.beta_max}]")

        if company.interest_expense and company.total_debt > 0:
            cost_of_debt = company.interest_expense / company.total_debt
        else:
            warn(f"Cost of debt unavailable; assuming {a.fallback_cost_of_debt:.1%}")
            cost_of_debt = a.fallback_cost_of_debt

        wacc = calculate_wacc(
            risk_free_rate=rf.rate,
            market_risk_premium=regions.equity_risk_premium(region),
            beta=beta,
            # Market cap is quoted in the listing currency, debt in the
            # statement currency; weights need both in the same unit.
            market_cap=self.fx.convert(company.market_cap, company.price_currency, fin_ccy),
            total_debt=company.total_debt,
            cost_of_debt=cost_of_debt,
            tax_rate=self._tax_rate(company, warn),
            assumptions=a,
        )

        # --- Valuation ------------------------------------------------------
        net_debt = company.total_debt - company.total_cash
        dcf = discount_cash_flows(forecasts, wacc.wacc, a.terminal_growth, company.shares_outstanding, net_debt)
        if dcf.terminal_value_share > TERMINAL_VALUE_WARNING_SHARE:
            warn(f"Terminal value is {dcf.terminal_value_share:.0%} of enterprise value")
        if dcf.equity_value <= 0:
            warn("Net debt exceeds enterprise value; equity value is not positive")

        # --- Convert outputs ------------------------------------------------
        to_target = self.fx.rate(fin_ccy, a.target_currency)
        table = None
        if sensitivity:
            table = sensitivity_table(
                forecasts, wacc.wacc, a.terminal_growth, company.shares_outstanding, net_debt, scale=to_target
            )

        return ValuationResult(
            symbol=company.symbol,
            company_name=company.name,
            region=region,
            price_currency=company.price_currency,
            financial_currency=fin_ccy,
            target_currency=a.target_currency,
            current_price=self.fx.convert(company.price, company.price_currency, a.target_currency),
            historical_fcf=historical * to_target,
            forecast_fcf=[value * to_target for value in forecasts],
            cagr=growth.cagr,
            wacc=wacc,
            dcf=_scale(dcf, to_target),
            terminal_growth=a.terminal_growth,
            shares_outstanding=company.shares_outstanding,
            total_debt=company.total_debt * to_target,
            total_cash=company.total_cash * to_target,
            sensitivity=table,
            warnings=warnings,
        )

    def _tax_rate(self, company: CompanySnapshot, warn: Callable[[str], None]) -> float:
        if self.assumptions.tax_rate is not None:
            return self.assumptions.tax_rate
        reported = company.effective_tax_rate
        if reported is not None and 0 <= reported <= MAX_PLAUSIBLE_TAX_RATE:
            return reported
        warn(f"Effective tax rate unavailable; assuming {self.assumptions.default_tax_rate:.0%}")
        return self.assumptions.default_tax_rate


def _scale(dcf: DCFBreakdown, factor: float) -> DCFBreakdown:
    return replace(
        dcf,
        pv_forecast=dcf.pv_forecast * factor,
        terminal_value=dcf.terminal_value * factor,
        pv_terminal=dcf.pv_terminal * factor,
        enterprise_value=dcf.enterprise_value * factor,
        equity_value=dcf.equity_value * factor,
        value_per_share=dcf.value_per_share * factor,
    )


def price_stock(
    symbol: str,
    assumptions: Assumptions = DEFAULT_ASSUMPTIONS,
    *,
    provider: MarketDataProvider | None = None,
    sensitivity: bool = True,
) -> ValuationResult:
    """Convenience wrapper: value a single ticker."""
    return Valuator(provider, assumptions).value(symbol, sensitivity=sensitivity)
