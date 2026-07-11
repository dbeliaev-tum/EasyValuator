"""Human-readable formatting of a ValuationResult."""

from .valuation import ValuationResult


def print_report(result: ValuationResult) -> None:
    cur = result.target_currency

    print(f"\n{'=' * 60}")
    print(f"DCF Analysis for {result.ticker}")
    print(f"{'=' * 60}\n")

    print(f"Company: {result.company_name}")
    print(f"Country/Region: {result.country}")
    print(f"Price Currency: {result.price_currency}")
    print(f"Financial Statement Currency: {result.financial_currency}")
    if result.current_price_original and result.current_price:
        print(
            f"Current Price: {result.current_price_original:.2f} {result.price_currency} "
            f"= {result.current_price:.2f} {cur}\n"
        )
    else:
        print("Current Price: Not available\n")

    print("Step 1: Historical Free Cash Flow")
    print("-" * 60)
    print(f"(Converted from {result.financial_currency} to {cur})")
    for date, value in result.historical_fcf_converted.items():
        year = date.year if hasattr(date, "year") else date
        print(f"{year}: {value:,.0f} {cur}")
    print()

    print(f"Step 2: Forecast FCF (next {len(result.forecasts_converted)} years)")
    print("-" * 60)
    print(f"Historical CAGR: {result.cagr * 100:.2f}%")
    for year, value in enumerate(result.forecasts_converted, start=1):
        print(f"Year {year}: {value:,.0f} {cur}")
    print()

    print("Step 3: Market Parameters")
    print("-" * 60)
    print(f"Risk-Free Rate ({result.country}): {result.risk_free_rate * 100:.2f}%")
    print(f"Market Risk Premium ({result.country}): {result.market_risk_premium * 100:.2f}%")
    print()

    print("Step 4: Calculate WACC")
    print("-" * 60)
    print(f"Beta: {result.beta:.2f}")
    print(f"WACC: {result.wacc * 100:.2f}%")
    print()

    print("Step 5: Company Financials")
    print("-" * 60)
    print(f"Shares Outstanding: {result.shares_outstanding:,.0f}")
    print(f"Total Debt: {result.total_debt_converted:,.0f} {cur}")
    print(f"Total Cash: {result.total_cash_converted:,.0f} {cur}")
    print(f"Net Debt: {result.net_debt_converted:,.0f} {cur}")
    print()

    print("Step 6: DCF Valuation")
    print("-" * 60)
    print(f"PV of Forecast Period: {result.pv_fcf:,.0f} {cur}")
    print(f"PV of Terminal Value: {result.pv_terminal:,.0f} {cur}")
    print(f"Enterprise Value: {result.enterprise_value:,.0f} {cur}")
    print(f"Equity Value: {result.equity_value:,.0f} {cur}")
    print()

    print("=" * 60)
    print(f"FAIR VALUE: {result.fair_price:.2f} {cur}")
    if result.current_price:
        print(f"CURRENT PRICE: {result.current_price:.2f} {cur}")
        upside = ((result.fair_price - result.current_price) / result.current_price) * 100
        print(f"UPSIDE/DOWNSIDE: {upside:+.1f}%")
    print("=" * 60)
