"""Command-line entry point: `python -m easyvaluator TICKER [options]`."""

import argparse
import logging
import sys
from typing import Optional, Sequence

from .assumptions import Assumptions
from .report import print_report
from .valuation import price_stock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="easyvaluator",
        description="Discounted Cash Flow (DCF) stock valuation tool.",
    )
    parser.add_argument("ticker", nargs="?", help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--years", type=int, default=5, help="Forecast horizon in years (default: 5)")
    parser.add_argument("--currency", default="EUR", help="Target output currency (default: EUR)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show warnings when the model falls back on defaults"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.verbose else logging.ERROR,
        format="%(levelname)s: %(message)s",
    )

    ticker = args.ticker or input("Enter stock ticker: ").strip()
    if not ticker:
        parser.error("No ticker symbol provided.")

    assumptions = Assumptions(target_currency=args.currency.upper(), forecast_years=args.years)

    try:
        result = price_stock(ticker, assumptions)
    except Exception as exc:
        print(f"Error analyzing {ticker}: {exc}", file=sys.stderr)
        return 1

    print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
