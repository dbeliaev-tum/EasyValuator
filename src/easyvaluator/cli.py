"""Command-line entry point: `easyvaluator TICKER [TICKER ...] [options]`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .assumptions import DEFAULT_ASSUMPTIONS, Assumptions
from .exceptions import EasyValuatorError
from .market_data import MarketDataProvider
from .report import render_json, render_text
from .valuation import Valuator

EXIT_OK = 0
EXIT_FAILURE = 1


def _percent(text: str) -> float:
    """Parse '2.5' or '2.5%' as 0.025."""
    try:
        return float(text.rstrip("%")) / 100
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a percentage: {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    d = DEFAULT_ASSUMPTIONS
    parser = argparse.ArgumentParser(
        prog="easyvaluator",
        description="Discounted Cash Flow (DCF) stock valuation from public market data.",
        epilog="Example: easyvaluator AAPL SAP.DE --currency USD --years 7",
    )
    parser.add_argument("tickers", nargs="+", metavar="TICKER", help="ticker symbol(s), e.g. AAPL SAP.DE 7203.T")
    parser.add_argument("-c", "--currency", default=d.target_currency, help="output currency (default: %(default)s)")
    parser.add_argument(
        "-y", "--years", type=int, default=d.forecast_years, help="forecast horizon in years (default: %(default)s)"
    )
    parser.add_argument(
        "-g",
        "--terminal-growth",
        type=_percent,
        default=d.terminal_growth,
        metavar="PCT",
        help=f"perpetual growth rate in percent (default: {d.terminal_growth * 100:g})",
    )
    parser.add_argument(
        "--tax-rate",
        type=_percent,
        default=None,
        metavar="PCT",
        help="override the company's effective tax rate, in percent",
    )
    parser.add_argument("--no-sensitivity", action="store_true", help="skip the WACC x growth sensitivity table")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a report")
    parser.add_argument("-v", "--verbose", action="store_true", help="log model fallbacks as they happen")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None, provider: MarketDataProvider | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        assumptions = Assumptions(
            target_currency=args.currency,
            forecast_years=args.years,
            terminal_growth=args.terminal_growth,
            tax_rate=args.tax_rate,
        )
    except EasyValuatorError as exc:
        parser.error(str(exc))

    valuator = Valuator(provider, assumptions)
    exit_code = EXIT_OK
    json_results = []

    for ticker in args.tickers:
        try:
            result = valuator.value(ticker, sensitivity=not args.no_sensitivity)
        except EasyValuatorError as exc:
            print(f"error: {ticker}: {exc}", file=sys.stderr)
            exit_code = EXIT_FAILURE
            continue

        if args.json:
            json_results.append(json.loads(render_json(result)))
        else:
            print(render_text(result), end="\n\n")

    if args.json:
        payload = json_results[0] if len(args.tickers) == 1 and json_results else json_results
        print(json.dumps(payload, indent=2))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
