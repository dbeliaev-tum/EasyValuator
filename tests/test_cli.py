import json

import pytest

from easyvaluator import Valuator, render_text
from easyvaluator.cli import main


@pytest.fixture
def populated(provider, make_snapshot):
    provider.companies["TEST"] = make_snapshot()
    return provider


def test_text_report(populated, capsys):
    assert main(["test", "--currency", "eur"], provider=populated) == 0
    out = capsys.readouterr().out
    assert "Test Corp (TEST)" in out
    assert "FAIR VALUE PER SHARE" in out
    assert "Sensitivity" in out


def test_json_output_is_valid(populated, capsys):
    assert main(["TEST", "--json", "--no-sensitivity"], provider=populated) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["symbol"] == "TEST"
    assert data["sensitivity"] is None
    assert data["fair_value"] == pytest.approx(data["dcf"]["value_per_share"])
    assert set(data["historical_fcf"]) == {"2021", "2022", "2023", "2024"}


def test_multiple_tickers_with_one_failure(populated, capsys):
    assert main(["TEST", "NOPE", "--json"], provider=populated) == 1
    captured = capsys.readouterr()
    assert len(json.loads(captured.out)) == 1
    assert "NOPE" in captured.err


def test_assumption_flags_are_applied(populated, capsys):
    main(["TEST", "--json", "-y", "7", "-g", "2%", "--tax-rate", "30"], provider=populated)
    data = json.loads(capsys.readouterr().out)
    assert len(data["forecast_fcf"]) == 7
    assert data["terminal_growth"] == pytest.approx(0.02)
    assert data["wacc"]["tax_rate"] == pytest.approx(0.30)


def test_invalid_assumptions_exit_with_usage_error(populated):
    with pytest.raises(SystemExit) as exc:
        main(["TEST", "-g", "10"], provider=populated)  # g above wacc_min
    assert exc.value.code == 2


def test_render_text_lists_warnings(populated, make_snapshot):
    populated.companies["TEST"] = make_snapshot(beta=None)
    text = render_text(Valuator(populated).value("TEST"))
    assert "Model warnings" in text
    assert "Beta unavailable" in text
