"""Walk-forward aggregation and deterministic HTML tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

import pytest

from quant_earning_edge.evaluation import (
    HtmlTearsheetWriter,
    PerformanceEvaluator,
    WalkForwardEvaluator,
)
from tests.evaluation.test_report import _result

if TYPE_CHECKING:
    from pathlib import Path


def test_fold_aggregation_exposes_positive_sharpe_gate() -> None:
    base = PerformanceEvaluator(bootstrap_resamples=10).evaluate(_result())
    sharpes = (1.0, 0.5, 0.1, -0.2)
    folds = tuple(
        (
            index,
            date(2025, index + 1, 1),
            date(2025, index + 1, 28),
            replace(base, net_sharpe=sharpe),
        )
        for index, sharpe in enumerate(sharpes)
    )

    report = WalkForwardEvaluator().aggregate(folds)

    assert report.positive_sharpe_fold_count == 3
    assert report.positive_sharpe_fraction == 0.75
    assert report.passes_positive_fold_gate
    assert report.mean_net_sharpe == pytest.approx(0.35)


def test_html_tearsheet_is_immutable_and_self_contained(tmp_path: Path) -> None:
    result = _result()
    report = PerformanceEvaluator(bootstrap_resamples=10).evaluate(result)
    output = tmp_path / "tearsheet.html"
    writer = HtmlTearsheetWriter()

    writer.write(report=report, result=result, output=output)
    writer.write(report=report, result=result, output=output)

    text = output.read_text(encoding="utf-8")
    assert report.input_sha256 in text
    assert "Net equity" in text
    assert "<polyline" in text
    assert "https://" not in text
