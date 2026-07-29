"""Deterministic, self-contained HTML backtest tearsheet."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from quant_earning_edge.backtest import BacktestResult
    from quant_earning_edge.evaluation.report import PerformanceReport
    from quant_earning_edge.evaluation.strategy_gate import Phase4GateEvaluation


class HtmlTearsheetWriter:
    """Render audited metrics and equity without remote or time-varying assets."""

    def write(
        self,
        *,
        report: PerformanceReport,
        result: BacktestResult,
        output: Path,
    ) -> None:
        """Write immutable HTML tied to the report and engine input hashes."""
        if report.input_sha256 != result.input_sha256:
            raise ValueError("report and backtest result input hashes differ")
        encoded = self._render(report=report, result=result).encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"HTML tearsheet collision at {output}") from None

    @staticmethod
    def _render(*, report: PerformanceReport, result: BacktestResult) -> str:
        metrics = (
            ("Gross Sharpe", report.gross_sharpe),
            ("Net Sharpe", report.net_sharpe),
            ("Annualized return", report.annualized_return),
            ("Max drawdown", report.max_drawdown),
            ("Hit rate", report.hit_rate),
            ("Average gross exposure", report.average_gross_exposure),
            ("Turnover", report.turnover),
        )
        metric_rows = "".join(
            f"<tr><th>{html.escape(label)}</th><td>{value:.6f}</td></tr>"
            for label, value in metrics
        )
        cost_rows = "".join(
            "<tr>"
            f"<td>{html.escape(item.component)}</td>"
            f"<td>{item.dollars:.6f}</td>"
            f"<td>{item.marginal_sharpe_loss:.6f}</td>"
            "</tr>"
            for item in report.cost_attribution
        )
        polyline = _equity_polyline([item.net_equity for item in result.daily])
        title = f"quant_earning_edge report {report.sha256[:12]}"
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{title}</title><style>"
            "body{font-family:system-ui;margin:2rem;max-width:960px}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0}"
            "th,td{border:1px solid #bbb;padding:.45rem;text-align:right}"
            "th:first-child,td:first-child{text-align:left}"
            "svg{width:100%;height:240px;background:#f7f8fa}"
            "</style></head><body>"
            f"<h1>{title}</h1>"
            f"<p>Engine: {html.escape(report.engine)} · "
            f"Input SHA-256: <code>{report.input_sha256}</code></p>"
            "<h2>Headline metrics</h2><table><tbody>"
            f"{metric_rows}</tbody></table>"
            "<h2>Net equity</h2>"
            '<svg viewBox="0 0 900 220" role="img" aria-label="Net equity curve">'
            f'<polyline fill="none" stroke="#1665d8" stroke-width="3" points="{polyline}"/>'
            "</svg><h2>Cost attribution</h2><table><thead><tr>"
            "<th>Component</th><th>Dollars</th><th>Marginal Sharpe loss</th>"
            f"</tr></thead><tbody>{cost_rows}</tbody></table>"
            "<p>This report is model evidence, not a claim of live performance.</p>"
            "</body></html>"
        )


class Phase4HtmlTearsheetWriter:
    """Render fold and cohort diagnostics from canonical Phase 4 evidence."""

    def write(self, *, report: Phase4GateEvaluation, output: Path) -> None:
        encoded = self._render(report).encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise RuntimeError(f"Phase 4 HTML tearsheet collision at {output}") from None

    @staticmethod
    def _render(report: Phase4GateEvaluation) -> str:
        overall = report.overall
        headline_rows = "".join(
            f"<tr><th>{html.escape(label)}</th><td>{value:.6f}</td></tr>"
            for label, value in (
                ("Net Sharpe", overall.net_sharpe),
                ("Gross Sharpe", overall.gross_sharpe),
                ("Annualized return", overall.annualized_return),
                ("Max drawdown", overall.max_drawdown),
                ("Hit rate", overall.hit_rate),
                ("Turnover", overall.turnover),
            )
        )
        fold_rows = "".join(
            "<tr>"
            f"<td>{item.fold_index}</td>"
            f"<td>{item.test_start_date.isoformat()}</td>"
            f"<td>{item.test_end_date.isoformat()}</td>"
            f"<td>{item.trade_count}</td>"
            f"<td>{item.net_sharpe:.6f}</td>"
            f"<td>{item.annualized_return:.6f}</td>"
            f"<td>{item.max_drawdown:.6f}</td>"
            "</tr>"
            for item in report.walk_forward.folds
        )
        cohort_rows = "".join(
            "<tr>"
            f"<td>{html.escape(item.dimension)}</td>"
            f"<td>{html.escape(item.value)}</td>"
            f"<td>{item.trade_count}</td>"
            f"<td>{item.session_count}</td>"
            f"<td>{item.total_net_pnl:.6f}</td>"
            f"<td>{item.mean_net_return:.6f}</td>"
            f"<td>{item.net_sharpe:.6f}</td>"
            f"<td>{item.hit_rate:.6f}</td>"
            f"<td>{'n/a' if item.payoff is None else f'{item.payoff:.6f}'}</td>"
            "</tr>"
            for item in report.cohorts
        )
        title = f"quant_earning_edge Phase 4 {report.sha256[:12]}"
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{title}</title><style>"
            "body{font-family:system-ui;margin:2rem;max-width:1200px}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0}"
            "th,td{border:1px solid #bbb;padding:.45rem;text-align:right}"
            "th:first-child,td:first-child{text-align:left}"
            ".pass{color:#176b35}.fail{color:#a12622}"
            "</style></head><body>"
            f"<h1>{title}</h1>"
            f"<p>Input SHA-256: <code>{overall.input_sha256}</code><br>"
            f"Strategy: <code>{report.strategy_sha256}</code><br>"
            f"Assembly manifest: <code>{report.assembly_manifest_sha256}</code><br>"
            f"Walk-forward run: <code>{report.walkforward_run_sha256}</code><br>"
            f"Optuna study: <code>{report.hyperparameter_study_sha256}</code></p>"
            "<h2>Headline metrics</h2><table><tbody>"
            f"{headline_rows}</tbody></table>"
            "<h2>Walk-forward folds</h2><table><thead><tr>"
            "<th>Fold</th><th>Start</th><th>End</th><th>Trades</th>"
            "<th>Net Sharpe</th><th>Annualized return</th><th>Max drawdown</th>"
            f"</tr></thead><tbody>{fold_rows}</tbody></table>"
            "<h2>Cohort diagnostics</h2><table><thead><tr>"
            "<th>Dimension</th><th>Value</th><th>Trades</th><th>Sessions</th>"
            "<th>Net P&amp;L</th><th>Mean net return</th><th>Net Sharpe</th>"
            f"<th>Hit rate</th><th>Payoff</th></tr></thead><tbody>{cohort_rows}</tbody></table>"
            "<p>IV regime is reported as <code>unavailable</code> when no causal "
            "point-in-time options classification exists; it is never inferred.</p>"
            "<p>This report is research evidence, not a claim of live performance.</p>"
            "</body></html>"
        )


def _equity_polyline(values: list[float]) -> str:
    low = min(values)
    high = max(values)
    span = high - low
    denominator = max(len(values) - 1, 1)
    points = []
    for index, value in enumerate(values):
        x = index / denominator * 900
        y = 200 - ((value - low) / span * 180 if span else 90)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)
