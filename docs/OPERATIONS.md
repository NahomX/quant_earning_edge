# Phase 1 operations

All commands fail with a nonzero exit code when configuration, credentials, or
point-in-time data contracts are incomplete. Provider secrets are read from the
process environment first and an optional `.env` file second.

## Configure

```powershell
Copy-Item .env.example .env
# Fill POLYGON_API_KEY and FINNHUB_API_KEY. Never commit .env.
```

The committed universe thresholds live in `configs/universe/default.yaml`.

## Ingest provider data

```powershell
uv run qee ingest earnings --start 2026-07-01 --end 2026-07-31
uv run qee ingest bars --symbol AAPL --start 2021-07-01 --end 2026-07-31
uv run qee ingest corporate-actions --start 2021-07-01 --end 2026-07-31
uv run qee ingest minute-bars `
  --symbol AAPL `
  --start-at 2026-07-28T04:00:00-04:00 `
  --end-at 2026-07-28T09:25:00-04:00 `
  --event-date 2026-07-28
```

These commands write the raw response to bronze before writing validated,
partitioned silver Parquet.

The corporate-action command uses Polygon/Massive's current `/stocks/v1/splits`
and `/stocks/v1/dividends` endpoints. Splits partition by execution date and
cash dividends by ex-dividend date. Provider event IDs are required and
duplicate IDs or out-of-range results abort ingestion.

## Register stable DuckDB views

After silver data exists, validate every artifact against its declared Arrow
schema and expose stable SQL names:

```powershell
uv run qee data register-views --database .\data\research.duckdb
```

The default requires all five datasets and creates `silver_daily_bars`,
`silver_earnings_events`, `silver_stock_splits`, and
`silver_cash_dividends`, plus `silver_minute_bars`. Use repeated `--dataset`
options to register a strict subset. Missing datasets and schema drift fail
before a view is replaced.

## Compute point-in-time price features

Supply every required silver daily-bar partition explicitly. The loader resolves
only revisions ingested by `--observed-at`, discards sessions after
`--asof-date`, and fails on missing history:

```powershell
uv run qee features compute `
  --asof-date 2026-07-27 `
  --observed-at 2026-07-27T21:00:00Z `
  --bars-file .\data\silver\asset_class=us-equity\dataset=daily-bars\date=2026-07-27\part-<hash>.parquet `
  --symbol AAPL `
  --feature return_1d `
  --feature return_5d `
  --feature return_20d `
  --feature realized_vol_20d `
  --feature realized_vol_60d `
  --feature distance_to_vwap_20d
```

Repeat `--bars-file` for the complete lookback. Gold output is long-form and
keyed by symbol, as-of date, and feature name. Every value records the feature
implementation hash, a hash of only its allowed PIT inputs, and the computation
cutoff. Registered feature property tests append arbitrary future observations
and require exact output and lineage equality.

The registered baseline also includes Kalman-filtered 7/30-session volume,
relative volume, RSI(14), MACD 12/26/9 histogram, distance to the 252-session
high, earnings timing, days since the prior report, and prior EPS surprise.
Event features additionally require repeatable `--candidate-file` and
`--earnings-file` inputs plus `--target-date`. `--asof-date` remains the
prior-close feature boundary; `--target-date` is the next trading session. The
current event contributes timing only; reported EPS is read exclusively from
earlier events observed by the cutoff.

For `premarket_gap_pct`, also repeat `--minute-file` with the target-date
minute-bar artifact and provide `--target-date`. The loader uses only bars whose
full one-minute window completed by `--observed-at`; an aggregate beginning at
the cutoff is excluded as incomplete. The feature is the latest completed
pre-market close divided by the frozen prior close, minus one.

## Materialize forward labels and training data

Labels are intentionally computed only after all five future sessions exist:

```powershell
uv run qee labels compute `
  --asof-date 2026-07-27 `
  --observed-at 2026-08-04T22:00:00Z `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --bars-file .\data\silver\asset_class=us-equity\dataset=daily-bars\date=2026-07-27\part-<hash>.parquet `
  --symbol AAPL
```

Repeat `--bars-file` through the fifth subsequent market session. The three
labels are next-session open-to-close, next-session close versus the as-of
close, and fifth-session close versus the as-of close. Offsets come only from
the explicit market-session file, never weekdays or calendar-day arithmetic.

Join complete feature and label key sets:

```powershell
uv run qee labels assemble `
  --feature-file .\data\gold\feature_group=price\month=2026-07\part-<hash>.parquet `
  --label-file .\data\gold\feature_group=forward-labels\month=2026-07\part-<hash>.parquet `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --assembled-at 2026-08-04T23:00:00Z
```

Assembly rejects missing keys, incomplete feature vectors, mixed code versions,
mixed input lineage, schema drift, non-next-session targets, and features
computed at or after the target open. Features and labels remain separate
artifacts; only this immutable training table combines them.

## Plan purged walk-forward folds

Build the split plan from one or more assembled training partitions:

```powershell
uv run qee backtest plan-splits `
  --dataset-file .\data\gold\feature_group=training-dataset\month=2026-01\part-<hash>.parquet `
  --dataset-file .\data\gold\feature_group=training-dataset\month=2026-02\part-<hash>.parquet `
  --output .\data\manifests\backtest\walk-forward.json `
  --minimum-train-sessions 504 `
  --test-sessions 63 `
  --embargo-sessions 5
```

The inputs are concatenated in sorted path order and every fold stores indices
into that stable order. The manifest hashes each input file and records the
exact configuration, embargo dates, and train/test indices. A training row is
included only when its full label horizon ends strictly before the first test
session. Re-running against the same content is idempotent; an existing
different manifest at the requested path is rejected.

## Run and evaluate a daily backtest ledger

Create a JSON specification with `initial_cash`, an increasing `sessions`
array, daily `marks`, and fully specified `trades`. Every trade requires a
unique ID, long/short side, entry/exit session and price, integer shares,
entry/exit ADV, and holding-session count. The engine requires a mark for every
session during which each position is held.

Run the reconciled vectorbt ledger and standardized evaluation:

```powershell
uv run qee backtest run-ledger `
  --spec-file .\backtest-spec.json `
  --output .\data\manifests\backtest\performance-report.json `
  --tearsheet-output .\data\manifests\backtest\performance-report.html `
  --bootstrap-resamples 10000 `
  --seed 20260427 `
  --env-file .\.env
```

The engine values gross and net portfolios separately, applies the documented
commission, half-spread, square-root impact, borrow, and long sell-stop
slippage assumptions, and aborts unless both final and daily accounting
reconcile. The output includes the semantic input hash, vectorbt version,
gross/net Sharpe, annualized return, max drawdown, hit rate, payoff, exposure,
turnover, sequential per-component Sharpe loss, and deterministic 95%
trade-resampled confidence intervals. A one-trade diagnostic run has no
bootstrap interval because it cannot estimate dispersion.

Every ledger, event-plan backtest, and Phase 4 aggregation is fail-closed on
MLflow tracking. The run records the exact package source-tree hash, semantic
backtest input hash, hashes of every source artifact, cost-model parameters,
engine version, bootstrap settings and seed, plus the canonical report. It
uses `MLFLOW_TRACKING_URI` when configured; otherwise it creates an
output-local `.mlflow` store and `.mlflow-artifacts` directory for an auditable
offline run.

The optional HTML output is self-contained and deterministic: it reads the
same reconciled result as the JSON report, embeds no remote assets or current
timestamps, and rejects a pre-existing different file.

## Verify the published momentum benchmark gate

The Phase 3 exit gate is not a free-form comparison. Prepare:

- a locally pinned copy of the publication/source artifact;
- a canonical reference JSON containing its SHA-256, HTTPS source URL,
  published net Sharpe, the historical-membership source URL, the exact
  build-spec SHA-256, tolerance no greater than `0.1`, and at least 252
  required sessions;
- an authoritative session file covering the 60-session lookback and all
  post-signal exits;
- adjusted daily-bar Parquet partitions; and
- point-in-time membership Parquet with the exact non-null schema `symbol:
  string`, `effective_from: date32`, `effective_through: date32`. Intervals for
  a symbol must not overlap.

Copy `configs/evaluation/momentum_build.example.json`, set the signal window and
the methodology documented by the publication, canonicalize it, and build the
ledger:

```powershell
uv run qee evaluation build-momentum-baseline `
  --build-spec .\configs\evaluation\momentum_build.json `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --universe-artifact .\data\benchmarks\historical-spy-components.parquet `
  --daily-bar-file .\data\silver\daily-bars\year=2024\part-<hash>.parquet `
  --daily-bar-file .\data\silver\daily-bars\year=2025\part-<hash>.parquet `
  --trade-plan-output .\data\benchmarks\momentum-trade-plan.json `
  --manifest-output .\data\benchmarks\momentum-baseline.json
```

The builder ranks only members effective on each signal date, uses exactly 61
closes for the 60-session return, sizes from the known signal-date close and
20-session ADV, enters at the next session's open, and exits after the
precommitted holding interval. The generated manifest binds the methodology,
calendar, membership, every bar partition, trade plan, and semantic vectorbt
input. Run that trade plan through `backtest run-ledger` to create the
standardized performance report and MLflow provenance.

Then evaluate:

```powershell
uv run qee evaluation momentum-benchmark-gate `
  --reference-spec .\data\benchmarks\momentum-reference.json `
  --reference-artifact .\data\benchmarks\published-source.pdf `
  --build-spec .\configs\evaluation\momentum_build.json `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --daily-bar-file .\data\silver\daily-bars\year=2024\part-<hash>.parquet `
  --daily-bar-file .\data\silver\daily-bars\year=2025\part-<hash>.parquet `
  --baseline-manifest .\data\benchmarks\momentum-baseline.json `
  --universe-artifact .\data\benchmarks\historical-spy-components.parquet `
  --trade-plan .\data\benchmarks\momentum-trade-plan.json `
  --performance-report .\data\benchmarks\momentum-performance.json `
  --output .\data\benchmarks\momentum-gate.json
```

The command rebuilds the trade plan and manifest from those pinned sources,
then reruns the standardized daily vectorbt engine. It rejects any changed
publication, methodology, calendar, universe, bar partition, trade plan,
manifest, or performance value (including bootstrap evidence). Only then does
it persist the comparison verdict; it exits `1` when the absolute net-Sharpe
difference exceeds the committed tolerance. Fixture results cannot satisfy the
credentialed Phase 3 gate.

Multi-session specs omit `entry_at`/`exit_at` and require daily marks. Earnings
open-to-close specs provide offset-aware entry and exit timestamps on the same
declared session, set `holding_sessions` to zero, and supply no daily marks.
The CLI automatically selects the timestamped vectorbt engine, executes
separate entry/exit orders, and aggregates the reconciled result into the same
daily evaluation contract.

## Portfolio-construction contract

The Phase 4 constructor ranks candidates deterministically by absolute model
score and symbol, then applies quarter-Kelly sizing from at most the latest 60
realized session outcomes. During the first 20 prior sessions it uses the
precommitted 1% per-position calibration allocation, still subject to every
sector and gross cap. This bounded warm-up prevents a zero-risk cold-start
deadlock without inventing a Kelly estimate. Starting with the 21st session,
quarter-Kelly is used; a nonpositive or one-sided estimate produces a zero-risk
plan. Only outcomes closed strictly before the explicit decision date
participate. Every plan records `sizing_mode` and the effective
`per_position_weight`.

Integer-share rounding is always downward. The resulting plan enforces at most
5% per position, 20% per sector, and 50% gross exposure. The effective weights
can therefore be slightly below caps but never above them.

## Tune and train purged LightGBM folds

After assembling training partitions and writing the matching split plan:

```powershell
uv run qee model tune-walkforward `
  --dataset-file .\data\gold\feature_group=training-dataset\month=2026-01\part-<hash>.parquet `
  --split-plan .\data\manifests\backtest\walk-forward.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --study-database .\data\models\earnings-v1\optuna.sqlite3 `
  --output .\data\models\earnings-v1\optuna-study.json

uv run qee model train-walkforward `
  --dataset-file .\data\gold\feature_group=training-dataset\month=2026-01\part-<hash>.parquet `
  --split-plan .\data\manifests\backtest\walk-forward.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --hyperparameter-study .\data\models\earnings-v1\optuna-study.json `
  --output-dir .\data\models\earnings-v1
```

The tuning command runs the strategy-configured Optuna trial count (capped at
200) with a seeded TPE sampler and median pruner. It can resume the exact study
from SQLite. Each objective evaluation fits only the purged inner-training
rows, ranks the inner-validation candidates by probability, takes at most the
configured top-K candidates above 0.5, and maximizes mean annualized validation
Sharpe across folds. Outer test indices never participate in parameter
selection.

Both commands hash every dataset before parsing and require exact agreement
with the split plan. The immutable study artifact records every completed or
pruned trial, package versions, complete training contract, winner, selected
parameters, and their hashes. Training rejects a study from different data,
plan, feature order, label, threshold, seed, top-K, or trial count.
For `earnings_v1`, the strategy config selects
`forward_1d_open_to_close`: the next session's adjusted close divided by that
same session's adjusted open, minus one. This exactly matches the event
backtest's next-open to next-close holding window. The workflow and proof queue
reject a model trained against a different label, even if its features match.

Each fold reserves the latest 20% of its training sessions for early stopping
and purges an additional five sessions plus overlapping label horizons before
that validation block. The declared outer test indices are used only for OOS
probabilities and realized-label evidence.

Every booster is stored separately under its model hash. Canonical run JSON
records the plan and dataset hashes, exact feature order, threshold, seed,
Optuna study and parameter hashes, LightGBM version, best iterations, partition
counts, and OOS row keys.
Each fold also records mean absolute SHAP contribution per feature, calculated
only from that fold's OOS rows. The expected feature-plus-bias contribution
shape is validated before evidence is written.

## Refit a cutoff-safe production model

After the OOS strategy gate has passed, refit the future-scoring booster with
an exclusive label-closure cutoff:

```powershell
uv run qee model train-production `
  --dataset-file .\data\gold\feature_group=training-dataset\month=2026-01\part-<hash>.parquet `
  --training-cutoff 2026-07-28 `
  --phase4-gate .\data\evaluation\phase4-gate.json `
  --split-plan .\data\manifests\backtest\walk-forward.json `
  --hyperparameter-study .\data\models\earnings-v1\optuna-study.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --output-dir .\data\models\earnings-v1-production
```

Rows are eligible only when both `asof_date` and `horizon_end_date` precede
the cutoff. The latest 20% of eligible sessions are held out for early
stopping, with five sessions and overlapping label horizons purged before
that block. The command writes a content-addressed booster and canonical
evidence containing the source hashes, exact feature order, causal date
boundaries, partition counts, seed, LightGBM version, and model hash.
The exact Optuna-selected parameters and immutable study SHA-256 are embedded
as well. Production refitting rejects any study that does not match the
dataset, split plan, and current strategy configuration. The continuous proof
queue rejects production artifacts without this study binding.
The command independently recomputes the documented Phase 4 research and
pre-paper thresholds from canonical report metrics. A false or inconsistent
verdict is rejected; the passing report SHA-256 is embedded in the model
artifact and is mandatory when that artifact is reloaded for live scoring.

## Assemble the complete OOS Phase 4 history

Do not hand-author the production Phase 4 session set. Once the walk-forward
run, point-in-time candidate partitions, adjusted daily bars, and authoritative
calendar are available, materialize every fold and session in one pass:

```powershell
uv run qee evaluation assemble-phase4 `
  --walkforward-run-evidence .\data\models\earnings-v1\run-<hash>.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --candidate-file .\data\gold\event-candidates\for_trade_date=2026-01-05\candidates-<hash>.parquet `
  --candidate-file .\data\gold\event-candidates\for_trade_date=2026-01-06\candidates-<hash>.parquet `
  --daily-bar-file .\data\silver\daily-bars\year=2026\part-<hash>.parquet `
  --initial-cash 100000 `
  --output-dir .\data\manifests\backtest\phase4-plans `
  --manifest-output .\data\manifests\backtest\phase4-assembly.json `
  --aggregation-output .\data\manifests\backtest\phase4-folds.json
```

The assembler requires the candidate key set to equal the complete OOS
prediction ledger. Each candidate must bind the supplied session file and use
the immediately prior authoritative session. Entry and exit evidence comes
only from adjusted open/close bars on the mapped trade session. Missing,
duplicate, unadjusted, or schema-drifted sources fail closed.

Plans are built chronologically across fold boundaries. Equity and realized
session returns flow into the next decision automatically, including explicit
zero-return abstention sessions. The canonical assembly manifest hashes the
strategy, OOS run, calendar, every candidate/bar partition, and every generated
plan. `phase4-gate` re-hashes that complete graph and rejects a fold map whose
ordered plan set differs from the manifest.

The strategy YAML is also the executable cost contract. Commission, market
impact, borrow, and every inclusive price-tier spread floor are translated
directly into the vectorbt cost model during assembly and gate replay. The same
values are recorded in MLflow; a separately hard-coded research cost model is
not permitted.

## Plan and evaluate one OOS event session (diagnostic)

Prepare a strict JSON object containing `equity`, the session's OOS
`predictions`, matching `observations`, and prior `outcomes`. Observations
separate the sizing price and timestamp known at decision time from later
entry/exit execution evidence. Every observation also declares the authoritative
`event_timing` (`bmo` or `amc`). `iv_regime` may be `low`, `medium`, `high`, or
`unavailable`; use `unavailable` unless a causal point-in-time options
classification actually exists.

```powershell
uv run qee model plan-event-backtest `
  --planning-spec .\event-planning-input.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --walkforward-run-evidence .\data\models\earnings-v1\run-<hash>.json `
  --plan-output .\data\manifests\backtest\event-trades-2026-07-28.json `
  --evaluation-output .\data\manifests\backtest\event-evaluation-2026-07-28.json
```

Only probabilities at or above 0.5 become long candidates. Ranking and sizing
use probability, frozen sector, decision-time price, frozen ADV, equity, and
outcomes closed before the trade date. Realized labels and exit prices cannot
change selection or share counts. The command persists the immutable plan,
runs its timestamped vectorbt orders, and writes the standardized reconciled
evaluation. This is an OOS backtest path; it does not claim live fills.
The supplied probabilities must exactly match rows in the canonical
walk-forward run, which itself must be bound to an Optuna study. Every plan
stores that run hash and all source predictions, including candidates below
the trading threshold. A session where the model selects nothing persists as
an explicit zero-return ledger rather than disappearing from evaluation.

## Aggregate the Phase 4 gates

Use the aggregation JSON emitted by `assemble-phase4`. It contains
`assembly_manifest`, `walkforward_run_evidence`, and `folds` entries declaring
consecutive `fold_index`, test start/end dates, and ordered
`event_plan_files`. Relative paths resolve from the aggregation file:

```powershell
uv run qee evaluation phase4-gate `
  --aggregation-spec .\phase4-folds.json `
  --output .\data\manifests\backtest\phase4-gate.json `
  --tearsheet-output .\data\manifests\backtest\phase4-tearsheet.html `
  --bootstrap-resamples 10000
```

Every plan is replayed through the timestamped cost ledger. The next plan's
starting equity must equal the prior plan's final net equity, preventing hidden
capital resets. Sessions and folds must be increasing and non-overlapping.
All event plans must bind the same walk-forward run, every stored probability
must exactly match its OOS row, and the union of plans must cover the complete
OOS prediction ledger exactly once. This prevents manual probability
substitution and omission of model-abstention sessions.
The gate report also stores the exact strategy YAML and Phase 4 assembly
manifest SHA-256 values. Production refitting rejects a different strategy
file, and proof-loop admission repeats that comparison before any session can
be queued. A passing report therefore cannot be reused after changing sizing,
risk, costs, features, labels, or any other committed strategy field.
Every executed trade must have exactly one immutable cohort record. The
canonical report and self-contained HTML include BMO/AMC, sector, and IV-regime
tables with trade/session counts, net P&L, mean return, Sharpe, hit rate, and
payoff. Missing or duplicate cohort mappings fail the gate. An unavailable IV
classification remains visibly `unavailable`; the evaluator never manufactures
an options regime. The Phase 4 MLflow run stores the HTML as a supplemental
artifact alongside the canonical report and source manifest.

The report keeps two separate decisions:

- Phase 4 research gate: net Sharpe at least 0.8, bootstrap lower bound at
  least 0.3, and max drawdown at most 20%.
- Pre-paper backtest gate: net Sharpe above 1.0, bootstrap lower bound above
  0.5, max drawdown below 15%, and at least 75% positive-Sharpe folds.

Synthetic or fixture runs validate this machinery but cannot satisfy either
operational performance gate.

## Replay one order against normalized NBBO and trades

Create a strict JSON input containing `order`, `decision_snapshot`, `quotes`,
and optional `trades` and `config` fields. Every timestamp must include a UTC
offset. The order must freeze `decision_time`, `submitted_at`, `expires_at`,
quantity, ADV, side, aggressiveness, and any explicit limit price.

```powershell
uv run qee backtest replay-nbbo `
  --spec-file .\nbbo-replay-input.json `
  --output .\data\manifests\replay\proof-order.json
```

Aggressive orders consume recorded opposite-side NBBO size. Mid and passive
limits consume the configured conservative probability-weighted fraction of
eligible trade prints; opening-auction prints receive a separate haircut. The
report records fill fragments, partial or missed quantity, predicted and
realized slippage, opening-auction skew, the semantic input hash, and every
quote/trade timestamp actually consumed. It is immutable: an identical retry is
allowed, while different evidence at the same path is rejected.

This command validates normalized replay mechanics only. Fixture output and
synthetic quotes do not count toward the 90-trading-day Phase 6 proof.

## Ingest historical Polygon NBBO and trades

With Polygon Advanced credentials configured, fetch an intended order's exact
SIP-time execution window:

```powershell
uv run qee ingest market-events `
  --symbol AAPL `
  --event-date 2026-07-28 `
  --start-at 2026-07-28T13:30:00Z `
  --end-at 2026-07-28T20:00:00Z
```

Both quote and trade endpoints paginate in ascending SIP timestamp order. Every
raw page is captured in bronze. Silver Parquet retains quote/trade sequence
numbers, exchange fields, participant timestamps, conditions, fractional
sizes, and correction indicators.

Replay normalization accepts only two-sided, non-crossed NBBO with positive
whole-share sizes. It conservatively excludes corrected trades and sub-share
prints, floors fractional shares, and reports every exclusion. Opening-auction
classification must use explicitly supplied provider condition codes; the
normalizer does not guess their meaning.

## Aggregate one replay session

After producing immutable order-level replay evidence, create a strict session
mapping:

```json
{
  "session_date": "2026-07-28",
  "initial_cash": 100000,
  "commission_bps_per_side": 1,
  "evidence_files": ["entry-AAPL.json", "exit-AAPL.json"],
  "round_trips": [
    {
      "trade_id": "earnings-AAPL-20260728",
      "entry_order_id": "entry-AAPL",
      "exit_order_id": "exit-AAPL",
      "side": "long"
    }
  ]
}
```

Then run:

```powershell
uv run qee evaluation replay-session `
  --aggregation-spec .\replay-session-2026-07-28.json `
  --output .\data\manifests\replay\sessions\2026-07-28.json
```

Every order must occur exactly once in an entry/exit pair. The report computes
both fully-filled-order and share-weighted fill rates, adverse slippage
percentiles, opening-auction quantity, commission, and matched-quantity P&L. A
partial-fill mismatch leaves net P&L/return unset and records a reconciliation
break; it is never silently marked to an invented closing price.

An operationally successful day with no candidates uses empty
`evidence_files` and `round_trips`. It produces an explicit zero return and
counts toward uptime without creating a fill-rate denominator.

## Evaluate the 90-session Phase 6 gate

Create an aggregation spec that references an immutable Alpaca session file and
all available daily replay reports:

```json
{
  "session_file": "sessions-<hash>.json",
  "proof_start": "2026-07-28",
  "proof_end": "2026-12-02",
  "initial_cash": 100000,
  "session_report_files": ["sessions/2026-07-28.json"],
  "minimum_session_count": 90,
  "bootstrap_resamples": 10000,
  "seed": 20260427
}
```

```powershell
uv run qee evaluation phase6-gate `
  --aggregation-spec .\phase6-proof.json `
  --output .\data\manifests\replay\phase6-gate.json
```

The 90-session minimum is schema-locked and cannot be lowered in configuration.
Missing authoritative sessions count as downtime and zero return. A daily
reconciliation break makes performance metrics unavailable and fails the gate;
it is not converted to a zero return. Daily starting capital must equal the
prior resolved equity, preventing hidden account resets.

The final verdict requires all of: net Sharpe above 0.8, bootstrap lower bound
above 0.3, fully-filled intended-order rate above 90%, global 90th-percentile
adverse slippage below twice modeled, uptime above 95%, at least 90
authoritative sessions, and no reconciliation breaks. Paper-broker P&L is not
an input.

Cost attribution begins at gross P&L between entry/exit arrival midpoints. It
then subtracts modeled spread, modeled square-root impact, the realized
execution residual (queue/auction effects beyond those models), and commission.
Every daily report reconciles those dollar components to replay fill-price P&L
and net P&L. The 90-session report also records each component's sequential
marginal Sharpe loss.

## Fetch authoritative market sessions

The Alpaca calendar reports real trading dates and session-specific open/close
times, including early closes. With paper-account credentials configured, run:

```powershell
uv run qee calendar sessions --start 2021-01-01 --end 2026-01-01
```

The command writes the raw provider response to bronze and creates an immutable,
content-addressed JSON session file under `manifests/market-calendar`. Use that
file as the explicit input to coverage and trading-date decisions. A locally
constructed weekday list is not acceptable operational evidence.

## Build the next-session universe

A halt snapshot is mandatory even when no symbols are halted:

```json
{
  "asof_date": "2026-07-27",
  "captured_at": "2026-07-27T21:00:00Z",
  "symbols": []
}
```

Run:

```powershell
uv run qee universe build `
  --trade-date 2026-07-28 `
  --asof-date 2026-07-27 `
  --lookback-start 2026-06-01 `
  --halt-snapshot-file .\halts-2026-07-27.json `
  --trigger scheduled
```

The job enumerates tickers as of the prior date, requires 20 sessions and an
exact prior-close bar for every candidate, and persists either a success or
failure manifest. A fixture or manual invocation cannot prove unattended
readiness.

## Build point-in-time earnings candidates

After the frozen universe and silver earnings partitions exist, join them using
the immutable session file and an explicit decision cutoff:

```powershell
uv run qee universe events `
  --trade-date 2026-07-28 `
  --decision-at 2026-07-28T01:30:00Z `
  --universe-snapshot .\data\gold\universe-snapshots\for_trade_date=2026-07-28\snapshot-<hash>.parquet `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --earnings-file .\data\silver\asset_class=us-equity\dataset=earnings-events\date=2026-07-27\part-<hash>.parquet `
  --earnings-file .\data\silver\asset_class=us-equity\dataset=earnings-events\date=2026-07-28\part-<hash>.parquet `
  --split-file .\data\silver\asset_class=us-equity\dataset=stock-splits\date=2026-07-28\part-<hash>.parquet `
  --dividend-file .\data\silver\asset_class=us-equity\dataset=cash-dividends\date=2026-07-28\part-<hash>.parquet
```

The decision timestamp must fall after the prior session close and before the
trade-session open. Only observations whose `ingested_at` is at or before that
cutoff participate. Prior-session after-close (`amc`) and trade-date
before-open (`bmo`) events are eligible; during-market-hours events are
explicitly excluded. Output Parquet contains estimates but never actual results,
and annotates same-trade-date split and dividend event IDs known by the cutoff.
Its manifest hashes every source, records corporate-action overlap counts, and
retains exclusion counts. A valid empty candidate set is persisted instead of
silently inventing a trade.

## Evaluate the five-session gate

Pass dates from an authoritative market calendar:

```powershell
uv run qee universe readiness `
  --trade-date 2026-07-20 `
  --trade-date 2026-07-21 `
  --trade-date 2026-07-22 `
  --trade-date 2026-07-23 `
  --trade-date 2026-07-24
```

The command exits `0` only when the latest persisted manifest for every supplied
date is a scheduled success with a snapshot hash. It otherwise exits `1`.

## Resumable five-year bars backfill

First build a union of historical active tickers. Supply multiple historical
dates so delisted names are not lost by querying only today's listings:

```powershell
uv run qee backfill symbols `
  --asof-date 2021-01-04 `
  --asof-date 2022-01-03 `
  --asof-date 2023-01-03 `
  --asof-date 2024-01-02 `
  --asof-date 2025-01-02 `
  --asof-date 2026-01-02 `
  --output .\symbols-2021-2026.json
```

Run or resume deterministic symbol batches:

```powershell
uv run qee backfill bars `
  --symbols-file .\symbols-2021-2026.json `
  --start 2021-01-01 `
  --end 2026-01-01 `
  --batch-size 100
```

Successful batches are skipped on the next invocation. Failed attempts remain
as evidence. Retries reuse the plan's fixed ingestion timestamp and therefore
cannot create different content-addressed Parquet files solely because time
passed.

Coverage requires an authoritative JSON `sessions` list or line-delimited
market-calendar file:

```powershell
uv run qee backfill coverage `
  --plan-id <sha256-from-backfill-output> `
  --sessions-file .\market-sessions-2021-2026.json
```

Readiness requires every batch, no missing symbol/session pairs, a five-year
date span, and at least 1,200 explicit sessions. It never infers the expected
calendar from the data being audited.

## Evaluate paper-order circuit breakers

Create an ordered JSON observation history. Provider timestamps are fail-closed:
an absent timestamp is treated as stale. A reconciliation age of `0` is the
original close and `1` is T+1 close.

```json
{
  "observations": [
    {
      "session_date": "2026-07-28",
      "replay_source_date": "2026-07-27",
      "evaluated_at": "2026-07-28T13:25:00Z",
      "replay_notional": 100000.0,
      "replay_net_pnl": -500.0,
      "replay_fill_rate": 0.95,
      "polygon_data_observed_at": "2026-07-28T13:24:30Z",
      "alpaca_data_observed_at": "2026-07-28T13:24:45Z",
      "reconciliation_break_age_sessions": null
    }
  ]
}
```

Run:

```powershell
uv run qee monitoring circuit-breakers `
  --spec-file .\breaker-observations.json `
  --output .\breaker-decision.json
```

Build that observation file from completed frozen/replay sessions while
retaining the distinction between evidence date and the new control date:

```powershell
uv run qee monitoring prepare-breaker-controls `
  --control-date 2026-07-29 `
  --evaluated-at 2026-07-29T13:20:00Z `
  --polygon-data-observed-at 2026-07-29T13:19:00Z `
  --alpaca-data-observed-at 2026-07-29T13:19:30Z `
  --frozen-orders .\artifacts\trade_date=2026-07-27\frozen-daily-orders.json `
  --replay-report .\artifacts\trade_date=2026-07-27\replay-session.json `
  --frozen-orders .\artifacts\trade_date=2026-07-28\frozen-daily-orders.json `
  --replay-report .\artifacts\trade_date=2026-07-28\replay-session.json `
  --output .\controls\2026-07-29-breakers.json
```

Frozen portfolio notionals, replay P&L, and fill rates must reconcile by source
session. The latest completed source is explicitly mapped to the new control
date, and both date sequences survive into the breaker decision. Current
provider timestamps remain explicit because fabricating freshness would defeat
the fail-closed control.

For an operational run, capture those timestamps directly from the providers
instead of entering them by hand:

```powershell
uv run qee monitoring probe-freshness `
  --symbol SPY `
  --output .\controls\2026-07-29-freshness.json
```

The probe reads Polygon's latest single-ticker snapshot timestamp and Alpaca's
paper-account clock timestamp, stores both raw responses in bronze, and binds
their payload hashes and request IDs into canonical immutable evidence. It
refuses noncanonical provider hosts and does not substitute the local receipt
time for provider freshness.

Derive the age of any unresolved paper reconciliation break from the
authoritative session calendar. Repeated reports for one session are revisions;
the latest clean revision resolves an earlier break:

```powershell
uv run qee monitoring reconciliation-age `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --control-date 2026-07-29 `
  --evaluated-at 2026-07-29T13:20:00Z `
  --report .\artifacts\trade_date=2026-07-28\paper-reconciliation.json `
  --output .\controls\2026-07-29-reconciliation-age.json
```

The age is `0` on the broken session and increments only across authoritative
completed session closes. Prepare the complete breaker input from immutable
freshness, replay, frozen-order, reconciliation, and calendar evidence:

```powershell
uv run qee monitoring prepare-breaker-evidence `
  --control-date 2026-07-29 `
  --freshness-file .\controls\2026-07-29-freshness.json `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --frozen-orders .\artifacts\trade_date=2026-07-28\frozen-daily-orders.json `
  --replay-report .\artifacts\trade_date=2026-07-28\replay-session.json `
  --reconciliation-report .\artifacts\trade_date=2026-07-28\paper-reconciliation.json `
  --reconciliation-age-output .\controls\2026-07-29-reconciliation-age.json `
  --output .\controls\2026-07-29-breakers.json
```

The unattended worker uses the combined retry-safe form. It discovers complete
prior frozen/replay pairs and every content-addressed reconciliation revision,
then probes both providers at execution time:

```powershell
uv run qee monitoring prepare-breaker-bundle `
  --control-date 2026-07-29 `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --artifact-root .\workflow-artifacts `
  --output-directory .\workflow-artifacts\trade_date=2026-07-29\control-evidence
```

This emits content-addressed freshness, reconciliation-age, and breaker input
paths. Retries may safely create newer bundles; no mutable “latest” pointer is
used. Preparation fails closed for a partial frozen/replay pair, a
non-authoritative artifact date, or missing completed bootstrap evidence.

The command writes immutable evidence and exits `1` when any documented halt is
active: replay loss above 2% of notional, three consecutive applicable fill
rates below 70%, either provider more than 30 minutes stale, or a reconciliation
break still open at T+1 close. A no-trade day has a `null` fill rate and does not
count as a low-fill day.

## Submit and reconcile Alpaca paper orders

Every request needs a unique deterministic `client_order_id`. The submission
command requires an allow decision produced within the preceding 30 minutes:

```json
{
  "client_order_id": "20260728-AAPL-entry",
  "symbol": "AAPL",
  "quantity": 10,
  "side": "buy",
  "order_type": "limit",
  "time_in_force": "day",
  "limit_price": 200.25,
  "extended_hours": false
}
```

```powershell
uv run qee paper submit-order `
  --spec-file .\paper-order.json `
  --breaker-decision .\breaker-decision.json `
  --output .\paper-submission.json
```

The client refuses every host except `https://paper-api.alpaca.markets`, queries
the client ID before posting, rejects a conflicting existing order, captures the
provider response in bronze, and writes immutable submission evidence. It never
targets a live-capital account.

For the production workflow, submit the complete sorted session plan as one
restart-safe operational unit:

```powershell
uv run qee paper submit-batch `
  --spec-file .\paper-order-batch.json `
  --breaker-decision .\breaker-decision.json `
  --output .\paper-batch-submission.json
```

If the process fails after some orders reach Alpaca, rerunning the same command
verifies and reuses those client IDs before continuing. The batch artifact is
written only after every planned order is accounted for. An empty `orders` list
is valid and creates explicit no-trade evidence without calling the broker.

After the close, construct a reconciliation spec with the replay evidence paths
and Alpaca order resources, then run:

```powershell
uv run qee paper reconcile `
  --spec-file .\paper-reconciliation-spec.json `
  --output .\paper-reconciliation.json
```

The command exits `1` for identity, quantity, or non-terminal-order breaks.
Paper-versus-replay price divergence is diagnostic only; Alpaca paper P&L is
explicitly prohibited from entering the strategy proof gate.

For the frozen live workflow, do not copy broker resources into a spec. Fetch
the exact paper orders by their deterministic frozen client IDs:

```powershell
uv run qee paper reconcile-frozen `
  --frozen-orders .\frozen-daily-orders.json `
  --evidence-file .\replay-evidence-entry.json `
  --evidence-file .\replay-evidence-exit.json `
  --output .\paper-reconciliation.json
```

Before any Alpaca request, this command requires the replay evidence IDs and
complete intended-order fields to exactly equal the frozen artifact. It queries
only those verified client IDs from the canonical paper host, writes the same
non-gating reconciliation report, and exits `1` for operational breaks. An
explicit frozen no-trade day needs no evidence files or broker credentials.

The generated unattended workflow writes content-addressed revisions so a
nonterminal order can be observed again later without overwriting its first
state:

```powershell
uv run qee paper reconcile-frozen-revision `
  --frozen-orders .\frozen-daily-orders.json `
  --evidence-file .\replay-evidence-entry.json `
  --evidence-file .\replay-evidence-exit.json `
  --output-directory .\workflow-artifacts\trade_date=2026-07-28
```

An unresolved revision is persisted and exits `1`; a later worker retry writes
a new revision. Pre-open control discovery selects the most recent evaluation
for each session while preserving the earlier evidence.

## Inspect the restart-safe daily workflow

Initialize one append-only state chain for an authoritative trade date:

```powershell
uv run qee workflow initialize `
  --trade-date 2026-07-28
```

Initialization is idempotent. Inspect the current stage at any time:

```powershell
uv run qee workflow status `
  --trade-date 2026-07-28
```

Status exits `1` until all required stages have succeeded and every recorded
artifact still matches its captured SHA-256 digest. It reports the current
stage, retry count, worker ID, and lease expiry. State revisions are append-only
under `manifests/job=daily-paper-workflow/trade_date=<date>` and form a verified
hash chain.

The in-process runner loops through these mandatory stages in order: freeze
inputs, generate the order plan, evaluate breakers, submit paper orders,
capture market events, replay orders, reconcile the session, and evaluate Phase
6 progress. A worker crash leaves a lease; another worker can reclaim the same
stage only after expiry. A stage cannot succeed without at least one immutable
output artifact, so a missing step cannot be silently marked complete.

Generate the complete eight-stage run spec from the four daily control inputs:

First prepare the rolling Phase 6 health and aggregation files:

```powershell
uv run qee evaluation prepare-phase6-controls `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --proof-start 2026-07-28 `
  --proof-end 2026-07-29 `
  --current-trade-date 2026-07-29 `
  --initial-cash 100000 `
  --artifact-root .\workflow-artifacts `
  --health-output .\controls\2026-07-29-health.json `
  --aggregation-output .\controls\2026-07-29-phase6.json
```

The command evaluates workflow health from the append-only state store, reloads
and validates every existing deterministic daily replay report, and includes
the current workflow's expected report path before that file exists. Both
outputs are immutable and idempotent. The pre-run health snapshot will classify
the current date as incomplete; a final terminal verdict must be reevaluated
after the workflow completes with refreshed health evidence.

The preferred daily boundary combines that preparation with generation of the
self-refreshing workflow:

```powershell
uv run qee workflow prepare `
  --trade-date 2026-07-29 `
  --candidate-file .\data\gold\event-candidates\for_trade_date=2026-07-29\candidates-<hash>.parquet `
  --model-evidence .\data\models\earnings-v1-production\production-<hash>.json `
  --model-file .\data\models\earnings-v1-production\production-<hash>.txt `
  --feature-file .\data\gold\feature_group=earnings-v1\month=2026-07\part-<hash>.parquet `
  --prior-replay-file .\workflow-artifacts\trade_date=2026-07-28\replay-session.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --proof-start 2026-07-28 `
  --proof-end 2026-12-02 `
  --initial-cash 100000 `
  --artifact-root .\workflow-artifacts `
  --output .\workflow-inbox\2026-07-29.json `
  --worker-id paper-worker-1
```

This writes immutable pre-run workflow-health and Phase 6 aggregation controls
under the current trade-date artifact directory, includes the deterministic
future daily report path, and emits the complete inbox specification in one
idempotent command. In worker-time automatic mode, the upstream loop freezes
the universe and earnings candidates after 21:00 ET, then waits until 09:10 ET
to capture completed pre-market observations and compute the final feature
vector. The first workflow stage then captures paper-account and Polygon
decision evidence and freezes causal source lineage; the next stage scores
exact feature vectors and freezes linked paper/replay orders. The breaker
stage independently waits until ten minutes before entry, then obtains fresh
provider and reconciliation controls before submission. `--planning-source`
and `--planning-spec` remain exclusive compatibility modes.

```powershell
uv run qee workflow generate `
  --trade-date 2026-07-28 `
  --planning-spec .\live-order-planning.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --breaker-spec .\circuit-breaker-input.json `
  --phase6-spec .\phase6-proof.json `
  --artifact-root .\workflow-artifacts `
  --output .\workflow-inbox\2026-07-28.json `
  --worker-id paper-worker-1
```

For the operational self-refreshing mode, omit `--breaker-spec` and provide the
authoritative calendar:

```powershell
uv run qee workflow generate `
  --trade-date 2026-07-29 `
  --planning-spec .\live-order-planning.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --breaker-session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --phase6-spec .\controls\2026-07-29-phase6.json `
  --artifact-root .\workflow-artifacts `
  --output .\workflow-inbox\2026-07-29.json `
  --worker-id paper-worker-1
```

At the first-stage readiness boundary this mode prepares the breaker bundle,
then passes the single breaker input produced by that successful attempt into
the evaluation stage through a typed artifact binding. Provider secrets remain
environment-only.

The Phase 6 input must include the deterministic daily report path
`<artifact-root>/trade_date=<date>/replay-session.json`. The generator validates
the planning/breaker dates and strategy schema, emits all eight stages in exact
order, binds content-addressed quote/trade and replay outputs, and is
write-once/idempotent. Secrets remain in the worker environment.

To execute rather than only inspect the loop, run the generated spec:

```powershell
uv run qee workflow run `
  --spec-file .\daily-workflow-2026-07-28.json
```

The spec contains the trade date, explicit `trigger` (`scheduled` or `manual`),
worker ID, lease/command timeouts, and exactly one entry for each of the eight
stages in the order above. Each stage entry has one or more `commands`,
expressed as arguments after `qee`, plus either known
`output_files` or `artifact_json_keys` naming path fields in command JSON
output. For example:

```json
{
  "stage": "capture_market_events",
  "commands": [
    {
      "arguments": [
        "ingest", "market-events",
        "--symbol", "AAPL",
        "--event-date", "2026-07-28",
        "--start-at", "2026-07-28T13:30:00Z",
        "--end-at", "2026-07-28T20:00:00Z"
      ],
      "artifact_json_keys": ["quote_path", "trade_path"]
    }
  ],
  "output_files": []
}
```

Only stage-appropriate existing `qee` command prefixes are accepted. Commands
are passed directly to the current Python interpreter without a shell. API
keys, tokens, passwords, and secrets are rejected as command arguments and must
come from the runtime environment. Nonzero command exits become durable failed
stages and are retried on the next loop invocation.

The generated first stage has a timezone-aware `not_before` boundary ten
minutes before the planned entry, preventing early queueing from consuming
stale order controls. Market capture has a second boundary five minutes after
the frozen exit expiry. While either boundary is in the future, a polling
worker leaves the stage pending with zero attempts; it does not manufacture
retry failures. Once ready, the same loop resumes automatically.

The first four stages also carry a `not_after` boundary equal to the planned
entry expiry. If the worker cannot finish pre-open preparation, order planning,
breaker evaluation, and paper submission by that instant, it records one
`WorkflowWindowExpired` failure and never retries that daily order window.
Invalid inbox specifications and expired windows create stable
content-addressed records under
`manifests/job=workflow-worker/attention`; repeated polling does not duplicate
them. Provider or infrastructure failures before expiry remain retryable.

Every stage also declares `maximum_attempts`, `retry_delay_seconds`, and
`maximum_retry_delay_seconds`. Failed command attempts use capped exponential
backoff and are not reclaimed on every inbox poll. When the attempt budget is
consumed, the state is terminalized as `WorkflowRetryExhausted` without adding
a synthetic command attempt, and the worker writes operator-attention evidence.
Generated workflows use shorter retry budgets inside the entry window, 12
attempts for post-close market capture, and 24 attempts for broker
reconciliation.

Evaluate unattended readiness only against an authoritative calendar:

```powershell
uv run qee workflow health `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --start 2026-07-20 `
  --end 2026-07-24 `
  --output .\workflow-health.json
```

The command exits `0` only after five consecutive intact `scheduled` workflow
completions. Manual runs, missing dates, unfinished or failed stages, and
tampered artifacts never count as uptime. The immutable report records each
category, excess retry attempts, scheduled uptime, source state hashes, and the
calendar hash.

The terminal `evaluation phase6-gate` aggregation spec must include
`workflow_health_file`. Its calendar hash, start/end bounds, and complete
authoritative session sequence must exactly match the Phase 6 calendar and
proof window. The terminal uptime threshold is calculated from intact scheduled
workflow completions; replay-report presence is tracked separately and cannot
satisfy the uptime gate.

Run the continuous inbox worker:

```powershell
uv run qee workflow worker `
  --inbox .\workflow-inbox `
  --worker-id paper-worker-1 `
  --poll-seconds 10
```

For the proof, copy
`configs/workflow/phase6_loop.example.json`, replace the content-addressed
calendar, Phase 4 gate, and model paths and proof dates, then attach it to the
same persistent worker:

```powershell
uv run qee workflow worker `
  --inbox .\workflow-inbox `
  --worker-id paper-worker-1 `
  --loop-spec .\configs\workflow\phase6_loop.json `
  --poll-seconds 10 `
  --env-file .\.env
```

This is a state-driven loop, not a wall-clock schedule. On every cycle it finds
the first unfinished authoritative proof session, requires the prior session
to be complete, discovers exactly one immutable event-candidate artifact and
the latest complete causal feature artifact, prepares the rolling controls,
and writes the dated inbox specification. It stages the first proof session
outside the inbox for `admit-proof-start`; after admission, later sessions are
queued automatically. Missing inputs produce a visible `waiting_for_inputs`
heartbeat and no synthetic substitute. A valid zero-candidate earnings day is
queued without a fake feature artifact.

The loop refuses to queue any session unless `phase4_gate_file` independently
reloads as a passing pre-paper report, its SHA-256 exactly matches the digest
embedded in the production-model evidence, and the model training cutoff is no
later than proof start. The gate path is mandatory; a model's self-declared
digest alone is not deployment authority. It also projects model age through
proof end and refuses deployment when that exceeds
`maximum_model_age_calendar_days` (180 by default), so the proof keeps one
frozen model without silently aging beyond its precommitted limit.

When `universe_config` and `halt_snapshot_directory` are present in the loop
spec, the same cycle also runs `workflow prepare-session-inputs`. The halt
directory must contain the retained authoritative file
`halt-YYYY-MM-DD.json` for the prior session. After 21:00 ET the command ingests
the current Finnhub earnings interval and Polygon corporate actions, writes
explicit empty silver partitions when a provider validly reports no events,
builds the scheduled universe and event candidates, and persists all lineage.
For non-empty candidates it waits until 20 minutes before the authoritative
open, fetches adjusted daily history and completed pre-market minutes, and
writes the exact strategy feature vector. It fails after the ten-minute
pre-open safety boundary rather than submitting from a late model decision.

Before starting a real unattended proof, run the fail-closed deployment audit:

```powershell
uv run qee workflow audit-readiness `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --control-date 2026-07-29 `
  --artifact-root .\workflow-artifacts `
  --inbox .\workflow-inbox `
  --worker-id paper-worker-1 `
  --output .\controls\2026-07-29-readiness.json
```

The immutable report contains no credential values. It requires configured
Polygon, Finnhub, and Alpaca credentials; exact Polygon and Alpaca paper hosts;
a live provider-clock/snapshot probe; successful historical Polygon NBBO access
with at least one quote; at least 90 authoritative calendar sessions; writable
data, artifact, and inbox roots; a recent heartbeat from the expected worker;
and a complete prior frozen/replay bootstrap pair. It exits `1` and records
every failed check until all prerequisites pass. Include at least one
authoritative pre-proof bootstrap session in the calendar so the first proof
day has prior control evidence.

After readiness succeeds, exercise the full workflow without placing an order
or contaminating proof state:

```powershell
uv run qee workflow smoke-no-trade `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --smoke-date 2026-07-29 `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --initial-cash 100000 `
  --smoke-root .\smoke\2026-07-29 `
  --output .\controls\2026-07-29-smoke.json `
  --env-file .\.env
```

This creates deterministic empty planning inputs plus one prior no-trade
bootstrap session inside the isolated smoke root, removes wall-clock gates only
for this manual run, and executes all eight real commands with live
provider-freshness controls. It verifies that frozen orders and replay both
contain zero intended orders. Its state, artifacts, worker cycles, and
post-completion Phase 6 report use an isolated data lake; the success evidence
sets `trigger=manual` and `counts_toward_phase6=false`.

The first scheduled proof specification cannot be prepared normally. Stage it
with `workflow prepare --stage-for-admission` to a path outside the worker
inbox, then publish it through the admission boundary:

```powershell
uv run qee workflow admit-proof-start `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --proof-start 2026-07-30 `
  --readiness-file .\controls\2026-07-30-readiness.json `
  --smoke-file .\controls\2026-07-29-smoke.json `
  --workflow-spec .\staging\2026-07-30.json `
  --inbox-output .\workflow-inbox\2026-07-30.json `
  --output .\controls\2026-07-30-proof-start-admission.json
```

Admission strictly reloads canonical evidence, requires every readiness check
to pass for the same calendar and proof-start date, limits readiness age to 30
minutes by default, requires the credentialed manual zero-order smoke to
precede the proof, and accepts only a `scheduled` workflow for the exact first
session. Failure writes neither the inbox specification nor admission evidence.
Successful evidence hashes the calendar, readiness, smoke, and admitted
workflow. Subsequent proof sessions use normal `workflow prepare`.

Use `--once` for a deployment smoke test. Every scan writes a
content-addressed heartbeat under `manifests/job=workflow-worker/cycles`,
including empty inboxes and invalid specifications. A bad spec is reported
without preventing later specs from being inspected.

Every attempted stage command also writes a receipt under `.qee/receipts`.
Receipts retain the workflow hash, stage/attempt, allowed command prefix,
argument hash, exit code, duration, and stdout/stderr hashes. Raw command
streams and credentials are never stored in receipts.

After all eight stages are durably complete, the worker automatically reruns
Phase 6 with refreshed workflow health. It writes content-addressed health,
aggregation, gate, and finalization-manifest files beneath
`trade_date=<date>/post-completion`. The manifest links the complete workflow
state hash to all three artifact hashes. A failed finalizer makes the worker
cycle incomplete and is retried; an intact manifest suppresses duplicate
bootstrap work. This post-completion report is the one that can count the
current session's scheduled completion.

When `workflow worker --env-file` is used, dotenv settings are merged into a
child-only subprocess environment. Existing process variables still take
precedence, the parent environment is not mutated, and secrets never become
workflow command arguments. This makes the Windows launcher equivalent to
systemd's `EnvironmentFile` behavior for provider-using stages.

Deployment assets are in `ops/`: a hardened non-root systemd service with a
restricted writable data path, and a Windows PowerShell launcher using the
project virtual environment. Provider secrets remain in the external env file.

## Materialize replay specifications after market-event ingestion

The replay command does not read raw provider data directly. First create a
strict materialization JSON containing sorted frozen intended orders, one
decision-time snapshot per symbol, and the silver quote/trade files returned by
`ingest market-events`. Then run:

```powershell
uv run qee backtest materialize-replay-specs `
  --materialization-spec .\replay-materialization.json `
  --output-dir .\replay-specs `
  --manifest-output .\replay-materialization-manifest.json
```

The command filters each symbol to the order's submitted/expiry interval,
normalizes corrected, one-sided, and sub-share events conservatively, and emits
one self-contained canonical replay spec per order. The immutable manifest
hashes semantic inputs and source files and records every filtering count.
Relative silver paths resolve against the materialization JSON location. An
explicit empty order set produces auditable no-trade evidence.

## Freeze live-safe daily orders

The live paper workflow must not reuse `plan-event-backtest`, because that
research artifact contains realized entry/exit outcomes. Produce probabilities
from provider-backed market observations and the frozen production booster.
First capture the probability-free source:

```powershell
uv run qee model capture-live-source `
  --trade-date 2026-07-28 `
  --candidate-file .\data\gold\event-candidates\for_trade_date=2026-07-28\candidates-<hash>.parquet `
  --session-file .\data\manifests\market-calendar\sessions-<hash>.json `
  --initial-cash 100000 `
  --prior-replay-file .\workflow-artifacts\trade_date=2026-07-27\replay-session.json `
  --source-output .\live-market-observations.json `
  --evidence-output .\live-market-observations-evidence.json
```

The event-candidate artifact now carries its point-in-time SIC-division
exposure bucket, prior-close sizing price, and frozen 20-session ADV from the
universe snapshot. Capture reads authenticated Alpaca paper equity for
operational evidence and obtains one Polygon two-sided NBBO/last-trade
snapshot per candidate, retaining provider timestamps, payload hashes, and
bronze responses. Sizing equity is not taken from Alpaca: it is chained from
initial proof capital plus clean prior NBBO-replay P&L so Phase 6 accounting
cannot drift. Prior replay reports must be canonical, reconciled,
chronological, and equity-continuous.

Then score the captured source from point-in-time feature artifacts:

```powershell
uv run qee model score-live-planning `
  --source-spec .\live-market-observations.json `
  --model-evidence .\data\models\earnings-v1-production\production-<hash>.json `
  --model-file .\data\models\earnings-v1-production\production-<hash>.txt `
  --feature-file .\data\gold\feature_group=earnings-v1\month=2026-07\part-<hash>.parquet `
  --planning-output .\live-order-planning.json `
  --evidence-output .\live-order-planning-evidence.json
```

The source contains causal sizing observations, historical outcomes closed
before the trade date, frozen NBBO snapshots, and intended execution windows;
it cannot contain a manually entered probability. Then freeze orders:

```powershell
uv run qee model plan-live-orders `
  --planning-spec .\live-order-planning.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --output .\frozen-daily-orders.json `
  --paper-batch-output .\paper-order-batch.json
```

The command rejects realized labels, future observations, same-day Kelly
outcomes, inconsistent timestamps, and paper/replay identity mismatches. It
writes an explicit zero-order batch when closed history is insufficient.
Closing paper orders use Alpaca's closing-auction time-in-force while replay
orders retain their explicit close execution window.

After silver market events exist, avoid copying nested order fields manually:

```powershell
uv run qee backtest materialize-frozen-replay-specs `
  --frozen-orders .\frozen-daily-orders.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --quote-file .\silver-quotes-AAPL.parquet `
  --trade-file .\silver-trades-AAPL.parquet `
  --output-dir .\replay-specs `
  --manifest-output .\replay-materialization-manifest.json
```

The strategy file must match the hash frozen with the orders. Silver files are
grouped by their stored symbol rather than trusted filenames, and their symbol
set must exactly match the selected portfolio.

The generated workflow captures selected symbols without per-ticker operator
commands:

```powershell
uv run qee ingest frozen-market-events `
  --frozen-orders .\frozen-daily-orders.json `
  --manifest-output .\frozen-market-events.json
```

It combines the entry and exit windows for each frozen symbol, refuses to read
until five minutes after every order has expired, writes Polygon responses to
bronze and exact quote/trade files to silver, and persists a manifest linked to
the frozen-order hash. A no-trade artifact writes an empty manifest without
provider credentials.

Replay every spec in the resulting immutable manifest as one restart-safe
batch:

```powershell
uv run qee backtest replay-materialization `
  --manifest-file .\replay-materialization-manifest.json `
  --spec-directory .\replay-specs `
  --output-directory .\replay-evidence `
  --index-output .\replay-evidence\index.json
```

The runner reloads a canonical manifest, verifies each safe leaf filename,
identity, and SHA-256 digest before execution, and writes one immutable replay
evidence file per sorted order plus a canonical evidence index. Repeating the
same batch is idempotent; a changed spec or output collision fails closed. A
no-trade manifest writes an empty evidence index.

Finally, derive the daily Phase 6 lifecycle report without hand-authored
entry/exit mappings:

```powershell
uv run qee evaluation replay-frozen-session `
  --frozen-orders .\frozen-daily-orders.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --evidence-file .\replay-evidence\replay-evidence-entry.json `
  --evidence-file .\replay-evidence\replay-evidence-exit.json `
  --output .\replay-session.json
```

Each generated trade must contain exactly one `-entry` and one `-exit` order.
The command verifies every replay order field against the frozen artifact,
uses frozen portfolio equity and the hashed strategy commission, and fails
closed on unmatched fills. Empty frozen orders produce explicit zero-return
session evidence.

Workflow command specs may consume content-addressed outputs through typed
`artifact_bindings`. A binding names an earlier stage, a repeated long option,
a filename glob, required path markers, and minimum/maximum match counts. The
runner resolves and sorts matching immutable artifact paths and passes them as
direct argv values without invoking a shell. Missing or ambiguous inputs fail
the stage before the command runs. This is how replay materialization selects
`dataset=nbbo-quotes` and `dataset=stock-trades` Parquet outputs from the market
capture stage without predicting their content hashes.
