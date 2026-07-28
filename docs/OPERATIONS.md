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
  --seed 20260427
```

The engine values gross and net portfolios separately, applies the documented
commission, half-spread, square-root impact, borrow, and long sell-stop
slippage assumptions, and aborts unless both final and daily accounting
reconcile. The output includes the semantic input hash, vectorbt version,
gross/net Sharpe, annualized return, max drawdown, hit rate, payoff, exposure,
turnover, sequential per-component Sharpe loss, and deterministic 95%
trade-resampled confidence intervals. A one-trade diagnostic run has no
bootstrap interval because it cannot estimate dispersion.

The optional HTML output is self-contained and deterministic: it reads the
same reconciled result as the JSON report, embeds no remote assets or current
timestamps, and rejects a pre-existing different file.

Multi-session specs omit `entry_at`/`exit_at` and require daily marks. Earnings
open-to-close specs provide offset-aware entry and exit timestamps on the same
declared session, set `holding_sessions` to zero, and supply no daily marks.
The CLI automatically selects the timestamped vectorbt engine, executes
separate entry/exit orders, and aggregates the reconciled result into the same
daily evaluation contract.

## Portfolio-construction contract

The Phase 4 constructor ranks candidates deterministically by absolute model
score and symbol, then applies quarter-Kelly sizing from at most the latest 60
realized trade outcomes. It requires at least 20 outcomes containing both wins
and losses; otherwise it returns a zero-risk plan rather than inventing a
payoff estimate. Only outcomes closed strictly before the explicit decision
date participate.

Integer-share rounding is always downward. The resulting plan enforces at most
5% per position, 20% per sector, and 50% gross exposure. The effective weights
can therefore be slightly below caps but never above them.

## Train purged LightGBM folds

After assembling training partitions and writing the matching split plan:

```powershell
uv run qee model train-walkforward `
  --dataset-file .\data\gold\feature_group=training-dataset\month=2026-01\part-<hash>.parquet `
  --split-plan .\data\manifests\backtest\walk-forward.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --output-dir .\data\models\earnings-v1
```

The command hashes every dataset before parsing and requires exact agreement
with the split plan. Each fold reserves the latest 20% of its training sessions
for early stopping and purges an additional five sessions plus overlapping
label horizons before that validation block. The declared test indices are
used only for OOS probabilities and realized-label evidence.

Every booster is stored separately under its model hash. Canonical run JSON
records the plan and dataset hashes, exact feature order, threshold, seed,
LightGBM version, best iterations, partition counts, and OOS row keys.
Each fold also records mean absolute SHAP contribution per feature, calculated
only from that fold's OOS rows. The expected feature-plus-bias contribution
shape is validated before evidence is written.

## Plan and evaluate one OOS event session

Prepare a strict JSON object containing `equity`, the session's OOS
`predictions`, matching `observations`, and prior `outcomes`. Observations
separate the sizing price and timestamp known at decision time from later
entry/exit execution evidence.

```powershell
uv run qee model plan-event-backtest `
  --planning-spec .\event-planning-input.json `
  --strategy-config .\configs\strategies\earnings_v1.yaml `
  --plan-output .\data\manifests\backtest\event-trades-2026-07-28.json `
  --evaluation-output .\data\manifests\backtest\event-evaluation-2026-07-28.json
```

Only probabilities at or above 0.5 become long candidates. Ranking and sizing
use probability, frozen sector, decision-time price, frozen ADV, equity, and
outcomes closed before the trade date. Realized labels and exit prices cannot
change selection or share counts. The command persists the immutable plan,
runs its timestamped vectorbt orders, and writes the standardized reconciled
evaluation. This is an OOS backtest path; it does not claim live fills.

## Aggregate the Phase 4 gates

Create an aggregation JSON whose `folds` entries declare consecutive
`fold_index`, test start/end dates, and ordered `event_plan_files`. Relative
paths resolve from the aggregation file:

```powershell
uv run qee evaluation phase4-gate `
  --aggregation-spec .\phase4-folds.json `
  --output .\data\manifests\backtest\phase4-gate.json `
  --bootstrap-resamples 10000
```

Every plan is replayed through the timestamped cost ledger. The next plan's
starting equity must equal the prior plan's final net equity, preventing hidden
capital resets. Sessions and folds must be increasing and non-overlapping.

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

To execute rather than only inspect the loop, provide a complete JSON run spec
and run:

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

Run the continuous inbox worker:

```powershell
uv run qee workflow worker `
  --inbox .\workflow-inbox `
  --worker-id paper-worker-1 `
  --poll-seconds 10
```

Use `--once` for a deployment smoke test. Every scan writes a
content-addressed heartbeat under `manifests/job=workflow-worker/cycles`,
including empty inboxes and invalid specifications. A bad spec is reported
without preventing later specs from being inspected.

Every attempted stage command also writes a receipt under `.qee/receipts`.
Receipts retain the workflow hash, stage/attempt, allowed command prefix,
argument hash, exit code, duration, and stdout/stderr hashes. Raw command
streams and credentials are never stored in receipts.

Deployment assets are in `ops/`: a hardened non-root systemd service with a
restricted writable data path, and a Windows PowerShell launcher using the
project virtual environment. Provider secrets remain in the external env file.
