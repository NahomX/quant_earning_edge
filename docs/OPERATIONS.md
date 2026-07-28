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
