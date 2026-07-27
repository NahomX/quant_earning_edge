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
```

These commands write the raw response to bronze before writing validated,
partitioned silver Parquet.

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
