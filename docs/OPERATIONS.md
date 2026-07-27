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
