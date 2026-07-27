# Implementation status

This file is the durable handoff for autonomous development. It records verified
implementation state, not intended or assumed progress.

## Current phase

**Phase 1 operational proof pending; Phase 2 feature store in progress**

Phase 0 repository hygiene is complete. The local Python 3.12 environment is
reproducible through `uv.lock`.

## Verified deliverables

| Deliverable | State | Evidence |
|---|---|---|
| Python 3.12 package and development environment | Complete | `pyproject.toml`, `uv.lock` |
| CI lint, format, strict type-check, and test workflow | Complete | `.github/workflows/ci.yml` |
| Bronze/silver/gold deterministic lake layout | Complete | `data/layout.py`, data tests |
| Immutable canonical bronze JSON persistence | Complete | `data/bronze.py`, data tests |
| DuckDB connection/query boundary | Complete | `data/store.py`, persistence test |
| Finnhub earnings client and retry/rate-limit handling | Complete | `data/clients/finnhub.py`, contract tests |
| Polygon aggregate-bars client and pagination | Complete | `data/clients/polygon.py`, contract tests |
| Finnhub bronze-to-silver earnings ingestion | Complete | `data/ingest.py`, ingestion test |
| Earnings silver schema and Parquet writer | Complete | `data/silver.py`, schema/idempotency tests |
| US-equity bars silver schema and writer | Complete | `data/silver.py`, bars ingestion tests |
| Resumable historical backfill tooling | Complete | `data/backfill.py`, resume tests |
| Five-year historical backfill execution | Blocked on provider credentials | No local credentials |
| Explicit-session coverage auditing | Complete | coverage auditor/tests |
| Authoritative market-calendar client and immutable session files | Complete | `data/clients/alpaca.py`, `data/calendar.py`, contract tests |
| Combined PIT earnings/split/dividend candidate audit | Complete | `universe/events.py`, cutoff/timing/overlap tests |
| Polygon splits/dividends ingestion | Complete | current `/stocks/v1` clients, silver schemas, integration tests |
| Stable schema-validated DuckDB silver views | Complete | `data/store.py`, view query tests |
| Point-in-time universe eligibility engine | Complete | `universe/builder.py`, PIT property tests |
| Immutable universe snapshot persistence | Complete | `universe/snapshot.py`, idempotency test |
| Daily universe snapshot production job | Implemented, not operationally proven | `universe/job.py`, production-path tests |
| Five-run unattended readiness evidence | Implemented, awaiting real scheduled runs | manifest store/readiness tests |
| Credential-safe CLI and validated universe config | Complete | `cli.py`, `runtime.py`, CLI/config tests |
| Typed feature registry with code hashes and PIT input boundary | Complete | `features/registry.py`, active property tests |
| Baseline causal feature set | 15 of 15 complete | price, Kalman volume, momentum, and earnings-event features |
| Deterministic long-form gold feature store | Complete | `features/store.py`, lineage/idempotency tests |

## Phase 1 exit gate

Phase 1 is complete only when:

1. Provider clients ingest Polygon market data and Finnhub earnings events.
2. Raw responses are stored immutably in bronze.
3. Validated US-equity bars and events are written to partitioned silver Parquet.
4. DuckDB can query those partitions through stable typed interfaces.
5. A point-in-time eligible-universe snapshot is produced from prior-close data.
6. At least five years of required history are stored.
7. The daily snapshot job runs unattended for five consecutive market days.

Items 6–7 require valid provider access and elapsed observation time. They must
not be marked complete from fixtures or synthetic data.

## Next implementation slice

Build the point-in-time label maker for D+1 open-to-close, D+1 close, and D+5
close horizons, then expose a reproducible feature/label dataset assembly.
Credentialed calendar, backfill, and five-session unattended execution remain
pending. Those operational gates cannot be replaced by fixtures.
