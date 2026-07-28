# Implementation status

This file is the durable handoff for autonomous development. It records verified
implementation state, not intended or assumed progress.

## Current phase

**Phase 6 execution infrastructure in progress; real-data and operational proofs pending**

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
| Polygon pre-market minute aggregates | Complete | adjusted minute client/silver schema/cutoff tests |
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
| Baseline causal feature set | 16 scalar features complete | price, gap, Kalman volume, momentum, and earnings-event features |
| Deterministic long-form gold feature store | Complete | `features/store.py`, lineage/idempotency tests |
| Session-indexed D+1/D+5 forward label maker | Complete | `labels/forward.py`, explicit-offset tests |
| Leakage-guarded feature/label dataset assembly | Complete | `labels/dataset.py`, pre-open and exact-key tests |
| Purged expanding walk-forward splitter | Complete | strict label-horizon purge and embargo tests |
| Immutable walk-forward plan manifests | Complete | Parquet schema/source hashes, deterministic JSON, CLI test |
| Deterministic 60-session momentum baseline | Complete | causal ranking/future-data invariance tests |
| Decomposed execution cost model | Complete | commission, spread, impact, borrow, and stop tests |
| Vectorbt daily round-trip engine | Complete | long/short mark-to-market, determinism, rejection tests |
| Exact daily cost attribution invariant | Complete | gross minus five cost components equals net on every session |
| Standardized machine-readable evaluation | Complete | headline metrics, 10,000-resample CI, immutable JSON |
| Reproducible backtest CLI | Complete | validated JSON input, semantic input hash, report hash |
| Fold-level walk-forward gate evidence | Complete | ordered OOS folds and explicit 75% positive-Sharpe gate |
| Deterministic HTML tearsheet | Complete | self-contained equity curve, headline and cost tables |
| Fractional-Kelly portfolio constructor | Complete | causal 60-trade history, integer shares, all hard caps |
| Strict earnings-v1 strategy configuration | Complete | registry feature match and cross-field risk/cost validation |
| Deterministic LightGBM walk-forward trainer | Complete | purged internal validation, OOS-only predictions, model hashes |
| Immutable fold-model artifacts and CLI | Complete | content-linked boosters, run evidence, tamper rejection |
| OOS LightGBM SHAP attribution | Complete | mean absolute contribution per feature and fold |
| Timestamped same-session vectorbt ledger | Complete | open/close orders and exact daily cost reconciliation |
| Causal OOS event-trade planning | Complete | frozen sizing inputs, future-invariance, immutable plan and CLI |
| Phase 4 multi-fold gate aggregation | Complete | capital continuity and both documented threshold sets |
| Causally bounded NBBO/trade replay engine | Complete | partial/missed fills, impact, limit queues, auction skew |
| Active NBBO pre-decision-read property gate | Complete | consumed timestamps and pre-boundary mutation invariance |
| Immutable single-order replay evidence and CLI | Complete | semantic input hash, collision-safe JSON, CLI tests |

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

## Phase 2 code gate

The documented registry, monthly gold persistence, scalar baseline feature set,
and D+1/D+5 label maker are implemented. Every one of the 16 registered scalar
features runs through the non-vacuous property suite with 30 generated histories:
the value and PIT input lineage must remain exactly equal after future daily
bars, earnings events, and pre-market observations are appended. Training
assembly independently rejects any feature computed at or after target open.

This proves the code-level no-lookahead contract. It does not prove five-year
data completeness or strategy performance.

## Next implementation slice

Implement normalized Polygon quote/trade ingestion and replay-session
aggregation so the same frozen orders can produce a daily Phase 6 record.

The deterministic replay core now models displayed aggressive liquidity,
probability-weighted mid/passive limit fills, partial and missed fills,
square-root impact, and opening-auction skew. It does not claim exchange queue
reconstruction, and fixture evidence cannot count toward the 90-session proof.

Credentialed calendar/backfill, five-session unattended execution, the
published momentum comparison, five-year Phase 4 evaluation, and 90 observed
replay sessions remain operationally pending. Those gates cannot be replaced
by fixtures or synthetic performance.
