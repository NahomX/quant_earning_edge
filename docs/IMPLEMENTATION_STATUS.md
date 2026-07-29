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
| Session-indexed D+1/D+5 forward label maker | Complete | `labels/forward.py`, including next-session open-to-close target and explicit-offset tests |
| Leakage-guarded feature/label dataset assembly | Complete | `labels/dataset.py`, pre-open and exact-key tests |
| Purged expanding walk-forward splitter | Complete | strict label-horizon purge and embargo tests |
| Immutable walk-forward plan manifests | Complete | Parquet schema/source hashes, deterministic JSON, CLI test |
| Deterministic 60-session momentum baseline | Complete | PIT membership + adjusted-bar builder, causal next-open ledger, immutable source manifest |
| Source-bound Phase 3 momentum comparison gate | Complete | rebuilds methodology/calendar/universe/bars/trades/report before strict ±0.1 verdict |
| Strategy-bound decomposed execution cost model | Complete | validated YAML-to-engine translation for commission, inclusive spread tiers, impact, and borrow; exact MLflow parameters |
| Vectorbt daily round-trip engine | Complete | long/short mark-to-market, determinism, rejection tests |
| Exact daily cost attribution invariant | Complete | gross minus five cost components equals net on every session |
| Standardized machine-readable evaluation | Complete | headline metrics, 10,000-resample CI, immutable JSON |
| Reproducible backtest CLI | Complete | validated JSON input, semantic input hash, report hash |
| Fail-closed MLflow backtest provenance | Complete | code/data/source hashes, cost parameters, seed, engine, and canonical report for all research entry points |
| Fold-level walk-forward gate evidence | Complete | ordered OOS folds and explicit 75% positive-Sharpe gate |
| Deterministic HTML tearsheet | Complete | self-contained equity curve, headline and cost tables |
| Fractional-Kelly portfolio constructor | Complete | bounded 1% cold-start calibration, causal 60-session history, explicit sizing mode, integer shares, all hard caps |
| Strict earnings-v1 strategy configuration | Complete | registry feature match, next-open-to-next-close target alignment, and cross-field risk/cost validation |
| Deterministic nested Optuna search | Complete | seeded TPE, median pruning, resumable SQLite, mean purged-validation Sharpe, 200-trial cap |
| Immutable hyperparameter-selection evidence | Complete | full trial ledger, exact data/plan/config binding, canonical winner and parameter hashes |
| Deterministic LightGBM walk-forward trainer | Complete | purged internal validation, OOS-only predictions, model hashes |
| Immutable fold-model artifacts and CLI | Complete | content-linked boosters, run evidence, tamper rejection |
| OOS LightGBM SHAP attribution | Complete | mean absolute contribution per feature and fold |
| OOS prediction-to-Phase 4 provenance | Complete | canonical run reload, exact probability checks, complete row coverage, shared Optuna/run hashes |
| Automated historical Phase 4 assembly | Complete | exact OOS/candidate key join, authoritative session mapping, adjusted execution bars, chained equity/outcomes, source-and-plan manifest |
| Timestamped same-session vectorbt ledger | Complete | open/close orders and exact daily cost reconciliation |
| Causal OOS event-trade planning | Complete | frozen sizing inputs, future-invariance, immutable plan and CLI |
| Explicit event-model abstention sessions | Complete | zero-return ledgers preserve non-trading OOS dates in Phase 4 metrics |
| Phase 4 multi-fold gate aggregation | Complete | capital continuity and both documented threshold sets |
| Phase 4 cohort evidence and tearsheet | Complete | exact trade mapping; BMO/AMC, sector, and explicit IV-availability metrics in canonical JSON/HTML |
| Causally bounded NBBO/trade replay engine | Complete | partial/missed fills, impact, limit queues, auction skew |
| Active NBBO pre-decision-read property gate | Complete | consumed timestamps and pre-boundary mutation invariance |
| Immutable single-order replay evidence and CLI | Complete | semantic input hash, collision-safe JSON, CLI tests |
| Polygon historical NBBO/trade client | Complete | bounded SIP-time pagination, bronze capture, contract tests |
| Tick-level silver schemas and SQL views | Complete | conditions/corrections retained, content-addressed Parquet |
| Conservative silver-to-replay normalization | Complete | one-sided/corrected/sub-share rejection audit |
| Daily replay lifecycle reconciliation | Complete | exact entry/exit mapping, partial-fill breaks, commission/net P&L |
| Immutable daily Phase 6 report and CLI | Complete | evidence hashes, fill/slippage percentiles, collision-safe JSON |
| No-trade operational session evidence | Complete | explicit zero return, no invented orders or fill-rate denominator |
| Locked 90-session Phase 6 hard gate | Complete | Sharpe CI, fills, global slippage, uptime, reconciliation, CLI |
| Replay execution-cost attribution | Complete | arrival gross -> spread -> impact -> residual -> commission -> net |
| Alpaca paper-only order adapter | Complete | canonical paper-host lock, idempotent client IDs, strict response validation |
| Immutable paper submission/reconciliation evidence | Complete | CLI records, exact paper/replay identity checks, non-gating divergence |
| Operational circuit breakers | Complete | loss, three-day fill, provider freshness, T+1 reconciliation auto-halts |
| Fail-closed paper submission boundary | Complete | fresh non-halted breaker decision required before any broker request |
| Restart-safe daily workflow state machine | Complete | ordered stages, leases, retries, hash-chained revisions, artifact verification |
| In-process workflow advancement loop | Complete | advances until complete/leased/failed and durably records every transition |
| Concrete qee stage-command adapter | Complete | shell-free allowlist, environment-only secrets, JSON artifact discovery, CLI run |
| Restart-safe session paper-order batch | Complete | sorted client IDs, partial-process resume, explicit no-trade evidence, CLI |
| Unattended workflow health evidence | Complete | durable trigger provenance, authoritative sessions, uptime, five-run gate |
| Non-secret command execution receipts | Complete | exit/timing and stream digests, argument hash, success/failure persistence |
| Persistent workflow inbox worker | Complete | continuous scan/resume, cycle heartbeats, invalid-spec isolation, CLI |
| Continuous next-session proof queue | Complete | one deployment spec, authoritative calendar order, immutable input discovery, proof-start staging, replay-gated advancement |
| Provider-backed daily input preparation | Complete | T-1 universe/events, explicit empty-event evidence, candidate bar history, causal pre-open minute features, automatic queue handoff |
| Worker deployment packaging | Complete | hardened systemd unit, Windows launcher, deployment asset tests |
| Workflow-backed Phase 6 uptime gate | Complete | exact calendar/range binding; scheduled intact completions, not report presence |
| Silver-to-replay specification materialization | Complete | frozen orders/snapshots, source hashes, interval filtering audits, CLI |
| Live-safe frozen daily order artifact | Complete | decision-only scores/sizing/NBBO; linked intended and paper order IDs |
| Frozen-order-to-replay linkage | Complete | strategy hash check, silver symbol grouping, direct causal materialization CLI |
| Manifest-wide replay execution | Complete | canonical manifest reload, exact spec/hash verification, immutable evidence index, no-trade support |
| Frozen-order broker reconciliation | Complete | exact replay/frozen field match, per-client-ID Alpaca fetch, automatic immutable report |
| Frozen daily replay aggregation | Complete | deterministic entry/exit pairing, frozen equity, strategy commission, exact evidence match |
| Frozen market-event batch capture | Complete | per-symbol full execution windows, post-expiry readiness gate, immutable capture manifest |
| Complete daily workflow generation | Complete | one validated command emits all eight stages and dynamic artifact bindings |
| Temporal workflow readiness | Complete | post-close stages remain pending without false attempts while the worker loop polls |
| Rolling Phase 6 control preparation | Complete | calendar-bound workflow health, discovered prior reports, deterministic current report path |
| Rolling breaker control preparation | Complete | prior frozen/replay pairing, distinct source/control dates, current provider freshness |
| Provider-native freshness evidence | Complete | Polygon snapshot and Alpaca clock timestamps, raw bronze payload hashes, canonical immutable evidence |
| Reconciliation-break age derivation | Complete | authoritative-session close count, latest-revision resolution, immutable report provenance |
| Pre-open workflow readiness | Complete | order-control stages stay pending until ten minutes before the planned entry |
| Self-refreshing pre-open breaker bundle | Complete | discovers prior daily evidence, probes providers at execution, emits content-addressed controls |
| Retry-safe paper reconciliation revisions | Complete | content-addressed broker observations allow later clean revisions to resolve earlier breaks |
| Unified daily workflow preparation | Complete | one command prepares rolling Phase 6 controls and queues the self-refreshing eight-stage spec |
| Secret-free operational readiness audit | Complete | credentials, canonical hosts, live clocks, NBBO entitlement, roots, heartbeat, calendar, bootstrap evidence |
| Automatic post-completion Phase 6 finalization | Complete | worker refreshes health and verdict after terminal state, with hash-linked retry-safe evidence |
| External env-file propagation to stage subprocesses | Complete | child-only merged environment; process precedence; no secret arguments or mutation |
| Expiring pre-open execution boundary | Complete | first four stages stop permanently at entry expiry instead of submitting or retrying late |
| Durable operator-attention evidence | Complete | invalid specs and expired windows emit idempotent content-addressed attention records |
| Bounded exponential workflow retries | Complete | per-stage attempt budgets, capped backoff, terminal exhaustion, attention evidence |
| Isolated credentialed no-trade smoke | Complete | manual zero-order full workflow, live safety controls, separate state/artifacts, no proof credit |
| Cutoff-safe production model refit | Complete | closed-label cutoff, purged final validation, deterministic immutable booster/evidence, exact-feature inference |
| Phase 4 model-promotion binding | Complete | independently recomputed passing pre-paper gate hash is mandatory in every production model artifact |
| Phase 6 deployment-promotion binding | Complete | loop requires the actual passing Phase 4 report, exact model digest match, causal cutoff, and proof-end model-age limit |
| Phase 6 hyperparameter-study binding | Complete | production refit requires the matching Optuna artifact; proof loop rejects unbound models |
| Automatic live planning scores | Complete | strict booster reload, PIT feature schema/timestamp/lineage checks, generated probabilities and planning evidence |
| Provider-backed live source capture | Complete | SIC exposure buckets, frozen prior-close price/ADV, paper-account audit, Polygon NBBO/trade snapshots, replay-chained equity/outcomes |
| Worker-time capture and scoring | Complete | T-1 decision stage captures providers and freezes scores/orders; pre-open stage independently refreshes breaker controls |
| Fail-closed proof-start admission | Complete | fresh passing readiness + prior isolated smoke + scheduled spec required before first inbox publication |
| Typed cross-stage artifact bindings | Complete | stage/path filters, repeated options, cardinality gates, no shell interpolation |

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

Run the remaining credentialed gates: historical coverage, the source-bound
momentum benchmark, the real 200-trial Optuna study, real-data Phase 4
evaluation, gate-and-study-bound production refit, deployment readiness,
isolated no-trade smoke, and proof-start admission. The loop then prepares and
queues each authoritative session until the real 90-session Phase 6 verdict exists.
It deliberately waits rather than inventing provider, halt, candidate, feature,
or replay evidence.

The deterministic replay core now models displayed aggressive liquidity,
probability-weighted mid/passive limit fills, partial and missed fills,
square-root impact, and opening-auction skew. It does not claim exchange queue
reconstruction, and fixture evidence cannot count toward the 90-session proof.

Credentialed calendar/backfill, five-session unattended execution, the
published momentum comparison, five-year Phase 4 evaluation, and 90 observed
replay sessions remain operationally pending. Those gates cannot be replaced
by fixtures or synthetic performance.
