# Architecture & Implementation Plan — Quant Edge System

**Role:** Principal Quant Developer
**Scope:** Rebuild the existing project (`Intiutive_trading_monte_carlo`) into a production-grade quantitative trading research platform, then promote to live paper trading on Alpaca.
**Budget (confirmed):** $300/month recurring for data, infra, and APIs.
**Terminal goal (confirmed):** Paper-trading proof at end of Phase 6. No live capital deployment in this project's scope.

---

## 0. Framing assumptions

| Assumption | Default | Knob |
|---|---|---|
| Budget cadence | $300/mo recurring (confirmed) | — |
| Team | Solo developer | If team: split data + research from execution |
| Capital sizing target | Notional $10–25k retail (paper) | — |
| Latency | End-of-day to next-day-open | This is not HFT |
| Universe | US equities, $5+ price, $500M+ market cap, ≥1M ADV | ~2,500 tradable names; earnings-day subset = 10–30/day |
| Strategy archetype | Event-driven (earnings) multi-factor, daily holding | Other archetypes can plug in later |
| Terminal deliverable | 90-day NBBO-replay execution-realism report (Alpaca paper = operational smoke test only) | — |

---

## 1. High-level architecture

```
                           +------------------------------+
                           |      Orchestration            |  Prefect (or APScheduler)
                           |  (walk-forward, daily live)   |  + MLflow run registry
                           +--------------+---------------+
                                          |
   +--------------+    +------------------v----------------------+    +--------------+
   |  Data sources |--->|  Bronze -> Silver -> Gold (Parquet)     |<--|   DuckDB     |
   |  Polygon      |    |  Point-in-time feature store            |    |   (queries)  |
   |  Alpaca       |    +------------------+----------------------+    +--------------+
   |  Finnhub      |                       |
   |  FRED         |                       v
   |  yfinance     |      +--------------------------------+
   +--------------+       |  Universe + Event Calendar     |
                          |  (earnings, splits, halts)     |
                          +----------------+---------------+
                                          |
                          +----------------v---------------+
                          |  Feature Registry (PIT-correct) |
                          |  price | vol | options | regime |
                          +----------------+---------------+
                                          |
                          +----------------v---------------+
                          |  Label maker (forward returns) |
                          +----------------+---------------+
                                          |
                          +----------------v---------------+
                          |  Signal model (LightGBM/HMM)   |
                          |  Walk-forward trained          |
                          +----------------+---------------+
                                          |
                          +----------------v---------------+
                          |  Portfolio constructor          |
                          |  (frac Kelly, sector caps)     |
                          +----------------+---------------+
                                          |
                          +----------------v---------------+
                          |  Backtest engine (vectorbt)     |
                          |  + cost/slippage model          |
                          +----------------+---------------+
                                          |
                          +----------------v---------------+
                          |  Evaluation: metrics, bootstrap |
                          |  tearsheet, attribution         |
                          +----------------+---------------+
                                          |
                          +----------------v---------------+
                          |  Live: Alpaca paper             |
                          |  + monitoring / circuit breakers|
                          +--------------------------------+
```

**Design principles** (priority order):
1. **Point-in-time correctness.** Every feature on date T uses data <= T. Enforced by the feature store schema and tested.
2. **Walk-forward by default.** No global parameter optimization. Every model retrains on a rolling window.
3. **Cost-aware from day 1.** Slippage + commission baked into the same simulator as P&L.
4. **Reproducibility.** Every backtest run logs code hash, data hash, params, seed. Same inputs -> same outputs.
5. **Separation of concerns.** Data, features, signals, portfolio, execution, evaluation are independent layers with typed interfaces.
6. **Fail loudly.** No silent NaN-fills, no silent skips. Bad data raises; missing features halt the pipeline.

---

## 2. Layer-by-layer design

### Layer 1 — Data ingestion & storage

| Source | What | Why | Cost |
|---|---|---|---|
| Polygon.io Stocks **Advanced** | Daily + 1-min bars **+ NBBO/Trades**, 10+ yrs, splits/dividends | Replaces yfinance; **NBBO is mandatory** for Phase 6 execution-realism proof. Buying realism, not extra features. | $199/mo |
| Polygon.io Options Developer | Options chain + IV + Greeks, EOD | **Deferred indefinitely.** Only subscribe if Phase 5 diagnosis identifies a specific options-driven feature gap (see Phase 5 gate). | $0 (deferred) |
| Alpaca | Paper broker + real-time data feed | Operational smoke test only — proves the pipeline runs end-to-end. Paper P&L is **not** a proof metric. | $9/mo data |
| Finnhub | Earnings calendar | Free tier covers earnings calendar adequately | $0 |
| FRED (St. Louis Fed) | Macro context (VIX, term spread, credit spreads) | Free, authoritative; universe-level features | $0 |
| SEC EDGAR | Fundamentals (10-Q deltas) | Free; optional Phase 5+ | $0 |
| yfinance | Fallback only | Keep for ad-hoc research; never in production path | $0 |

**Storage tiering** (Lakehouse pattern):
- **Bronze**: raw API responses, immutable, append-only, partitioned by `(source, date)`. Parquet.
- **Silver**: cleaned, survivorship-corrected, split-adjusted, schema-stable. Partitioned by `(asset_class, date)`.
- **Gold**: feature tables keyed by `(ticker, asof_date)`. One Parquet per feature group per month.
- **Query layer**: DuckDB reads Parquet directly — zero ETL, SQL queries, fast.

**Why not SQL Server?** Parquet+DuckDB is faster for analytical workloads, free, immutable (better for reproducibility), trivially version-controllable, and runs anywhere without a server.

### Layer 2 — Universe & event calendar

Single source of truth for "what's tradable on date T":
- Daily eligible-universe snapshot computed at close-of-T-1: price floor, market-cap floor, ADV floor, no halts, not in delisting, exchange = NYSE/NASDAQ/AMEX.
- Event calendar joining earnings (Finnhub) + splits (Polygon) + dividends (Polygon).
- Earnings rows tagged BMO / AMC / DMT.
- Snapshot is **frozen at T-1 close**; backtests on date T read this snapshot.

### Layer 3 — Feature registry (PIT-correct)

A `Feature` is a pure function: `(ticker, asof_date) -> scalar/vector`. Every feature is **registered** with metadata: `name`, `lookback_days`, `update_cadence`, `code_hash`, `dependencies`.

Initial feature set:
- **Price**: 1/5/20-day return, 20-day vol, 60-day vol, distance to 20-day VWAP.
- **Volume**: Kalman-**filtered** (causal, not smoothed) 7- and 30-day average; relative-to-30d ratio.
- **Gap**: pre-market gap %.
- **Momentum**: RSI(14), MACD signal, distance to 52w high.
- **Options-derived (Phase 5)**: ATM IV, IV30 percentile, term structure slope, **implied move** = ATM straddle / spot, put/call OI ratio.
- **Event**: earnings BMO/AMC flag, days-since-last-earnings, surprise % from prior quarter.
- **Regime (Phase 5)**: HMM state on 60-day realized vol; macro regime from VIX percentile.
- **Cross-sectional**: sector-relative versions of the above (z-scored within sector on date T).

**Critical**: the test suite includes `test_no_lookahead.py` — for each feature, recompute with truncated history at T-1 and assert equality with a value computed from full history truncated at T-1.

### Layer 4 — Label maker

Forward returns computed once per (ticker, T) over multiple horizons: D+1 open-to-close, D+1 close, D+5 close. Labels live in Gold, never recomputed in the strategy.

### Layer 5 — Signal model

**Phase 1 model: LightGBM binary classifier** predicting `P(forward return > threshold | features)`.
- Handles non-linearities and interactions.
- Built-in feature importance.
- Robust to scale and missing values.
- Trains fast; walk-forward feasible.

**Phase 5+:** ensemble over a regime gate. HMM identifies regime -> a regime-specific LightGBM activates. Don't predict regime transitions; gate on current regime.

**Anti-overfit discipline:**
- Walk-forward with **purged** cross-validation (Lopez de Prado): training fold ends at T_train, embargo gap of `max_label_horizon`, then test fold.
- Hyperparameter search via **Optuna** with TimeSeriesSplit objective. Objective = mean OOS Sharpe across folds, not training accuracy.
- Hard cap on trials (200) and on number of features (20).

### Layer 6 — Portfolio constructor

Daily routine at T-1 close:
1. Read universe snapshot for T.
2. Score each candidate via the current model.
3. Rank, take top-K (default K=5–10).
4. Position sizing: **fractional Kelly = 0.25x Kelly** based on rolling 60-day per-trade win rate * payoff. Hard cap: 5% per position, 20% per sector, 50% gross exposure.
5. Generate intended orders for T-open.

### Layer 7 — Backtest engine

**Choice: vectorbt** (free).
- Vectorized over the entire signal matrix; runs the earnings strategy on 5 years x 30 names/day in seconds.
- Native pandas integration.
- Easy parameter sweeps without re-running data.

**Cost model** (all baked in):
- Commission: 1 bp per side.
- Half-spread: lookup by price tier — 2 bps for liquid > $50, 5 bps for $10–50, 15 bps below $10.
- Market impact: `5 bps * sqrt(order_size / ADV)`.
- Borrow cost: 50 bps annualized for shorts (Phase 6+).
- Slippage on stops: stops fill at `stop_price - 1 * ATR(5)`.

### Layer 8 — Evaluation

Every backtest produces a standardized report:
- Headline: net Sharpe, gross Sharpe, max drawdown, hit rate, payoff, exposure, turnover.
- **Bootstrap 95% CI** on Sharpe and on annualized return — 10,000 resamples of the per-trade P&L vector.
- Walk-forward fold-by-fold table.
- Per-feature SHAP attribution.
- Cost-attribution: gross Sharpe minus net Sharpe per cost component.
- Cohort tearsheets: BMO vs AMC, by sector, by IV regime.

`pyfolio-reloaded` + `quantstats` generate the visuals.

### Layer 9 — Paper-trading deployment + NBBO-replay proof (terminus)

**Three gates before Phase 6 begins:**
1. **Backtest gate**: net Sharpe > 1.0, max DD < 15%, bootstrap lower-CI on Sharpe > 0.5, walk-forward fold count where Sharpe > 0 >= 75% of folds.
2. **Pipeline readiness**: full reproducibility tests pass; signal pipeline runs unattended for 5 days; decision-time NBBO snapshot logged for every order.
3. **Final proof**: see "Phase 6 methodology" below.

**Daily live workflow:**
- T-1 21:00 ET: pull data, compute features, freeze universe snapshot.
- T-1 21:30 ET: run model, generate orders, **log decision-time NBBO snapshot per order**, log to MLflow.
- T 09:25 ET: pre-market sanity check (halt list, news gates).
- T 09:30 ET: submit orders via Alpaca paper (smoke test).
- T 16:05 ET: reconcile fills, log paper-vs-NBBO-replay divergence, attribute slippage.

**Circuit breakers** (auto-halt new orders):
- Daily NBBO-replay modeled loss > 2% of notional.
- 3 consecutive days where NBBO-replay fill rate < 70%.
- Polygon or Alpaca data freshness > 30 min stale.
- Reconciliation break unresolved at T+1 close.

---

## 3. Locked budget ($300/mo hard cap, $235 committed, $65 buffer)

The data spend is redirected from "options-by-default" to **NBBO-by-default**. Polygon Advanced replaces Stocks Developer because NBBO/quote replay is mandatory for the Phase 6 proof, not optional.

| Item | Cost/mo | Status |
|---|---|---|
| Polygon Stocks **Advanced** (incl. NBBO/Trades) | $199 | Subscribe Week 1 |
| Polygon Options Developer | $0 | **Deferred indefinitely** — gated on Phase 5 diagnosis |
| Alpaca real-time data feed | $9 | Subscribe Week 17 (start of paper trading) |
| Hetzner CX22 VPS (4GB, 2 vCPU, EU) | ~$7 | Week 1 |
| Healthchecks.io / UptimeRobot / Sentry free tiers | $0 | Week 1 |
| Anthropic/OpenAI API for research helpers | ~$20 | Ongoing, on-demand |
| **Committed** | **~$235** | Within $300 cap with $65 headroom |
| **Buffer** | **~$65** | One-time costs only — see invariant below |

### Budget invariant (hard rule)

> **The buffer is not a slush fund.** It funds known one-time costs (e.g., Polygon historical backfill) and rare overruns. Any *recurring* upgrade requires a named offset chosen from this list **before** the upgrade is purchased:
>
> 1. Drop or defer LLM API spend (-$20/mo)
> 2. Halve VPS spec to Hetzner CX11 (-$3/mo)
> 3. Defer Alpaca data feed (-$9/mo) — only if Phase 6 hasn't started
> 4. Stop the project
>
> Without a named offset, the upgrade does not happen. This applies to every "if X reveals Y, upgrade to Z" pattern.

**Acceptable uses of the $65 buffer:**
- One-time Polygon historical bulk backfill if needed.
- Burst LLM API usage for specific research sprints (capped at $20 over the $20 baseline per month).
- Hetzner short-term scale-up (CCX23 at $25/mo) for time-bounded Optuna sweeps — must end same month.

**What the buffer is NOT for:**
- Adding Polygon Options ($79/mo recurring) — needs Phase 5 diagnosis + named offset.
- Adding any other recurring SaaS.
- Filling a "we might need more data later" gap. If you don't know what data, you don't need it yet.

---

## 4. Tech stack rationale

| Concern | Choice | Why over alternatives |
|---|---|---|
| Language | Python 3.11+ | Continuity; ecosystem |
| Storage | Parquet + DuckDB | Free, fast, immutable, columnar |
| Backtester | vectorbt | Vectorized for signal/factor strategies; backtrader is for stateful order management |
| Hyperparameter search | Optuna | TPE sampling, pruning, persistable studies |
| ML model | LightGBM | Tabular, fast, robust |
| Regime detection | hmmlearn | Standard HMM library |
| Time-series CV | sklearn TimeSeriesSplit + custom purge/embargo | Lopez de Prado purged CV |
| Portfolio analytics | quantstats + pyfolio-reloaded | Free, polished tearsheets |
| Experiment tracking | MLflow | Self-hosted, free, durable |
| Orchestration | Prefect 2.x (or APScheduler if minimal) | Failures, retries, schedules |
| Containers | Docker + docker-compose | Reproducible local + VPS |
| Broker | Alpaca | Free paper, REST + websocket |
| Secrets | `.env` + python-dotenv (dev), VPS env (prod) | Out of git |
| Tests | pytest + hypothesis (property-based for PIT) | Hypothesis catches look-ahead |
| Linting | ruff + mypy strict | Cheap insurance |

---

## 5. Repository layout

```
quant_edge/
├── pyproject.toml
├── docker-compose.yml
├── Makefile
├── .env.example
├── README.md
├── src/quant_edge/
│   ├── data/
│   │   ├── clients/        # polygon, alpaca, finnhub, fred
│   │   ├── ingest.py       # bronze writers
│   │   ├── clean.py        # bronze -> silver
│   │   └── store.py        # DuckDB session, partition helpers
│   ├── universe/
│   ├── features/
│   │   ├── registry.py     # @feature decorator + metadata
│   │   ├── price.py
│   │   ├── volume.py
│   │   ├── options.py
│   │   ├── regime.py
│   │   └── crosssection.py
│   ├── labels/
│   ├── signals/
│   │   ├── lgbm_model.py
│   │   └── walkforward.py
│   ├── portfolio/
│   ├── backtest/
│   │   ├── engine.py       # vectorbt wrapper
│   │   ├── costs.py
│   │   └── slippage.py
│   ├── evaluation/
│   ├── live/
│   │   ├── alpaca_executor.py
│   │   ├── reconcile.py
│   │   └── breakers.py
│   ├── monitoring/
│   └── cli.py              # Typer CLI
├── configs/
│   ├── strategies/earnings_v1.yaml
│   └── universe/default.yaml
├── notebooks/              # exploration only — never imported by src
├── tests/
│   ├── unit/
│   ├── property/
│   │   └── test_no_lookahead.py    # CRITICAL
│   └── integration/
└── ops/
    └── prefect_flows/
```

Hard rules: `src/` never imports from `notebooks/`. CI runs on every PR. Configs are pydantic-validated YAML. Every backtest run is invoked through `cli.py`, which logs to MLflow with code hash.

---

## 6. Phased delivery — 20 weeks build, then 90-day paper proof

| Phase | Weeks | Output | Hard exit gate |
|---|---|---|---|
| 0. Hygiene | 1 | Clean repo: secrets out, requirements pinned, Docker baseline, CI green, ruff/mypy strict | CI green |
| 1. Data layer | 2–3 | Polygon + Finnhub clients, bronze->silver->gold pipeline, DuckDB queries, universe snapshots | 5 yrs OHLCV stored, daily snapshot job runs unattended for 5 days |
| 2. Feature store | 4–6 | Feature registry, 15 baseline features, label maker, gold tables | 100% pass on `test_no_lookahead.py` (property-based, random T) |
| 3. Backtest chassis | 7–9 | vectorbt engine, cost model, slippage, walk-forward harness, evaluation/tearsheet | Reproduces 60-day momentum on SPY components within +/-0.1 Sharpe of published |
| 4. Strategy v1: earnings multi-factor | 10–12 | LightGBM on earnings-day features, walk-forward 5 years, full evaluation | **Net Sharpe >= 0.8 with bootstrap lower CI >= 0.3, max DD <= 20%.** If passes: skip Phase 5, go straight to Phase 6. If fails: enter Phase 5 conditionally (see entry gate). |
| 5. Options + regime features (**conditional**) | 13–16 | Polygon Options + HMM regime gate — **only if Phase 5 entry gate is met** | Improvement must survive recent-fold + cost + missing-data + NBBO-replay stress, not just backtest Sharpe lift |
| 6. NBBO-replay paper proof | 17–20+ | Alpaca paper deploy (smoke test) + NBBO-replay simulator running on the same orders, daily | 90 trading days, NBBO-replay net Sharpe > 0.8, fill rate > 90%, 90th-pct slippage < 2x modeled, ops uptime > 95% |

20 weeks of build + 90 days of paper = ~32 weeks to terminal verdict (or shorter if Phase 5 is skipped).

### Phase 5 entry gate (all three must hold)

Phase 5 is **not** a default progression. It is entered only if all of the following are true:

1. **v1 fails the Phase 4 gate** (net Sharpe < 0.8 OR bootstrap lower-CI < 0.3 OR max DD > 20%).
2. **Failure root-cause analysis identifies a specific feature gap** that options or regime data would fill. Examples that would qualify: "fold-by-fold Sharpe degrades systematically across high-VIX regimes" (regime gate), "strategy is run over by IV crush surprises" (options data). Examples that would NOT qualify: "Sharpe is just a bit low, options might help."
3. **Adding the proposed feature must show improvement that survives**: recent-fold validation (last 12 months), cost-attribution, missing-data robustness, AND NBBO-replay execution stress on a 30-day pilot. A backtest Sharpe lift > 0.2 alone is no longer sufficient.

If any of (1)/(2)/(3) fails: **skip Phase 5 entirely**. Go to Phase 6 with the v1 stack. This frees $79/mo (no Polygon Options) and 4 weeks for a tighter terminal proof.

---

## 7. Phase 6 methodology — NBBO-replay execution realism

**Why the original tracking-error gate was wrong.** Comparing cost-modeled expected P&L against Alpaca paper P&L gates on agreement between two estimators that may share the same optimistic fill assumptions. Alpaca paper fills any limit order the market touches; my cost model assumes fills at modeled price. Both can lie in the same direction (especially in thin opening auctions, partial fills, queue effects). Low tracking error then certifies *simulator agreement*, not *real-world executability*.

**The proof is now anchored to recorded NBBO/quote data, not to paper-vs-model agreement.**

### The pipeline

1. **Decision-time NBBO snapshot.** When orders are generated at T-1 21:30 ET, log the prevailing NBBO and last trade per intended order. This freezes "the world we thought we'd trade in."
2. **NBBO-replay simulator** (runs after market close from Polygon Advanced data):
   - For each intended order, replay against recorded quotes/trades through the day.
   - Compute realistic fill price = mid + signed half-spread + impact term (`5 bps × sqrt(order_size / ADV)`).
   - Model fill probability for limit orders by aggressiveness bracket (aggressive at NBBO, mid, passive at bid/ask).
   - Track partial fills, missed fills, opening-auction skew explicitly.
   - Output per-order: filled_qty, fill_price, slippage_bps_realized, slippage_bps_predicted.
3. **Alpaca paper run** in parallel as **operational smoke test**:
   - Confirms orders submit, fills come back, reconciliation balances.
   - Paper P&L is logged but is **not** a proof metric.
4. **Daily three-way reconciliation**: cost-modeled expected | NBBO-replay realized | Alpaca paper. Divergences are alerts.

### The proof statement (what we ship at end of Phase 6)

> **Across 90 trading days:**
> - NBBO-replay net Sharpe = X with bootstrap 95% CI [A, B]
> - Realized fill rate = Y% on intended orders
> - Slippage stress (10th / 50th / 90th percentile of realized vs predicted): [..]
> - Operational pipeline ran unattended Z% of days; reconciliation breaks resolved within 1 trading day
> - Cost-attribution: gross Sharpe minus net Sharpe per cost component

### Phase 6 hard gate (replaces tracking-error gate)

- NBBO-replay net Sharpe > 0.8 with bootstrap lower-CI > 0.3
- Realized fill rate > 90% on intended orders
- 90th-percentile realized slippage < 2x modeled slippage
- Operational uptime > 95% of trading days
- No unresolved reconciliation breaks at month-end

Tracking error between paper and NBBO-replay is logged as diagnostic, **not gated on**. The proof's quality argument no longer depends on Alpaca paper telling the truth.

---

## 8. Correctness tests that block PRs

Non-negotiable in CI:

1. **`test_no_lookahead.py`**: For every registered feature, compute `f(ticker, T)` using data clipped to T, then again using full history clipped at T. Assert exact equality.
2. **`test_pit_universe.py`**: Universe snapshot at T-1 close has only tickers active and listed at T-1 close.
3. **`test_walkforward_determinism.py`**: Fixed seed -> identical trade list, P&L, model parameters.
4. **`test_cost_attribution.py`**: Net P&L = Gross P&L - sum(modeled costs). No phantom drift.
5. **`test_label_horizon_purge.py`**: Last training date + label horizon <= first test date. No leak.
6. **`test_options_freshness.py`**: IV used for a date is the prior-close snapshot, not T's snapshot. (Only enforced if Phase 5 enters.)
7. **`test_nbbo_replay_no_future_read.py`**: For every replayed order at decision time T_d, the simulator only reads quotes timestamped >= T_d. Replay must never read quotes before the decision was taken.

---

## 9. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Cost model + Alpaca paper share optimism; terminal proof is false-positive** | High (was hidden) | Critical | NBBO-replay simulator is the primary proof source; paper is smoke test only |
| **Buffer treated as available capital; recurring upgrades push over $300 cap** | Medium | Medium | Named-offset invariant (Sec. 3); buffer reserved for one-time costs only |
| Earnings universe too small for stat significance | Medium | High | Pool tickers cross-sectionally for training; supplement with non-earnings event types only if Phase 5 entry gate is met |
| Polygon Options API gaps for low-liquidity names | High | Medium | Mostly moot under deferred Options strategy; if Phase 5 enters, filter by min open interest |
| Costs swallow the edge at retail size | Medium | High | Cost-attribution report is first-class output; if net Sharpe < 0.5 of gross, increase holding period or filter for liquid names |
| Walk-forward Sharpe degrades over recent folds (regime shift) | High | Medium | Model-staleness alert always on; HMM regime gate ONLY if Phase 5 entry gate is met |
| Behavioral drift: paper looks good -> deploy real anyway | Medium | High | Pre-commit: terminal is paper proof; live is a separate, fresh decision later |
| API key compromise (Finnhub + FMP currently in repo history) | Already happened | High | Rotate immediately; scrub history; .env from now on |
| VPS outage during market hours | Low | Low (paper only) | Single VPS; no replica needed for paper |
| Survivorship bias in historical universe | High in raw data | High | Polygon delisted-tickers feed; never reconstruct universe from "what's listed today" |
| Alpaca paper fills are too generous (real-world risk) | Certain | Medium | Already addressed: Phase 6 judged on NBBO replay, not paper P&L |
| NBBO-replay simulator itself becomes a new look-ahead/optimism source | Medium | High | Use only the NBBO snapshot logged at decision time + recorded subsequent quotes. Property test: replay never reads quotes from before the decision time. |

---

## 10. Week 1 kickoff checklist

1. Rotate Finnhub and FMP keys at the providers (current keys are in git history).
2. **Subscribe to Polygon Stocks Advanced ($199)** — includes NBBO/Trades, mandatory for Phase 6 proof. Backfill clock starts now. Do NOT subscribe to Polygon Options.
3. Provision Hetzner CX22, install Docker + docker-compose, set up SSH keys + UFW.
4. Bootstrap the repo: `pyproject.toml` (uv or Poetry), pinned deps, ruff, mypy --strict, pytest, GitHub Actions CI.
5. Write `tests/property/test_no_lookahead.py` first — empty test scaffold using `hypothesis`. No feature gets merged without passing it.
6. Set up MLflow in docker-compose alongside the app.
7. Draft `configs/strategies/earnings_v1.yaml` schema with pydantic validation.
8. Move existing earnings calendar code into `src/quant_edge/data/clients/finnhub.py`, cleaned up: env-var key, typed responses, retry/backoff.
9. Stub the **NBBO-replay simulator** in `src/quant_edge/backtest/nbbo_replay.py` — empty scaffold + test for "replay never reads quotes from before decision time."
10. Decide: new repo vs branch off existing. Recommendation: new repo (existing carries committed secrets and `myenv/` clutter).

---

## 11. The pre-commitment

Single highest-risk failure mode for paper-trading-proof projects: Phase 4 looks great, ship to paper, paper looks great after 30 days, the temptation to deploy real capital "just a little" overrides the discipline that produced the result.

The deliverable is the report at end of week ~32. Real capital is a separate decision made *after* the report exists, with a fresh evaluation, not a momentum carry. The architecture supports that future decision — but the commitment for *this* project is paper, full stop.
