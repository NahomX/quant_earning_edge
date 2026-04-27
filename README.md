# quant_earning_edge

Production-grade quantitative research platform for an event-driven (earnings) US-equities strategy. Terminal goal: 90-day NBBO-replay execution-realism proof on Alpaca paper. No live capital in this project's scope.

See [`docs/ARCHITECTURE_PLAN.md`](docs/ARCHITECTURE_PLAN.md) for the full design, budget, and phased plan.

## Quickstart

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv) (or pip).

```bash
# Clone + enter
git clone https://github.com/NahomX/quant_earning_edge.git
cd quant_earning_edge

# Set up env
cp .env.example .env  # fill in API keys

# Install (uv)
uv sync

# Run checks
make lint
make type
make test

# Start MLflow tracking server
docker compose up -d mlflow
```

## Project layout

```
src/quant_earning_edge/
  data/         data ingestion + storage (bronze/silver/gold parquet)
  universe/     ticker filtering, event calendar
  features/     feature registry (PIT-correct), price/volume/options/regime
  labels/       forward-return labeling
  signals/      LightGBM model + walk-forward harness
  portfolio/    fractional-Kelly sizing, risk caps
  backtest/     vectorbt engine, cost model, NBBO-replay simulator
  evaluation/   metrics, bootstrap CI, tearsheets
  live/         Alpaca paper executor + reconciliation
  monitoring/   drift detection, circuit breakers
configs/        YAML strategy/universe configs (pydantic-validated)
tests/property/ no-lookahead + no-future-read property tests (block PRs)
ops/            Prefect flows, deploy scripts
```

## Phases (see ARCHITECTURE_PLAN.md for gates)

| Phase | Weeks | Output |
|---|---|---|
| 0 | 1 | Hygiene |
| 1 | 2–3 | Data layer |
| 2 | 4–6 | Feature store + no-lookahead tests |
| 3 | 7–9 | Backtest chassis |
| 4 | 10–12 | Earnings v1 strategy |
| 5 | 13–16 | Options + regime (conditional) |
| 6 | 17–20+ | NBBO-replay paper proof |

## Status

Phase 0. Bootstrap only.
