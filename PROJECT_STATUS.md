# quant_earning_edge project status

Last updated: 2026-07-28

## Executive state

**Active development; terminal proof not yet achieved.**

The research, paper-execution, evidence, safety, and continuous-worker code paths
are implemented and locally verified. No real strategy-performance claim has
been made. The project still requires paid-provider credentials/entitlements,
five years of real point-in-time data, successful real-data research gates, and
90 observed market sessions of NBBO replay before the documented terminal goal
can be complete.

Live-capital trading is outside this project's authorized scope.

## Where the project is

- Canonical local repository:
  `C:\Users\Nahom\OneDrive\Documents\quant_earning_edge`
- Project-folder clone:
  `C:\Users\Nahom\OneDrive\Documents\New project\quant_earning_edge`
- GitHub:
  `https://github.com/NahomX/quant_earning_edge`
- Active branch:
  `agent/phase1-data-foundation`
- Draft pull request:
  `https://github.com/NahomX/quant_earning_edge/pull/1`

Both local repositories track the active branch. `main` still contains the old
Phase 0 scaffold until the draft pull request is reviewed and merged.

## Verified now

- 357 automated tests pass.
- Ruff formatting and lint pass.
- Mypy strict checking passes across 93 source files.
- The model target matches the traded next-open-to-next-close holding window.
- Seeded, resumable Optuna selection is bound to complete OOS model evidence.
- Historical Phase 4 plans are assembled automatically from canonical
  candidates, sessions, adjusted bars, and OOS predictions.
- Every OOS prediction carries the latest feature-information cutoff; Phase 4
  rejects a cutoff before candidate freeze or at/after the authoritative open,
  preventing premarket features from inheriting a false prior-close timestamp.
- The Phase 4 gate independently rebuilds those plans and requires byte-exact
  equality, so a re-hashed hand-edited plan cannot enter performance metrics.
- Phase 4 source files, generated plans, strategy, costs, model run, and study
  are hash-bound through promotion.
- A bounded calibration allocation prevents the fractional-Kelly cold-start
  deadlock; only matched per-trade returns enter its 60-day history, and mature
  negative/undefined edge still shuts risk off.
- The restart-safe worker continuously discovers, queues, executes, reconciles,
  and finalizes authoritative Phase 6 paper sessions.
- The terminal Phase 6 verdict binds the ordered SHA-256 of every daily replay
  report plus bootstrap count/seed, making the exact 90-session evidence and
  statistical procedure part of the verdict identity.
- Paper submission is locked to Alpaca's paper host. Circuit breakers and
  reconciliation failures halt new orders.

Detailed component evidence is in
[`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md). Operator
commands and evidence contracts are in [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## Real evidence still required

These items cannot be honestly satisfied with fixtures or synthetic results:

1. Configure valid Finnhub, Polygon, and Alpaca paper credentials.
2. Verify Polygon historical NBBO/trade entitlement.
3. Ingest and audit at least five years of required point-in-time history.
4. Complete five consecutive unattended universe runs.
5. Reproduce the source-bound published momentum benchmark.
6. Run the real 200-trial Optuna study.
7. Assemble and evaluate the complete real-data Phase 4 OOS history.
8. Pass the documented research and pre-paper thresholds.
9. Refit the exact gate/study/strategy-bound production model.
10. Pass readiness and isolated no-trade smoke admission.
11. Run the persistent worker through 90 authoritative market sessions.
12. Pass the final NBBO-replay execution-realism gate.

The local environment currently contains no `.env` file and no credentialed
provider dataset, so those real gates have not started.

## Automation state

The project does not rely on a calendar reminder. Its persistent loop is:

```powershell
qee workflow worker `
  --inbox <workflow-inbox> `
  --worker-id <worker-id> `
  --loop-spec <phase6-loop.json>
```

That loop is intentionally fail-closed. It waits for authoritative provider
inputs or market time rather than fabricating missing evidence. It can begin
receiving proof credit only after the historical gates, production-model
binding, readiness audit, and isolated smoke admission all pass.

## Current next action

Obtain/configure the three provider credentials and Polygon NBBO entitlement,
then execute the credentialed historical backfill and coverage audit. Until
those external inputs exist, further progress is limited to additional
code-level hardening and documentation; it is not valid to claim that the
strategy or 90-session proof passed.
