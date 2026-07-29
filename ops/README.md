# Persistent workflow worker

The worker runs a state-driven loop over an authoritative session file. It
prepares provider-backed daily inputs, stages the first proof session for
admission, queues later immutable workflow specifications only after the prior
session completes, and resumes every queued workflow until complete. It uses
no wall-clock scheduler.

Linux deployment uses `systemd/qee-workflow-worker.service`. Install the project
at `/opt/quant_earning_edge`, secrets at
`/etc/quant_earning_edge/qee.env` (mode `0600`), the deployment loop spec at
`/etc/quant_earning_edge/phase6-loop.json`, and queued specs under
`/var/lib/quant_earning_edge/workflow-inbox`. The service account needs write
access only to `/var/lib/quant_earning_edge`.

On Windows, run:

```powershell
.\ops\run-workflow-worker.ps1 `
  -Inbox C:\ProgramData\quant_earning_edge\workflow-inbox `
  -EnvFile C:\ProgramData\quant_earning_edge\qee.env `
  -LoopSpec C:\ProgramData\quant_earning_edge\phase6-loop.json
```

Use `--once` directly on `qee workflow worker` for deployment smoke tests. A
cycle heartbeat is written under
`manifests/job=workflow-worker/cycles` on every scan, including an empty inbox.
