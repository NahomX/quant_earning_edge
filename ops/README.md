# Persistent workflow worker

The worker continuously scans an inbox of immutable daily workflow JSON specs.
It does not invent trade dates or generate schedules. An external operational
process places an authoritative, fully validated spec in the inbox; the worker
resumes it until complete and keeps scanning.

Linux deployment uses `systemd/qee-workflow-worker.service`. Install the project
at `/opt/quant_earning_edge`, secrets at
`/etc/quant_earning_edge/qee.env` (mode `0600`), and specs under
`/var/lib/quant_earning_edge/workflow-inbox`. The service account needs write
access only to `/var/lib/quant_earning_edge`.

On Windows, run:

```powershell
.\ops\run-workflow-worker.ps1 `
  -Inbox C:\ProgramData\quant_earning_edge\workflow-inbox `
  -EnvFile C:\ProgramData\quant_earning_edge\qee.env
```

Use `--once` directly on `qee workflow worker` for deployment smoke tests. A
cycle heartbeat is written under
`manifests/job=workflow-worker/cycles` on every scan, including an empty inbox.
