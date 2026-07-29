param(
    [Parameter(Mandatory = $true)]
    [string]$Inbox,

    [Parameter(Mandatory = $true)]
    [string]$EnvFile,

    [Parameter(Mandatory = $true)]
    [string]$LoopSpec,

    [string]$WorkerId = $env:COMPUTERNAME,

    [ValidateRange(1, 60)]
    [int]$PollSeconds = 10
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$QeeExecutable = Join-Path $ProjectRoot ".venv\Scripts\qee.exe"
$ResolvedInbox = (Resolve-Path $Inbox).Path
$ResolvedEnvFile = (Resolve-Path $EnvFile).Path
$ResolvedLoopSpec = (Resolve-Path $LoopSpec).Path

if (-not (Test-Path -LiteralPath $QeeExecutable -PathType Leaf)) {
    throw "qee executable not found at $QeeExecutable"
}
if ([string]::IsNullOrWhiteSpace($WorkerId)) {
    throw "WorkerId must not be blank"
}

& $QeeExecutable workflow worker `
    --inbox $ResolvedInbox `
    --worker-id $WorkerId `
    --loop-spec $ResolvedLoopSpec `
    --poll-seconds $PollSeconds `
    --env-file $ResolvedEnvFile

exit $LASTEXITCODE
