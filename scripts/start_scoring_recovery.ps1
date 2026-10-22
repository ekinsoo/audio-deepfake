# Start one independent scoring-only recovery, keeping the failed run intact.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$jobsRoot = Join-Path $projectRoot 'outputs/jobs'
foreach ($lockName in @('full-seed42.lock', 'scoring-recovery-seed42.lock')) {
    if (Test-Path -LiteralPath (Join-Path $jobsRoot $lockName)) { throw ('Review existing lock: ' + $lockName) }
}
if (Test-Path -LiteralPath (Join-Path $projectRoot 'outputs/scoring-recovery-seed42')) {
    throw 'Recovery output already exists. Preserve and review it.'
}
$executableLine = @(Get-Content -LiteralPath (Join-Path $projectRoot '.venv/pyvenv.cfg') |
    Where-Object { $_.StartsWith('executable = ') })
if ($executableLine.Count -ne 1) { throw 'Cannot identify project base Python.' }
$pythonPath = $executableLine[0].Substring('executable = '.Length)
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath)) { throw 'Missing base Python.' }
$jobName = 'scoring-recovery-seed42-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
$jobDirectory = Join-Path $jobsRoot $jobName
New-Item -ItemType Directory -Path $jobDirectory -ErrorAction Stop | Out-Null
$jobScript = Join-Path $projectRoot 'scripts/scoring_recovery_job.py'
$launchArguments = @('-S', '-u', ('"' + $jobScript + '"'), '--job-dir', ('"' + $jobDirectory + '"'))
$process = Start-Process -FilePath $pythonPath -ArgumentList $launchArguments `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $jobDirectory 'supervisor.stdout.log') `
    -RedirectStandardError (Join-Path $jobDirectory 'supervisor.stderr.log')
@{ pid = $process.Id; job_dir = $jobDirectory; launched_utc = (Get-Date).ToUniversalTime().ToString('o');
   executable = $pythonPath; arguments = $launchArguments } |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $jobDirectory 'launch.json') -Encoding UTF8
Write-Output ('Supervisor PID: ' + $process.Id)
Write-Output ('Job status: ' + (Join-Path $jobDirectory 'status.json'))
