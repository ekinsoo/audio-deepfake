# Resume missing fixed seed 44 scoring; preserve original output and logs.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$jobsRoot = Join-Path $projectRoot 'outputs/jobs'
$configLines = Get-Content -LiteralPath (Join-Path $projectRoot '.venv/pyvenv.cfg')
$executableLine = @($configLines | Where-Object { $_.StartsWith('executable = ') })
if ($executableLine.Count -ne 1) { throw 'Cannot identify base executable.' }
$pythonPath = $executableLine[0].Substring('executable = '.Length)
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Actual Python executable is missing.'
}
foreach ($relative in @('outputs/jobs/full-robustness.lock', 'outputs/jobs/full-robustness-recovery.lock', 'outputs/jobs/full-seed42.lock',
                        'outputs/jobs/scoring-recovery-seed42.lock', 'outputs/jobs/full-multiseed.lock',
                        'outputs/robustness-full-v1/recovery_execution.json',
                        'outputs/robustness-full-v1/summary.json', 'docs/validation/full_robustness_results_v1.json')) {
    if (Test-Path -LiteralPath (Join-Path $projectRoot $relative)) { throw ('Preserve conflicting artifact: ' + $relative) }
}
$oldStatus = Get-Content -Raw -LiteralPath (Join-Path $jobsRoot 'full-robustness-20261010T095134725Z/status.json') | ConvertFrom-Json
if ($oldStatus.status -ne 'stopped' -or $null -ne $oldStatus.child_pid) { throw 'Original job is not safely stopped.' }
if (Get-Process -Id $oldStatus.pid -ErrorAction SilentlyContinue) { throw 'Original PID is present; review ownership.' }
$jobName = 'full-robustness-recovery-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
$jobDirectory = Join-Path $jobsRoot $jobName
New-Item -ItemType Directory -Path $jobDirectory -ErrorAction Stop | Out-Null
$scriptPath = Join-Path $projectRoot 'scripts/full_robustness_recovery_job.py'
$launchArguments = @('-S', '-u', ('"' + $scriptPath + '"'), '--job-dir', ('"' + $jobDirectory + '"'))
$process = Start-Process -FilePath $pythonPath -ArgumentList $launchArguments `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $jobDirectory 'supervisor.stdout.log') `
    -RedirectStandardError (Join-Path $jobDirectory 'supervisor.stderr.log')
@{ pid = $process.Id; job_dir = $jobDirectory; launched_utc = (Get-Date).ToUniversalTime().ToString('o');
   executable = $pythonPath; arguments = $launchArguments } |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $jobDirectory 'launch.json') -Encoding UTF8
Write-Output ('Supervisor PID: ' + $process.Id)
Write-Output ('Job status: ' + (Join-Path $jobDirectory 'status.json'))
