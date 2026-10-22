# Launch fixed narrowband scoring with the actual Python process, hidden.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$jobsRoot = Join-Path $projectRoot 'outputs/jobs'
$venvConfig = Get-Content -LiteralPath (Join-Path $projectRoot '.venv/pyvenv.cfg')
$executableLine = @($venvConfig | Where-Object { $_.StartsWith('executable = ') })
if ($executableLine.Count -ne 1) { throw 'Cannot identify the venv base executable.' }
$pythonPath = $executableLine[0].Substring('executable = '.Length)
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The actual Python executable is missing or is not absolute.'
}
foreach ($relative in @('outputs/jobs/full-robustness.lock', 'outputs/jobs/full-seed42.lock',
                        'outputs/jobs/scoring-recovery-seed42.lock', 'outputs/jobs/full-multiseed.lock',
                        'outputs/robustness-full-v1', 'docs/validation/full_robustness_results_v1.json')) {
    if (Test-Path -LiteralPath (Join-Path $projectRoot $relative)) {
        throw ('Conflicting job or existing output: ' + $relative)
    }
}
$jobName = 'full-robustness-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
$jobDirectory = Join-Path $jobsRoot $jobName
New-Item -ItemType Directory -Path $jobDirectory -ErrorAction Stop | Out-Null
$jobScript = Join-Path $projectRoot 'scripts/full_robustness_job.py'
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
