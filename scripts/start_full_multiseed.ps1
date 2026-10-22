# Start the remaining fixed full train/dev seeds 43 and 44 sequentially, without a visible window.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$jobsRoot = Join-Path $projectRoot 'outputs/jobs'
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$jobScript = Join-Path $projectRoot 'scripts/full_multiseed_job.py'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
# Venv python.exe is a Windows redirector. Launch its actual interpreter so the
# returned PID is the supervisor itself, rather than an unmonitored child.
$venvConfig = Get-Content -LiteralPath (Join-Path $projectRoot '.venv/pyvenv.cfg')
$executableLine = @($venvConfig | Where-Object { $_.StartsWith('executable = ') })
if ($executableLine.Count -ne 1) { throw 'Cannot identify the venv base executable.' }
$pythonPath = $executableLine[0].Substring('executable = '.Length)
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The actual Python executable is missing or is not absolute.'
}
if (Test-Path -LiteralPath (Join-Path $jobsRoot 'full-multiseed.lock')) {
    throw 'A multiseed lock exists. Review its PID/status; do not automatically remove it.'
}
if (Test-Path -LiteralPath (Join-Path $projectRoot 'outputs/baseline-full-seed43')) {
    throw 'Full seed 43 output already exists. Preserve and review it before another launch.'
}
$jobName = 'full-multiseed-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
foreach ($relative in @('outputs/jobs/full-seed42.lock', 'outputs/jobs/scoring-recovery-seed42.lock',
                        'outputs/baseline-full-seed44', 'docs/validation/full_multiseed_summary.json')) {
    if (Test-Path -LiteralPath (Join-Path $projectRoot $relative)) {
        throw ('Conflicting job or existing output: ' + $relative)
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'docs/validation/full_independent_review.json'))) {
    throw 'Independent seed 42 review is required before launch.'
}
$jobDirectory = Join-Path $jobsRoot $jobName
New-Item -ItemType Directory -Path $jobDirectory -ErrorAction Stop | Out-Null
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
