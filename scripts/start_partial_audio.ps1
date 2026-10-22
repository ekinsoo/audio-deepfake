# Start only bounded audio preparation; this never starts model training.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$configuration = Get-Content -LiteralPath (Join-Path $projectRoot '.venv/pyvenv.cfg')
$executableLine = @($configuration | Where-Object { $_.StartsWith('executable = ') })
if ($executableLine.Count -ne 1) { throw 'Cannot identify the actual Python executable.' }
$pythonPath = $executableLine[0].Substring('executable = '.Length)
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Actual Python executable is missing.'
}
$jobName = 'partial-audio-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
$jobDirectory = Join-Path $projectRoot ('outputs/jobs/' + $jobName)
New-Item -ItemType Directory -Path $jobDirectory -ErrorAction Stop | Out-Null
$jobScript = Join-Path $projectRoot 'scripts/partial_audio_job.py'
$launchArguments = @('-S', '-u', ('"' + $jobScript + '"'), '--job-dir', ('"' + $jobDirectory + '"'))
$process = Start-Process -FilePath $pythonPath -ArgumentList $launchArguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $jobDirectory 'stdout.log') `
    -RedirectStandardError (Join-Path $jobDirectory 'stderr.log')
@{ pid = $process.Id; job_dir = $jobDirectory; launched_utc = (Get-Date).ToUniversalTime().ToString('o');
   training_started = $false } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $jobDirectory 'launch.json') -Encoding UTF8
Write-Output ('Audio preparation PID: ' + $process.Id)
Write-Output ('Status: ' + (Join-Path $jobDirectory 'status.json'))
