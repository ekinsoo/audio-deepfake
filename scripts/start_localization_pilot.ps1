param([Parameter(Mandatory=$true)][string]$AudioJobDirectory)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$configuration = Get-Content -LiteralPath (Join-Path $projectRoot '.venv/pyvenv.cfg')
$line = @($configuration | Where-Object { $_.StartsWith('executable = ') })
if ($line.Count -ne 1) { throw 'Cannot identify actual Python executable.' }
$pythonPath = $line[0].Substring('executable = '.Length)
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Actual Python executable is missing.'
}
$audioPath = (Resolve-Path -LiteralPath $AudioJobDirectory).Path
$jobsRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot 'outputs/jobs')).Path + [IO.Path]::DirectorySeparatorChar
if (-not $audioPath.StartsWith($jobsRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Audio job directory must stay under outputs/jobs.'
}
$jobName = 'localization-pilot-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
$jobDirectory = Join-Path $projectRoot ('outputs/jobs/' + $jobName)
New-Item -ItemType Directory -Path $jobDirectory -ErrorAction Stop | Out-Null
$scriptPath = Join-Path $projectRoot 'scripts/localization_pilot_job.py'
$launchArguments = @('-S', '-u', ('"' + $scriptPath + '"'), '--execute', '--job-dir',
    ('"' + $jobDirectory + '"'), '--audio-job-dir', ('"' + $audioPath + '"'))
$process = Start-Process -FilePath $pythonPath -ArgumentList $launchArguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $jobDirectory 'stdout.log') `
    -RedirectStandardError (Join-Path $jobDirectory 'stderr.log')
@{ pid = $process.Id; job_dir = $jobDirectory; audio_job_dir = $audioPath;
    launched_utc = (Get-Date).ToUniversalTime().ToString('o'); training_started = $false } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $jobDirectory 'launch.json') -Encoding UTF8
Write-Output ('Localization queue PID: ' + $process.Id)
Write-Output ('Status: ' + (Join-Path $jobDirectory 'status.json'))
