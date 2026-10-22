# Run the loopback upload demo independently of the Codex launcher shell.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvConfig = Get-Content -LiteralPath (Join-Path $projectRoot '.venv/pyvenv.cfg')
$executableLine = @($venvConfig | Where-Object { $_.StartsWith('executable = ') })
if ($executableLine.Count -ne 1) { throw 'Cannot identify the venv base executable.' }
$pythonPath = $executableLine[0].Substring('executable = '.Length)
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The actual Python executable is missing or is not absolute.'
}
$jobName = 'demo-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
$jobDirectory = Join-Path $projectRoot ('outputs/demo/' + $jobName)
New-Item -ItemType Directory -Path $jobDirectory -ErrorAction Stop | Out-Null
$jobScript = Join-Path $projectRoot 'scripts/demo_job.py'
$launchArguments = @('-S', '-u', ('"' + $jobScript + '"'), 'serve', '--host', '127.0.0.1', '--port', '8765')
$process = Start-Process -FilePath $pythonPath -ArgumentList $launchArguments `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $jobDirectory 'stdout.log') `
    -RedirectStandardError (Join-Path $jobDirectory 'stderr.log')
@{ pid = $process.Id; job_dir = $jobDirectory; launched_utc = (Get-Date).ToUniversalTime().ToString('o');
   url = 'http://127.0.0.1:8765'; executable = $pythonPath; arguments = $launchArguments } |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $jobDirectory 'launch.json') -Encoding UTF8
Write-Output ('Demo PID: ' + $process.Id)
Write-Output ('Logs: ' + $jobDirectory)
Write-Output 'URL: http://127.0.0.1:8765'
