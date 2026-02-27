param(
  [string]$Config = 'configs/edgeai.yaml',
  [string]$RunName = '',
  [string]$InitCkpt = '',
  [switch]$Resume
)

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:COUGHCOUNT_WORKSPACE = $workspace

$configAbs = Join-Path $workspace $Config
$trainScript = Join-Path $PSScriptRoot '07_train_edgeai.py'

$args = @($trainScript, '--config', $configAbs)
if (-not [string]::IsNullOrWhiteSpace($RunName)) {
  $runDir = Join-Path $workspace (Join-Path 'runs' $RunName)
  $args += @('--run-dir', $runDir)
}
if (-not [string]::IsNullOrWhiteSpace($InitCkpt)) {
  $args += @('--init-ckpt', $InitCkpt)
}
if ($Resume) {
  $args += '--resume'
}

python @args
