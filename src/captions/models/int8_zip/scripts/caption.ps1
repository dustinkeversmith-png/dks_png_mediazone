# Launcher for the `captions` command line (batch files/video and live mic).
# Examples: ./scripts/caption.ps1 --input lecture.mp4
#           ./scripts/caption.ps1 --mic
#           ./scripts/caption.ps1 --help
$ErrorActionPreference = 'Stop'
$modelRoot = Split-Path -Parent $PSScriptRoot                       # .../captions/models/int8_zip
$repoRoot = (Resolve-Path (Join-Path $modelRoot '../../../..')).Path
$captionExe = Join-Path $repoRoot 'build/bin/Release/captions.exe'
if (!(Test-Path -LiteralPath $captionExe)) {
    throw "Build captions first (cmake -S . -B build; cmake --build build --config Release --target captions); see $modelRoot/README.md."
}
& $captionExe @args
exit $LASTEXITCODE
