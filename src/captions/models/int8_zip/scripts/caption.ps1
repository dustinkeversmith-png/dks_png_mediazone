# Streaming neural captions (models/int8_zip) are the default entry point.
# Examples: ./scripts/caption.ps1 --file recording.wav
#           ./scripts/caption.ps1 --mic
$ErrorActionPreference = 'Stop'
$modelRoot = Split-Path -Parent $PSScriptRoot                       # .../captions/models/int8_zip
$repoRoot = (Resolve-Path (Join-Path $modelRoot '../../../..')).Path
$captionExe = Join-Path $repoRoot 'build/bin/Release/caption-streaming.exe'
if (!(Test-Path -LiteralPath $captionExe)) {
    throw "Build caption-streaming first (cmake -S . -B build -DCAPTIONS_ENABLE_STREAMING_ASR=ON); see $modelRoot/STREAMING_ASR.md."
}
$captionModels = Join-Path $modelRoot 'models/compact'
if ($args -contains '--models') {
    & $captionExe @args
} else {
    & $captionExe --models $captionModels @args
}
exit $LASTEXITCODE
