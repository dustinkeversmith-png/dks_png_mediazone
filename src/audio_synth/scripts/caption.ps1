# Streaming neural captions are the default for this entry point.
# Examples: ./scripts/caption.ps1 --file recording.wav
#           ./scripts/caption.ps1 --mic
$ErrorActionPreference = 'Stop'
$captionRoot = Split-Path -Parent $PSScriptRoot
$captionExe = Join-Path $captionRoot 'build/asr/Release/caption-streaming.exe'
if (!(Test-Path -LiteralPath $captionExe)) {
    throw 'Build caption-streaming first; see docs/STREAMING_ASR.md.'
}
$captionModels = Join-Path $captionRoot 'artifacts/models/asr_streaming_int8/compact'
if ($args -contains '--models') {
    & $captionExe @args
} else {
    & $captionExe --models $captionModels @args
}
exit $LASTEXITCODE
