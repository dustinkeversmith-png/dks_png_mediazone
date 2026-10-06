# Downloads the caption models (native tools only: curl + tar), pinned by
# URL and SHA-256. Rerunning skips models that are already present.
#
#   models/parakeet-tdt-110m/   batch captions: NVIDIA Parakeet TDT 110M (INT8),
#                               36k hours of multi-domain English, punctuation + casing
#   models/silero-vad/          Silero VAD v5 (MIT) for segmentation and hallucination filtering
#   models/nemo-streaming-480ms/ live captions: NeMo cache-aware streaming FastConformer (INT8),
#                               multi-domain English, 560 ms chunks
#   (models/librispeech, the older streaming Zipformer2, is optional: fetch_streaming_asr.py)
#
# Usage (repository root): ./src/captions/models/int8_zip/scripts/fetch_models.ps1
$ErrorActionPreference = 'Stop'
$models = Join-Path (Split-Path -Parent $PSScriptRoot) 'models'

function Get-Verified([string]$url, [string]$sha256, [string]$out) {
    curl.exe -sL --fail -o $out $url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $url" }
    $actual = (Get-FileHash -Algorithm SHA256 $out).Hash
    if ($actual -ne $sha256) {
        Remove-Item -LiteralPath $out -Force
        throw "Checksum mismatch for $url`n  expected $sha256`n  actual   $actual"
    }
}

# Parakeet TDT 110M, sherpa-onnx INT8 export (CC-BY-4.0, NVIDIA).
$parakeet = Join-Path $models 'parakeet-tdt-110m'
if (!(Test-Path (Join-Path $parakeet 'encoder.int8.onnx'))) {
    $name = 'sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8'
    $archive = Join-Path $env:TEMP "$name.tar.bz2"
    Get-Verified "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$name.tar.bz2" `
        'F628312E9FDF8686374CB01A69425C41732529D540860311F16F37CBC32CFE9B' $archive
    tar -xjf $archive -C $models
    if ($LASTEXITCODE -ne 0) { throw "Extracting $archive failed" }
    Remove-Item -LiteralPath $archive -Force
    if (Test-Path $parakeet) { Remove-Item -LiteralPath $parakeet -Recurse -Force }
    Rename-Item (Join-Path $models $name) 'parakeet-tdt-110m'
    Write-Host "Parakeet TDT 110M -> $parakeet"
} else { Write-Host "Parakeet TDT 110M present" }

# Silero VAD v5 ONNX, pinned to an onnx-community/silero-vad revision.
$vad = Join-Path $models 'silero-vad'
if (!(Test-Path (Join-Path $vad 'silero_vad.onnx'))) {
    New-Item -ItemType Directory -Force $vad | Out-Null
    $rev = 'e71cae966052b992a7eca6b17738916ce0eca4ec'
    Get-Verified "https://huggingface.co/onnx-community/silero-vad/resolve/$rev/onnx/model.onnx" `
        'A4A068CD6CF1EA8355B84327595838CA748EC29A25BC91FC82E6C299CCDC5808' (Join-Path $vad 'silero_vad.onnx')
    curl.exe -sL --fail -o (Join-Path $vad 'LICENSE') "https://huggingface.co/onnx-community/silero-vad/resolve/$rev/LICENSE"
    Write-Host "Silero VAD -> $vad"
} else { Write-Host "Silero VAD present" }

# NeMo cache-aware streaming FastConformer (stt_en_fastconformer_hybrid_large_streaming_480ms,
# NVIDIA, CC-BY-4.0), sherpa-onnx INT8 export, pinned to a Hugging Face revision.
$live = Join-Path $models 'nemo-streaming-480ms'
if (!(Test-Path (Join-Path $live 'tokens.txt'))) {
    New-Item -ItemType Directory -Force $live | Out-Null
    $repo = 'csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8'
    $rev = 'df8ed95e44a70924450381e610770f9d656d1e15'
    $files = @{
        'encoder.int8.onnx' = '100C5616929A131B5B3C8C8AB0D83AABA2CDCAE163ACD8D190B4E5FFA5F7D051'
        'decoder.int8.onnx' = '4F04431988472F8C7B815F942AA6901976929C9E22353029681FCDB262DA0164'
        'joiner.int8.onnx'  = 'D2D8E7290A6A8245A83ED211AA0DD8CDF8EFFB8A7C64FE392399561111B52F30'
        'tokens.txt'        = '618DC110FC2213886B52E063FF42329BBDF37A266CA7705184090FA5F39F3131'
    }
    foreach ($f in $files.Keys) {
        Get-Verified "https://huggingface.co/$repo/resolve/$rev/$f" $files[$f] (Join-Path $live $f)
    }
    Write-Host "NeMo streaming FastConformer -> $live"
} else { Write-Host "NeMo streaming FastConformer present" }
