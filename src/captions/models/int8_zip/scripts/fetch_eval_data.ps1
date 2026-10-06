# Fetches the robustness evaluation data (native tools only: curl + ffmpeg).
#
#   data/audio/eval/<set>/NNNNNN.wav + .txt   out-of-domain speech, <=100 MB each
#   data/audio/noise/<env>.wav                DEMAND noise (channel 1), for SNR mixing
#
# Speech comes from the Hugging Face dataset viewer API, sampled evenly across
# each test split so many speakers/sessions are covered. Rerunning skips
# anything already present. Usage (repository root):
#   ./src/captions/models/int8_zip/scripts/fetch_eval_data.ps1 [-PerSet 300] [-Only ami,tedlium]
param(
    [int]$PerSet = 300,
    [int]$MaxMegabytes = 100,
    [string[]]$Only = @()
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).Path
$ffmpeg = Join-Path $repo 'dependencies/ffmpeg/bin/ffmpeg.exe'
if (!(Test-Path $ffmpeg)) { $ffmpeg = 'ffmpeg' }

# name, dataset, config, split, transcript field, minimum words
# Open ASR leaderboard test sets (hf-audio/open-asr-leaderboard) where the
# viewer serves them; AMI comes from its own repository (IHM headset mics).
$leaderboard = 'hf-audio/open-asr-leaderboard'
$sets = @(
    @{ name = 'ami';               dataset = 'edinburghcstr/ami'; config = 'ihm';          split = 'test';       field = 'text'; min = 4 },
    # Same meetings recorded by one distant table microphone: real room noise,
    # reverberation and overlapping talkers.
    @{ name = 'ami_sdm';           dataset = 'edinburghcstr/ami'; config = 'sdm';          split = 'test';       field = 'text'; min = 4; count = 900 },
    @{ name = 'gigaspeech';        dataset = $leaderboard;        config = 'gigaspeech';   split = 'test';       field = 'text'; min = 3 },
    @{ name = 'common_voice';      dataset = $leaderboard;        config = 'common_voice'; split = 'test';       field = 'text'; min = 3 },
    @{ name = 'earnings22';        dataset = $leaderboard;        config = 'earnings22';   split = 'test';       field = 'text'; min = 3 },
    @{ name = 'voxpopuli';         dataset = $leaderboard;        config = 'voxpopuli';    split = 'test';       field = 'text'; min = 3 },
    @{ name = 'librispeech_other'; dataset = $leaderboard;        config = 'librispeech';  split = 'test.other'; field = 'text'; min = 1 }
)
$noise = @('OMEETING', 'DLIVING', 'NPARK')

function Get-Json([string]$url) {
    for ($try = 0; $try -lt 4; ++$try) {
        try { return Invoke-RestMethod -Uri $url -TimeoutSec 120 } catch { Start-Sleep -Seconds (2 + 3 * $try) }
    }
    throw "Failed: $url"
}

foreach ($s in $sets) {
    if ($Only.Count -and $Only -notcontains $s.name) { continue }
    $target = if ($s.count) { $s.count } else { $PerSet }
    $dir = Join-Path $repo "data/audio/eval/$($s.name)"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $have = @(Get-ChildItem $dir -Filter *.wav -ErrorAction SilentlyContinue).Count
    if ($have -ge $target) { Write-Host "$($s.name): $have clips present"; continue }
    $base = "https://datasets-server.huggingface.co"
    $q = "dataset=$([uri]::EscapeDataString($s.dataset))&config=$($s.config)&split=$($s.split)"
    $total = (Get-Json "$base/rows?$q&offset=0&length=1").num_rows_total
    # Pages spread evenly over the split; a few rows from each page.
    $pages = [Math]::Max(1, [Math]::Ceiling($target / 10))
    $step = [Math]::Max(100, [Math]::Floor($total / $pages))
    $count = $have; $bytes = (Get-ChildItem $dir -Filter *.wav | Measure-Object Length -Sum).Sum
    for ($offset = 0; $offset -lt $total -and $count -lt $target; $offset += $step) {
        $rows = (Get-Json "$base/rows?$q&offset=$offset&length=100").rows
        $taken = 0
        foreach ($r in $rows) {
            if ($taken -ge 10 -or $count -ge $target) { break }
            $text = [string]$r.row.($s.field)
            $words = @($text -split '\s+' | Where-Object { $_ -and $_ -notmatch '^<.*>$' }).Count
            if ($words -lt $s.min) { continue }
            $id = '{0:D6}' -f $r.row_idx
            $wav = Join-Path $dir "$id.wav"
            if (Test-Path $wav) { continue }
            $src = $r.row.audio[0].src
            $tmp = Join-Path $env:TEMP "eval_$($s.name)_$id.bin"
            curl.exe -sL -o $tmp $src
            & $ffmpeg -nostdin -v error -y -i $tmp -ac 1 -ar 16000 -c:a pcm_s16le $wav
            Remove-Item -LiteralPath $tmp -Force
            [IO.File]::WriteAllText((Join-Path $dir "$id.txt"), $text.Trim())
            $bytes += (Get-Item $wav).Length; $count++; $taken++
            if ($bytes -ge $MaxMegabytes * 1MB) { break }
        }
        if ($bytes -ge $MaxMegabytes * 1MB) { break }
        Write-Host ("{0}: {1} clips, {2:N1} MB" -f $s.name, $count, ($bytes / 1MB))
    }
    Write-Host ("{0}: done, {1} clips, {2:N1} MB" -f $s.name, $count, ($bytes / 1MB))
}

if ($Only.Count -eq 0 -or $Only -contains 'noise') {
    $ndir = Join-Path $repo 'data/audio/noise'
    New-Item -ItemType Directory -Force $ndir | Out-Null
    foreach ($env in $noise) {
        $out = Join-Path $ndir "$env.wav"
        if (Test-Path $out) { Write-Host "noise $env present"; continue }
        $zip = Join-Path $env:TEMP "DEMAND_$env.zip"
        curl.exe -sL -o $zip "https://zenodo.org/records/1227121/files/$($env)_16k.zip?download=1"
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $archive = [IO.Compression.ZipFile]::OpenRead($zip)
        try {
            $entry = $archive.Entries | Where-Object { $_.FullName -match 'ch01\.wav$' } | Select-Object -First 1
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $out, $true)
        } finally { $archive.Dispose() }
        Remove-Item -LiteralPath $zip -Force
        Write-Host "noise $env -> $out"
    }
}
