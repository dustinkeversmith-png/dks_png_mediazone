"""Local trained-voice integration checks; never downloads assets."""
import hashlib
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

exe, assets, output = (Path(value).resolve() for value in sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)

def run(label, extra=(), root=assets, text="We synthesize a clear acoustic voice."):
    wav = output / (label + ".wav")
    result = subprocess.run([str(exe), "--assets", str(root), "--text", text,
                             "--output", str(wav), "--threads", "4", *extra],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(wav.with_suffix(".diagnostics.json").read_text())
    assert report["prosody_source"] == "trained-vctk-predictor"
    assert report["dictionary_hits"] > 0 and report["fallback_words"] == 0
    assert report["frames"] == sum(report["durations"]) == len(report["f0_hz"]) == len(report["energy"])
    with wave.open(str(wav)) as audio:
        assert audio.getframerate() == 24000 and audio.getsampwidth() == 2
        assert audio.getnframes() == report["frames"] * 300
    return report, hashlib.sha256(wav.read_bytes()).hexdigest()

neutral, normal_hash = run("neutral")
excited, excited_hash = run("excited", ["--emotion", "excited"])
somber, somber_hash = run("somber", ["--emotion", "somber"])
assert excited["frames"] < neutral["frames"] < somber["frames"]
assert sum(excited["f0_hz"]) / excited["frames"] > sum(neutral["f0_hz"]) / neutral["frames"]
assert len({normal_hash, excited_hash, somber_hash}) == 3
_, alternate_hash = run("alternate-speaker", ["--speaker-id", "1"])
assert alternate_hash != normal_hash
run("different-text", text="The little bird sings beside the window.")
# Adjacent shared CMUdict, while preserving the model's own vocabulary.
with tempfile.TemporaryDirectory(dir=output) as temporary:
    parent = Path(temporary)
    local = parent / "voice"
    shared = parent / "shared"
    local.mkdir(); shared.mkdir()
    for file in assets.iterdir():
        if file.is_file() and file.name != "cmudict.dict":
            (local / file.name).hardlink_to(file)
    (shared / "cmudict.dict").hardlink_to(assets / "cmudict.dict")
    _, fallback_hash = run("shared-dictionary", root=local)
    assert fallback_hash == normal_hash
print("Trained speech, cadence, pitch, speaker, and shared-dictionary checks passed")
