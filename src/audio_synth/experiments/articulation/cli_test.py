"""Local trained integration test for the experimental C++ executable; no downloads."""
import json
import subprocess
import sys
import wave
from pathlib import Path

exe, assets, output = (Path(value).resolve() for value in sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)

def run(name, flags=(), error=None):
    wav = output / (name + ".wav")
    result = subprocess.run([str(exe), "--assets", str(assets), "--text",
        "We synthesize a clear acoustic voice.", "--output", str(wav), *flags], capture_output=True, text=True)
    if error:
        assert result.returncode and error in result.stderr, result.stderr
        assert not wav.exists()
        return
    assert result.returncode == 0, result.stderr
    d = json.loads(wav.with_suffix(".diagnostics.json").read_text())
    with wave.open(str(wav)) as w:
        assert w.getframerate() == 24000 and w.getnframes() == d["frames"] * 300
    return d, wav.read_bytes()

base, wav = run("baseline")
repeat, repeat_wav = run("repeat")
assert wav == repeat_wav
edited, changed = run("local-pitch", ["--word-index", "3", "--pitch-scale", "1.12"])
assert changed != wav and edited["durations"] == base["durations"]
span = base["words"][3]
assert span["text"] == "clear"
for i, (old, new) in enumerate(zip(base["f0_hz"], edited["f0_hz"])):
    inside = span["frame_begin"] <= i < span["frame_end"]
    assert abs(new - old * (1.12 if inside else 1)) < .001
phone, _ = run("exact-phone", ["--word-index", "1", "--phone", "S", "--duration-scale", "2"])
for i, (old, new) in enumerate(zip(base["durations"], phone["durations"])):
    targeted = base["words"][1]["token_begin"] <= i < base["words"][1]["token_end"] and base["phones"][i] == "S"
    assert new == old * (2 if targeted else 1)
run("bad-index", ["--word-index", "90"], error="word index")
run("bad-phone", ["--phone", "CLICK"], error="matched no tokens")
run("bad-range", ["--pitch-scale", "nan"], error="bounds")
result = subprocess.run([str(exe), "--assets", str(assets) + "/", "--text", "Hello.",
    "--output", str(assets / "forbidden-experiment.wav")], capture_output=True, text=True)
assert result.returncode and "outside model assets" in result.stderr, result.stderr
assert not (assets / "forbidden-experiment.wav").exists()
print("Experimental C++ repeatability, word/phone isolation, WAV and invalid-request checks passed")
