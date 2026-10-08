"""End-to-end CLI contract checks using tiny untrained fixture graphs, no downloads."""
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

exe, fixtures, output = map(lambda s: Path(s).resolve(), sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)


def run(assets=fixtures, extra=(), error=None, label="speech"):
    wav = output / (label + ".wav")
    result = subprocess.run([str(exe), "--assets", str(assets), "--text", "hello", "--output", str(wav), *extra],
                            capture_output=True, text=True)
    if error:
        assert result.returncode != 0 and error in result.stderr, result.stderr
        return None
    assert result.returncode == 0, result.stderr
    report = json.loads(wav.with_suffix(".diagnostics.json").read_text())
    with wave.open(str(wav)) as audio:
        assert audio.getframerate() == 24000 and audio.getsampwidth() == 2 and audio.getnchannels() == 1
        assert audio.getnframes() == report["frames"] * 256 == report["samples"]
    assert report["frames"] == sum(report["durations"]) == len(report["f0_hz"]) == len(report["energy"])
    return report


normal = run(label="normal")
fast = run(extra=["--speed", "2", "--pitch-scale", "1.1"], label="fast")
assert fast["frames"] * 2 == normal["frames"]
assert all(abs(value - 198) < 1e-3 for value in fast["f0_hz"])
for emotion in ("neutral", "whisper", "excited", "somber", "authoritative", "calm"):
    report = run(extra=["--emotion", emotion, "--speaker-id", "2"], label=emotion)
    assert report["emotion"] == emotion and report["speaker_id"] == 2
    if emotion == "neutral":
        assert report["durations"] == normal["durations"] and report["f0_hz"] == normal["f0_hz"]
    elif emotion == "whisper":
        assert report["frames"] > normal["frames"] and all(abs(value - 0.6) < 1e-5 for value in report["energy"])
    elif emotion == "excited":
        assert report["frames"] < normal["frames"] and all(abs(value - 215) < 1e-3 for value in report["f0_hz"])
    elif emotion in ("somber", "calm"):
        assert report["frames"] > normal["frames"] and max(report["f0_hz"]) <= 155
    else:
        assert report["frames"] < normal["frames"] and max(report["energy"]) > 1.3
run(extra=["--emotion", "angry"], error="unknown emotion")
composed = run(extra=["--emotion", "excited", "--pitch-scale", "2"], label="composed")
assert all(abs(value - 430) < 1e-3 for value in composed["f0_hz"])
durations = output / "durations.csv"
durations.write_text(",".join("0" if i == 1 else "2" for i in range(normal["tokens"])))
target = run(extra=["--durations", str(durations)], label="target")
assert target["durations"][1] == 0 and target["frames"] == (normal["tokens"] - 1) * 2
f0 = output / "f0.csv"
f0.write_text(",".join("0" if i % 2 else "100" for i in range(target["frames"])))
energy = output / "energy.csv"
energy.write_text(",".join("1" if i % 2 else "3" for i in range(target["frames"])))
contours = run(extra=["--durations", str(durations), "--f0", str(f0), "--energy", str(energy),
                      "--pitch-scale", "2", "--energy-variance", "0"], label="contours")
assert set(contours["f0_hz"]) == {0, 200} and set(contours["energy"]) == {2}
for option, value, error in [("--speed", "0", "pitch/speed"), ("--pitch-scale", "nan", "pitch/speed"),
                             ("--threads", "0", "thread count"), ("--speaker-id", "-1", "speaker ID"),
                             ("--speed", "1abc", "invalid numeric"), ("--hop-length", "128", "vocoder audio output")]:
    run(extra=[option, value], error=error)
run(extra=["--f0", str(f0)], error="F0 and energy")
run(extra=["--unknown", "1"], error="unknown option")
run(extra=["--speed"], error="missing value")
with tempfile.TemporaryDirectory(dir=output) as name:
    assets = Path(name)
    for file in fixtures.iterdir():
        if file.is_file(): shutil.copyfile(file, assets / file.name)
    for bad, error in [("integer", "float32"), ("nan", "non-finite acoustic"),
                        ("wrong_frames", "sum(durations)"), ("wrong_name", "missing required ONNX input")]:
        shutil.copyfile(fixtures / (bad + ".onnx"), assets / "acoustic_generator.onnx")
        run(assets=assets, error=error)
    shutil.copyfile(fixtures / "acoustic_generator.onnx", assets / "acoustic_generator.onnx")
    shutil.copyfile(fixtures / "wrong_hop.onnx", assets / "vocoder_hifigan.onnx")
    run(assets=assets, error="vocoder audio output")
print("Explicit CLI control, WAV, and incompatible-model checks passed")
