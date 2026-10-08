"""Render reusable voice profiles with the existing explicit-tts executable.

No downloads or Python ML dependencies. Each profile becomes a normal CLI call.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=ROOT / "examples/vocal_profiles/profiles.json")
    parser.add_argument("--profile", action="append", help="Render one named profile; repeat to select several")
    parser.add_argument("--list", action="store_true", help="Print profile names without synthesizing")
    parser.add_argument("--text", default="Hello there. The little bird sings beside the window.")
    parser.add_argument("--speaker-id", type=int, help="Use one speaker for all selected profiles")
    parser.add_argument("--assets", type=Path, default=ROOT / "model_assets/explicit_neural")
    parser.add_argument("--bin", type=Path, default=ROOT / "build/onnx/Release/explicit-tts.exe")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/vocal_profiles")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    names = args.profile or list(profiles)
    if args.list:
        print("\n".join(profiles))
        return
    for name in names:
        if name not in profiles: parser.error(f"unknown profile: {name}; use --list")
        if Path(name).name != name or any(char in name for char in "\\/:"):
            parser.error("profile names must be simple filenames")
    args.output.mkdir(parents=True, exist_ok=True)
    rendered = []
    for name in names:
        settings = dict(profiles[name])
        if args.speaker_id is not None: settings["speaker_id"] = args.speaker_id
        output = args.output / (name + ".wav")
        command = [str(args.bin.resolve()), "--assets", str(args.assets.resolve()),
                   "--text", args.text, "--output", str(output.resolve()), "--threads", str(args.threads)]
        for key, value in settings.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        print(f"Rendering {name}", flush=True)
        subprocess.run(command, check=True)
        rendered.append({"profile": name, "settings": settings, "command": command,
                         "wav": str(output.resolve()), "diagnostics": str(output.with_suffix(".diagnostics.json").resolve())})
    (args.output / "profiles_manifest.json").write_text(json.dumps(
        {"text": args.text, "renders": rendered}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
