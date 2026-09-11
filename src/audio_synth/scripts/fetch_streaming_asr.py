"""Fetch the selected streaming ASR checkpoint; verify every artifact with SHA256.

Python 3.12+, standard library only. --runtime also installs the pinned Windows
ONNX Runtime SDK and kaldi-native-fbank source dependency for the C++ build.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
REVISION = "d42f2d9f7ca24806fb667456a18a9f1b60f70d16"
KNF_REVISION = "b09e686fe2084732ddd30d1ef80acfc0f13eaf01"
FILES = {
    "encoder-epoch-99-avg-1.int8.onnx": "3810755ce7c3ab26b42a8bcf39d191308fa27fb0f53358823ba46141d03b7eb3",
    "decoder-epoch-99-avg-1.int8.onnx": "21e2a2acd961b3ac72f55be2f10f1a285e1b0b0ba010d7c0b6eab141411b163c",
    "joiner-epoch-99-avg-1.int8.onnx": "e085d73b593cf9b0707f370dbd656d58327d3fe36d80d849202ef81df02cb01e",
    "tokens.txt": "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
    "README.md": "505f6b0e8a39f066a0794c4fb0b5689533d3bcd9d1dc5e5f47ccffeef1af9877",
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or digest(path) != expected:
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        if digest(temporary) != expected:
            raise RuntimeError(f"Checksum mismatch: {url}")
        temporary.replace(path)
    print(f"Verified {path.name}: {path.stat().st_size:,} bytes", flush=True)
    return {"url": url, "sha256": expected, "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "artifacts/models/asr_streaming_int8/compact"
    manifest = {"repository": REPO, "revision": REVISION, "license": "Apache-2.0 (upstream model card)",
                "architecture": "streaming Zipformer transducer, 20M", "files": {}}
    for name, expected in FILES.items():
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        manifest["files"][name] = fetch(url, destination / name, expected)
    (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.runtime:
        deps = ROOT / "third_party"
        ort = deps / "ort.zip"
        fetch("https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip",
              ort, "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd")
        with zipfile.ZipFile(ort) as archive:
            for name in archive.namelist():
                if not (deps / name).resolve().is_relative_to(deps.resolve()):
                    raise RuntimeError("Unsafe runtime archive member")
            archive.extractall(deps)
        knf = deps / "knf.tar.gz"
        fetch(f"https://api.github.com/repos/csukuangfj/kaldi-native-fbank/tarball/{KNF_REVISION}",
              knf, "a83c122310b278119629004b57a7796f908a40fca6d9b9263db1760063105603")
        with tarfile.open(knf) as archive:
            archive.extractall(deps, filter="data")


if __name__ == "__main__":
    main()
