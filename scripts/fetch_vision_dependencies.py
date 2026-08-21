"""Fetch C++ vision dependencies into ./dependencies (skip if already present)."""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "dependencies"
DEPS.mkdir(exist_ok=True)

UA = {"User-Agent": "generative-media-research/1.0"}


def have(path: Path) -> bool:
    return path.exists()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] GET {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
        print(f"[ok] {dest}")


def git_clone(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    print(f"[*] git clone --depth 1 {url}")
    subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])


def main() -> int:
    print("Vision dependency check (fastest = reuse existing copies)\n")

    status = []

    eigen = DEPS / "eigen" / "Eigen" / "Dense"
    status.append(("Eigen", have(eigen), str(DEPS / "eigen")))
    if not have(eigen):
        git_clone("https://gitlab.com/libeigen/eigen.git", DEPS / "eigen")

    stb = DEPS / "stb" / "stb_image.h"
    status.append(("stb_image.h", have(stb), str(DEPS / "stb")))
    if not have(stb):
        download(
            "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h",
            stb,
        )
        download(
            "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h",
            DEPS / "stb" / "stb_image_write.h",
        )

    fftw = DEPS / "fftw" / "fftw3.h"
    status.append(("FFTW3", have(fftw), str(DEPS / "fftw")))
    if not have(fftw):
        print("[!] FFTW3 missing. Windows: unzip https://www.fftw.org/install/windows.html into dependencies/fftw")
        print("    Linux: sudo apt install libfftw3-dev")

    ffmpeg_h = DEPS / "ffmpeg" / "include" / "libavcodec" / "avcodec.h"
    status.append(("FFmpeg", have(ffmpeg_h), str(DEPS / "ffmpeg")))
    if not have(ffmpeg_h):
        print("[!] FFmpeg missing. Fastest on Windows: gyan.dev shared build → dependencies/ffmpeg/{include,lib,bin}")

    sidx = DEPS / "libspatialindex" / "include" / "spatialindex" / "SpatialIndex.h"
    status.append(("libspatialindex", have(sidx), str(DEPS / "libspatialindex")))
    if not have(sidx):
        git_clone(
            "https://github.com/libspatialindex/libspatialindex.git",
            DEPS / "libspatialindex",
        )

    yacc = DEPS / "YACCLAB" / "CMakeLists.txt"
    status.append(("YACCLAB (optional)", have(yacc), str(DEPS / "YACCLAB")))
    if "--yacclab" in sys.argv and not have(yacc):
        git_clone("https://github.com/prittt/YACCLAB.git", DEPS / "YACCLAB")
        print("[i] YACCLAB needs OpenCV to build the full harness. Unit tests use in-tree SAUF CCL.")

    dt = ROOT / "src" / "video" / "math" / "distance_transform.hpp"
    status.append(("Felzenszwalb–Huttenlocher DT", have(dt), str(dt)))

    print("\nStatus:")
    for name, ok, path in status:
        flag = "OK " if ok else "MISS"
        print(f"  [{flag}] {name:32s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
