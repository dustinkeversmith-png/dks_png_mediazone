"""Compose a labelled contact sheet from an atom's .pgm artifact directory.

Usage: python scripts/qa_sheet.py <artifact_dir> <out.png> [--scale N] [--stems a,b,c]
       [--suffixes 00_input_base,07_final_overlay]
"""

import argparse
import os
import re
import sys

from PIL import Image, ImageDraw


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir")
    ap.add_argument("out")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--stems", default="")
    ap.add_argument("--suffixes", default="")
    return ap.parse_args()


def index_artifacts(directory):
    """Return (stems in order, suffixes in order, {(stem, suffix): path})."""
    table = {}
    stems, suffixes = [], []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".pgm"):
            continue
        # Greedy stem so that sample ids containing underscores (ADE_00000039)
        # still split at the stage suffix rather than at their own separator.
        match = re.match(r"^(.+)_((?:\d{2}|gt)_.+)\.pgm$", name)
        if not match:
            continue
        stem, suffix = match.group(1), match.group(2)
        if stem not in stems:
            stems.append(stem)
        if suffix not in suffixes:
            suffixes.append(suffix)
        table[(stem, suffix)] = os.path.join(directory, name)
    return stems, suffixes, table


def main():
    args = parse_args()
    stems, suffixes, table = index_artifacts(args.artifact_dir)
    if args.stems:
        wanted = args.stems.split(",")
        stems = [s for s in stems if s in wanted]
    if args.suffixes:
        wanted = args.suffixes.split(",")
        suffixes = [s for s in wanted if s in suffixes]
    if not stems or not suffixes:
        sys.exit("no artifacts matched")

    scale = args.scale
    pad, header, label_h = 6, 22, 16
    tiles = {}
    cell_w = cell_h = 0
    for stem in stems:
        for suffix in suffixes:
            path = table.get((stem, suffix))
            if path is None:
                continue
            image = Image.open(path).convert("L")
            image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
            tiles[(stem, suffix)] = image
            cell_w = max(cell_w, image.width)
            cell_h = max(cell_h, image.height)

    sheet_w = pad + len(suffixes) * (cell_w + pad)
    sheet_h = header + len(stems) * (cell_h + label_h + pad)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)

    for col, suffix in enumerate(suffixes):
        draw.text((pad + col * (cell_w + pad), 6), suffix, fill=(220, 220, 120))
    for row, stem in enumerate(stems):
        y = header + row * (cell_h + label_h + pad)
        draw.text((pad, y), stem, fill=(120, 220, 220))
        for col, suffix in enumerate(suffixes):
            tile = tiles.get((stem, suffix))
            if tile is None:
                continue
            sheet.paste(tile, (pad + col * (cell_w + pad), y + label_h))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sheet.save(args.out)
    print(f"{args.out}  {sheet.width}x{sheet.height}  stems={len(stems)} cols={len(suffixes)}")


if __name__ == "__main__":
    main()
