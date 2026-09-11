"""Independently check the frozen split, reference denominator, and WER report."""
import argparse
import json
from pathlib import Path
import re


def tokens(text):
    return [s for word in text.upper().split() if (s := re.sub("[^A-Z']", "", word))]


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        next_row = [i]
        for j, y in enumerate(b, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (x != y)))
        row = next_row
    return row[-1]


def main():
    root = Path(__file__).resolve().parents[1]       # src/captions/models/int8_zip
    repo_root = Path(__file__).resolve().parents[5]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=root / "artifacts/asr_streaming_test.json")
    parser.add_argument("--data", type=Path, default=repo_root / "data/audio/librispeech")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    rows = [json.loads(s) for s in Path(str(args.report) + ".jsonl").read_text().splitlines()]
    wavs = sorted(args.data.glob("*.wav"))
    assert len(wavs) == 600
    assert [r["file"] for r in rows] == [p.name for p in wavs[150:]]
    errors = count = 0
    for r, wav in zip(rows, wavs[150:]):
        reference = wav.with_suffix(".txt").read_text()
        assert r["reference"] == reference
        a, b = tokens(reference), tokens(r["hypothesis"])
        e = distance(a, b)
        assert e == r["S"] + r["D"] + r["I"]
        assert len(a) == r["N"]
        errors += e
        count += len(a)
    assert len(rows) == report["utterances"] == 450
    assert count == report["reference_words"] == 9650
    assert errors == report["substitutions"] + report["deletions"] + report["insertions"]
    assert abs(errors / count - report["wer"]) < 1e-9
    print(f"Verified all 450 test IDs and references: {errors}/{count} = {100*errors/count:.6f}% WER")


if __name__ == "__main__":
    main()
