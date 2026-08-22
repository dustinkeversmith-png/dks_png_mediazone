#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "src" / "video" / "tests"
old = 'mission.provider_name << "\n";'
# file literally has newline between quotes
old2 = 'mission.provider_name << "\n'
for p in root.rglob("test_*.cpp"):
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    i = 0
    changed = False
    while i < len(lines):
        line = lines[i]
        if 'mission.provider_name << "' in line and not line.rstrip().endswith('";'):
            if i + 1 < len(lines) and lines[i + 1].strip() == '";':
                out.append('        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\\n";\n')
                i += 2
                changed = True
                continue
        out.append(line)
        i += 1
    text = "".join(out)
    text2 = text.replace("if (!atom.load(cli))", "if (!atom.load(cli, argc, argv))")
    if text2 != p.read_text(encoding="utf-8"):
        p.write_text(text2, encoding="utf-8")
        print("fixed", p)
