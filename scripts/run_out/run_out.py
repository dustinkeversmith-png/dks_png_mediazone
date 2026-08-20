import subprocess
from pathlib import Path

EXE_PATH = "./build-samples/bin/Release/input_pipeline.exe"
DATA_DIR = Path("./data/audiomnist")
OUTPUT_FILE = "results.txt"

wav_files = sorted(DATA_DIR.glob("*.wav"))

with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
    for wav in wav_files:
        print(f"Processing: {wav.name}")
        
        # Execute binary and capture stdout
        result = subprocess.run(
            [EXE_PATH, str(wav.resolve())],
            capture_output=True,
            text=True
        )
        
        # Write formatted result to file
        out.write(f"--- File: {wav.name} ---\n")
        out.write(result.stdout)
        out.write("\n")