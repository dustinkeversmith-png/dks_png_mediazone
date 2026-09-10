#!/usr/bin/env bash
# Quick setup script for downloading and testing audio datasets

set -e  # Exit on error

echo "=========================================="
echo "Audio Dataset & Method Setup"
echo "=========================================="

# Step 1: Install Python dependencies
echo -e "\n[1/4] Installing Python dependencies..."
pip install -r requirements_audio.txt || {
    echo "WARNING: Some dependencies may have failed to install"
}
echo "✓ Dependencies installation attempted"

# Step 2: Ensure build directory exists
echo -e "\n[2/4] Building LPC and Fourier methods..."
if [ ! -d "build" ]; then
    mkdir build
    cd build
    cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
    cd ..
fi

cd build
cmake --build . --target lpc_method --config Release 2>&1 | tail -5
cmake --build . --target fourier_method --config Release 2>&1 | tail -5
cd ..

if [ -f "build/bin/Release/lpc_method" ] || [ -f "build/bin/Release/lpc_method.exe" ]; then
    echo "✓ LPC method built"
else
    echo "✗ LPC method build failed"
    exit 1
fi

if [ -f "build/bin/Release/fourier_method" ] || [ -f "build/bin/Release/fourier_method.exe" ]; then
    echo "✓ Fourier method built"
else
    echo "✗ Fourier method build failed"
    exit 1
fi

# Step 3: Download datasets
echo -e "\n[3/4] Downloading audio datasets (max 250 MB each)..."
echo "This may take 5-30 minutes depending on internet speed."
echo ""

python scripts/download_datasets.py

# Step 4: Validate and test
echo -e "\n[4/4] Running validation tests..."
python scripts/test_audio_methods.py

echo -e "\n=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Datasets downloaded to:"
echo "  • data/audio/digits/        (Google Speech Commands)"
echo "  • data/audio/timit/         (TIMIT - if available)"
echo "  • data/audio/librispeech/   (LibriSpeech test-clean)"
echo ""
echo "Next steps:"
echo "  1. Review the method outputs above"
echo "  2. Compare LPC vs Fourier results"
echo "  3. Read data/audio/README.md for detailed analysis"
echo "  4. See START_HERE.md for full documentation"
echo ""
