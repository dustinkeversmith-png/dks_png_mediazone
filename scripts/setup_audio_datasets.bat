@echo off
REM Quick setup script for Windows
REM Downloads and tests audio datasets with LPC and Fourier methods

setlocal enabledelayedexpansion

echo ==========================================
echo Audio Dataset ^& Method Setup (Windows)
echo ==========================================

REM Step 1: Check Python dependencies
echo.
echo [1/4] Checking Python dependencies...
python -c "import soundfile, librosa, datasets" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install soundfile librosa datasets
) else (
    echo Python dependencies OK
)

REM Step 2: Build methods
echo.
echo [2/4] Building LPC and Fourier methods...

if not exist "build" (
    mkdir build
    cd build
    cmake .. -G "Visual Studio 17 2022"
    cd ..
)

cd build
cmake --build . --target lpc_method --config Release 2>&1 | findstr /c:"lpc_method"
cmake --build . --target fourier_method --config Release 2>&1 | findstr /c:"fourier_method"
cd ..

if exist "build\bin\Release\lpc_method.exe" (
    echo OK: LPC method built
) else (
    echo ERROR: LPC method build failed
    exit /b 1
)

if exist "build\bin\Release\fourier_method.exe" (
    echo OK: Fourier method built
) else (
    echo ERROR: Fourier method build failed
    exit /b 1
)

REM Step 3: Download datasets
echo.
echo [3/4] Downloading audio datasets ^(max 250 MB each^)...
echo This may take 5-30 minutes depending on internet speed.
echo.

python scripts\download_datasets.py

REM Step 4: Validate and test
echo.
echo [4/4] Running validation tests...
python scripts\test_audio_methods.py

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Datasets downloaded to:
echo   * data\audio\digits\        ^(Google Speech Commands^)
echo   * data\audio\timit\         ^(TIMIT - if available^)
echo   * data\audio\librispeech\   ^(LibriSpeech test-clean^)
echo.
echo Next steps:
echo   1. Review the method outputs above
echo   2. Compare LPC vs Fourier results
echo   3. Read data\audio\README.md for detailed documentation
echo.
