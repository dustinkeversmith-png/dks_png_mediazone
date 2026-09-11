#!/usr/bin/env python3
"""
Test script to validate downloaded datasets and run LPC/Fourier methods.

Usage:
  python test_audio_methods.py              # Test all datasets
  python test_audio_methods.py digits       # Test only digits
  python test_audio_methods.py timit        # Test only TIMIT
  python test_audio_methods.py librispeech  # Test only LibriSpeech
"""

import sys
import subprocess
import json
from pathlib import Path
from collections import defaultdict


def list_audio_files(tier_dir: str) -> list:
    """List all .wav files in a tier directory (recursively)."""
    tier_path = Path(tier_dir)
    if not tier_path.exists():
        return []
    return sorted(tier_path.rglob("*.wav"))


def validate_dataset(tier_name: str, tier_dir: str) -> dict:
    """Validate a downloaded dataset tier."""
    print(f"\n{'='*70}")
    print(f"VALIDATING: {tier_name}")
    print(f"{'='*70}")
    
    wav_files = list_audio_files(tier_dir)
    
    if not wav_files:
        print(f"[XX] No .wav files found in {tier_dir}")
        return {"tier": tier_name, "status": "empty", "count": 0}
    
    print(f"[OK] Found {len(wav_files)} audio files")
    
    # Compute statistics
    stats = {
        "tier": tier_name,
        "status": "valid",
        "count": len(wav_files),
        "samples": {}
    }
    
    # Sample first, middle, and last files
    sample_indices = [0, len(wav_files) // 2, len(wav_files) - 1]
    for idx in sample_indices:
        wav_file = wav_files[idx]
        try:
            import soundfile as sf
            data, sr = sf.read(wav_file)
            duration = len(data) / sr
            stats["samples"][str(idx)] = {
                "file": str(wav_file.name),
                "sr": sr,
                "duration_sec": round(duration, 3),
                "samples": len(data)
            }
        except Exception as e:
            stats["samples"][str(idx)] = {"error": str(e)}
    
    print(f"  Sample files:")
    for sample_info in stats["samples"].values():
        if "error" not in sample_info:
            print(f"    {sample_info['file']}: {sample_info['duration_sec']}s @ {sample_info['sr']} Hz")
    
    return stats


def run_method_on_file(method_exe: str, wav_file: str, timeout: int = 30) -> dict:
    """Run LPC or Fourier method on a single WAV file."""
    try:
        result = subprocess.run(
            [method_exe, str(wav_file)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Parse output for vowel string
        output = result.stdout + result.stderr
        vowel_string = None
        
        for line in output.split("\n"):
            if "collapsed" in line.lower():
                # Extract vowel string from line like "Vowel string (collapsed): IY EH AA"
                parts = line.split(":")
                if len(parts) > 1:
                    vowel_string = parts[1].strip()
        
        return {
            "status": "success",
            "return_code": result.returncode,
            "vowels": vowel_string,
            "output": output[:500]  # First 500 chars
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "vowels": None}
    except Exception as e:
        return {"status": "error", "error": str(e), "vowels": None}


def test_tier(tier_name: str, tier_dir: str, lpc_exe: str, fourier_exe: str, max_samples: int = 5):
    """Test a dataset tier with both LPC and Fourier methods."""
    print(f"\n{'='*70}")
    print(f"TESTING TIER: {tier_name}")
    print(f"{'='*70}")
    
    wav_files = list_audio_files(tier_dir)
    if not wav_files:
        print(f"[XX] No audio files to test")
        return {}
    
    # Test up to max_samples
    test_files = wav_files[:max_samples]
    results = defaultdict(list)
    
    for wav_file in test_files:
        print(f"\n  Testing: {wav_file.name}")
        
        # Run LPC method
        if Path(lpc_exe).exists():
            lpc_result = run_method_on_file(lpc_exe, wav_file)
            print(f"    LPC:     {lpc_result.get('vowels', 'No output')}")
            results["lpc"].append({
                "file": wav_file.name,
                "vowels": lpc_result.get("vowels")
            })
        
        # Run Fourier method
        if Path(fourier_exe).exists():
            fourier_result = run_method_on_file(fourier_exe, wav_file)
            print(f"    Fourier: {fourier_result.get('vowels', 'No output')}")
            results["fourier"].append({
                "file": wav_file.name,
                "vowels": fourier_result.get("vowels")
            })
    
    return results


def main():
    """Main test suite."""
    print("\n" + "="*70)
    print("AUDIO ANALYSIS METHOD BENCHMARK")
    print("="*70)
    
    # Paths
    data_dir = Path("data/audio")
    lpc_exe = Path("build/bin/Release/lpc_method.exe")
    fourier_exe = Path("build/bin/Release/fourier_method.exe")
    
    # Check if executables exist
    if not lpc_exe.exists():
        print(f"[XX] LPC executable not found: {lpc_exe}")
        print("  Run: cmake --build build --target lpc_method --config Release")
        return 1
    
    if not fourier_exe.exists():
        print(f"[XX] Fourier executable not found: {fourier_exe}")
        print("  Run: cmake --build build --target fourier_method --config Release")
        return 1
    
    print(f"[OK] Found LPC method: {lpc_exe}")
    print(f"[OK] Found Fourier method: {fourier_exe}")
    
    # Determine which tiers to test
    test_all = len(sys.argv) == 1
    test_tiers = set(sys.argv[1:]) if not test_all else {"digits", "timit", "librispeech"}
    
    # Define tiers
    tiers = {
        "digits": data_dir / "digits",
        "timit": data_dir / "timit",
        "librispeech": data_dir / "librispeech",
    }
    
    results = {}
    
    # Validate datasets
    print("\n" + "="*70)
    print("STEP 1: VALIDATING DATASETS")
    print("="*70)
    
    for tier_name, tier_dir in tiers.items():
        if tier_name in test_tiers or test_all:
            validation = validate_dataset(tier_name, str(tier_dir))
            results[tier_name] = validation
    
    # Test with both methods
    print("\n" + "="*70)
    print("STEP 2: TESTING WITH LPC AND FOURIER METHODS")
    print("="*70)
    
    method_results = {}
    for tier_name, tier_dir in tiers.items():
        if tier_name in test_tiers or test_all:
            tier_results = test_tier(
                tier_name,
                str(tier_dir),
                str(lpc_exe),
                str(fourier_exe),
                max_samples=3
            )
            method_results[tier_name] = tier_results
    
    # Summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    for tier_name, validation in results.items():
        print(f"\n{tier_name.upper()}:")
        print(f"  Status: {validation.get('status', 'unknown')}")
        print(f"  Audio files: {validation.get('count', 0)}")
        
        if validation.get("status") == "valid":
            # Show one sample
            for sample_info in list(validation.get("samples", {}).values())[:1]:
                if "error" not in sample_info:
                    print(f"  Sample: {sample_info['file']} ({sample_info['duration_sec']}s)")
        
        # Show method results
        if tier_name in method_results and method_results[tier_name]:
            lpc_count = len([r for r in method_results[tier_name].get("lpc", []) if r.get("vowels")])
            fourier_count = len([r for r in method_results[tier_name].get("fourier", []) if r.get("vowels")])
            print(f"  LPC successful: {lpc_count}")
            print(f"  Fourier successful: {fourier_count}")
    
    print("\n" + "="*70)
    print("[OK] BENCHMARK COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review vowel outputs for accuracy")
    print("  2. Compare LPC vs Fourier method performance")
    print("  3. Adjust consonant thresholds if needed")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
