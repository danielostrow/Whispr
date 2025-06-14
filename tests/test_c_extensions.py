#!/usr/bin/env python3
"""
Test script to verify Whispr C extensions are working correctly.
This will run a benchmark comparing C extensions vs. Python implementations.
"""

import os
import sys
import time

import numpy as np

# Add parent directory to path to ensure we can import whispr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import C extensions
try:
    # First try to import from the module
    try:
        from whispr.c_ext import (
            C_EXTENSIONS_AVAILABLE,
            compute_frame_energy,
            frame_signal_c,
            separate_by_segmentation_c,
            simple_energy_vad_c,
        )
    except ImportError:
        # If that fails, try direct imports
        from whispr.c_ext.features_c import compute_frame_energy
        from whispr.c_ext.framing_c import frame_signal_c
        from whispr.c_ext.separation_c import separate_by_segmentation_c
        from whispr.c_ext.vad_c import simple_energy_vad_c

        C_EXTENSIONS_AVAILABLE = True
except ImportError as e:
    print(f"❌ C extensions not available. Error: {e}")
    print("Run build_extensions.sh first.")
    sys.exit(1)

from whispr.config import Config

# Import Python implementations for comparison
from whispr.dsp.framing import frame_signal
from whispr.ml.vad import simple_energy_vad


def time_function(func, *args, **kwargs):
    """Measure execution time of a function."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


def test_framing():
    """Test frame_signal function."""
    print("\n--- Testing framing function ---")
    # Create a test signal (1 second at 16kHz)
    signal = np.random.randn(16000).astype(np.float32)
    frame_length = 400  # 25ms at 16kHz
    hop_length = 160  # 10ms at 16kHz

    # Time Python implementation
    py_frames, py_time = time_function(frame_signal, signal, frame_length, hop_length)

    # Time C implementation
    c_frames, c_time = time_function(frame_signal_c, signal, frame_length, hop_length)

    # Verify results are the same
    if np.allclose(py_frames, c_frames):
        print("✅ Framing results match")
    else:
        print("❌ Framing results differ")

    # Print timing comparison
    speedup = py_time / c_time if c_time > 0 else float("inf")
    print(f"Python time: {py_time:.6f}s")
    print(f"C time:      {c_time:.6f}s")
    print(f"Speedup:     {speedup:.2f}x")

    # Use assertion instead of return
    assert np.allclose(py_frames, c_frames), "Framing results don't match"


def test_energy():
    """Test energy calculation function."""
    print("\n--- Testing energy calculation ---")
    # Create frames
    signal = np.random.randn(16000).astype(np.float32)
    frame_length = 400
    hop_length = 160
    frames = frame_signal(signal, frame_length, hop_length)

    # Python implementation (manual)
    start = time.time()
    windowed = frames * np.hanning(frame_length)
    py_energy = np.sum(windowed**2, axis=1)
    py_time = time.time() - start

    # C implementation
    c_energy, c_time = time_function(compute_frame_energy, frames)

    # Verify results are the same
    if np.allclose(py_energy, c_energy, rtol=1e-5):
        print("✅ Energy results match")
    else:
        print("❌ Energy results differ")

    # Print timing comparison
    speedup = py_time / c_time if c_time > 0 else float("inf")
    print(f"Python time: {py_time:.6f}s")
    print(f"C time:      {c_time:.6f}s")
    print(f"Speedup:     {speedup:.2f}x")

    # Use assertion instead of return
    assert np.allclose(py_energy, c_energy, rtol=1e-5), "Energy results don't match"


def test_vad():
    """Test VAD function."""
    print("\n--- Testing VAD function ---")
    # Create energy values
    energy = np.random.exponential(1.0, 1000).astype(np.float32)
    energy[300:500] *= 5  # Create a speech segment
    energy[700:800] *= 5  # Create another speech segment

    # Config
    cfg = Config()
    vad_energy_threshold = cfg.vad_energy_threshold
    min_frames = int((cfg.min_speech_duration_s * 1000) / cfg.hop_length_ms)

    # Python implementation
    py_segments, py_time = time_function(simple_energy_vad, energy, cfg)

    # C implementation
    c_segments, c_time = time_function(
        simple_energy_vad_c, energy, vad_energy_threshold, min_frames
    )

    # Verify results are the same
    if py_segments == c_segments:
        print("✅ VAD results match")
    else:
        print("❌ VAD results differ")
        print(f"Python segments: {py_segments}")
        print(f"C segments: {c_segments}")

    # Print timing comparison
    speedup = py_time / c_time if c_time > 0 else float("inf")
    print(f"Python time: {py_time:.6f}s")
    print(f"C time:      {c_time:.6f}s")
    print(f"Speedup:     {speedup:.2f}x")

    # Use assertion instead of return
    assert py_segments == c_segments, "VAD results don't match"


def main():
    """Run all tests."""
    print("=== Testing Whispr C Extensions ===")
    print(f"C extensions available: {C_EXTENSIONS_AVAILABLE}")

    # Run tests
    test_framing()
    test_energy()
    test_vad()

    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()
