#!/usr/bin/env python3
"""
Simple test script to verify Whispr C extensions are built and can be imported.
"""

import os
import sys

# Add current directory to path to ensure we can import the C extensions directly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

extensions_found = True
try:
    # Try direct imports first
    try:
        from features_c import compute_frame_energy
        from framing_c import frame_signal_c
        from separation_c import separate_by_segmentation_c
        from vad_c import simple_energy_vad_c

        print("✅ Successfully imported C extensions directly")
    except ImportError:
        # Try package imports
        sys.path.insert(0, os.path.join(current_dir, "../.."))
        from whispr.c_ext import (
            compute_frame_energy,
            frame_signal_c,
            separate_by_segmentation_c,
            simple_energy_vad_c,
        )

        print("✅ Successfully imported C extensions through package")
except ImportError as e:
    extensions_found = False
    print(f"❌ C extensions not available. Error: {e}")
    sys.exit(1)

print("\n=== C Extension Status ===")
print(f"Extensions found: {extensions_found}")
print("All C extensions verified.")
