"""C extensions for Whispr optimized functions."""

import os
import sys

# Add the current directory to the path to ensure extensions can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .features_c import compute_frame_energy
    from .framing_c import frame_signal_c
    from .separation_c import separate_by_segmentation_c
    from .vad_c import simple_energy_vad_c

    # Flag to indicate if C extensions are available
    C_EXTENSIONS_AVAILABLE = True
except ImportError:
    import warnings

    warnings.warn(
        "Whispr C extensions not available. Using slower Python implementations. "
        "Run 'pip install -e whispr/c_ext' to build and install the extensions."
    )
    C_EXTENSIONS_AVAILABLE = False
