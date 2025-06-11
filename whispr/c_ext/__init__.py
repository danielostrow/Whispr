"""C extensions for Whispr optimized functions."""

try:
    from .framing_c import frame_signal_c
    from .vad_c import simple_energy_vad_c
    from .features_c import compute_frame_energy
    from .separation_c import separate_by_segmentation_c
    
    # Flag to indicate if C extensions are available
    C_EXTENSIONS_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn(
        "Whispr C extensions not available. Using slower Python implementations. "
        "Run 'pip install -e whispr/c_ext' to build and install the extensions."
    )
    C_EXTENSIONS_AVAILABLE = False 