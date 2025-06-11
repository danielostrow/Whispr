from typing import List, Tuple

import numpy as np

from ..config import Config

try:
    from ..c_ext import simple_energy_vad_c, C_EXTENSIONS_AVAILABLE
except ImportError:
    C_EXTENSIONS_AVAILABLE = False


def simple_energy_vad(energies: np.ndarray, cfg: Config) -> List[Tuple[int, int]]:
    """Return index ranges (frame_start, frame_end) where energy exceeds threshold.

    The threshold is `cfg.vad_energy_threshold` times the median energy.
    """
    # Use C implementation if available
    if C_EXTENSIONS_AVAILABLE:
        # Calculate min_frames from config
        min_frames = int((cfg.min_speech_duration_s * 1000) / cfg.hop_length_ms)
        return simple_energy_vad_c(energies, cfg.vad_energy_threshold, min_frames)
    
    # Fall back to Python implementation
    med = np.median(energies)
    thresh = med * (1.0 + cfg.vad_energy_threshold)

    voiced = energies > thresh

    segments = []
    start = None
    for idx, is_voiced in enumerate(voiced):
        if is_voiced and start is None:
            start = idx
        elif not is_voiced and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(voiced) - 1))

    # Merge or discard tiny segments
    min_frames = int((cfg.min_speech_duration_s * 1000) / cfg.hop_length_ms)
    refined = []
    for beg, end in segments:
        if (end - beg + 1) >= min_frames:
            refined.append((beg, end))
    return refined
