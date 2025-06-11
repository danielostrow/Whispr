from typing import Tuple

import librosa
import numpy as np

from ..config import Config
from .framing import frame_signal

try:
    from ..c_ext import compute_frame_energy, C_EXTENSIONS_AVAILABLE
except ImportError:
    C_EXTENSIONS_AVAILABLE = False


def extract_features(signal: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MFCC + energy features for clustering.

    Returns
    -------
    mfcc : np.ndarray [shape=(n_frames, n_coeffs)]
    energies : np.ndarray [shape=(n_frames,)]
    """
    frame_len = int(cfg.sample_rate * cfg.frame_length_ms / 1000)
    hop_len = int(cfg.sample_rate * cfg.hop_length_ms / 1000)

    frames = frame_signal(signal, frame_len, hop_len)
    
    # Use C implementation for energy computation if available
    if C_EXTENSIONS_AVAILABLE:
        energies = compute_frame_energy(frames)
    else:
        # Fall back to numpy implementation
        windowed = frames * np.hanning(frame_len)
        energies = np.sum(windowed**2, axis=1)

    # Compute MFCCs on original continuous signal for convenience
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=cfg.sample_rate,
        n_mfcc=20,
        n_fft=frame_len * 2,
        hop_length=hop_len,
    ).T  # (n_frames, 20)

    # Align energy length to mfcc length in case of off-by-one
    min_len = min(len(energies), len(mfcc))
    return mfcc[:min_len], energies[:min_len]
