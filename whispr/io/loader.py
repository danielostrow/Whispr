from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from librosa import resample, to_mono

from ..config import Config


def load_audio(path: Path, cfg: Config) -> Tuple[np.ndarray, int]:
    """Load audio file, convert to mono, resample, and normalize to -1..1.

    Returns
    -------
    signal : np.ndarray [shape=(n_samples,)]
        Normalised mono signal.
    sr : int
        Sample rate after resampling.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    signal, sr = sf.read(path, always_2d=False)

    # If stereo, convert to mono for processing; keep copy for spatial cues later.
    if signal.ndim == 2:
        mono = to_mono(signal.T)  # librosa expects shape=(channels, n)
    else:
        mono = signal.astype(np.float32)

    if sr != cfg.sample_rate:
        mono = resample(mono, orig_sr=sr, target_sr=cfg.sample_rate)
        sr = cfg.sample_rate

    # Amplitude normalization
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = mono / peak

    return mono.astype(np.float32), sr
