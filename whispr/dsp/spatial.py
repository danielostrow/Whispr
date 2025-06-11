from typing import Tuple

import numpy as np
import scipy.signal as spsig

from ..config import Config


def stereo_cues(
    left: np.ndarray, right: np.ndarray, cfg: Config
) -> Tuple[float, float]:
    """Estimate mean ITD (in seconds) and ILD (dB) between left and right channels.

    Uses cross-correlation for ITD and RMS ratio for ILD.
    """
    # ITD via GCC-PHAT peak
    cross = spsig.fftconvolve(left, right[::-1], mode="full")
    lag = np.argmax(np.abs(cross)) - (len(right) - 1)
    itd = lag / cfg.sample_rate

    # ILD
    rms_l = np.sqrt(np.mean(left**2) + 1e-9)
    rms_r = np.sqrt(np.mean(right**2) + 1e-9)
    ild = 20 * np.log10(rms_l / rms_r + 1e-9)
    return float(itd), float(ild)
