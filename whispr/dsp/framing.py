import numpy as np

try:
    from ..c_ext import C_EXTENSIONS_AVAILABLE, frame_signal_c
except ImportError:
    C_EXTENSIONS_AVAILABLE = False


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Frame a 1-D signal into overlapping frames using stride tricks.

    Parameters
    ----------
    signal : np.ndarray
        1-D array.
    frame_length : int
        Number of samples per frame.
    hop_length : int
        Number of samples between successive frames.

    Returns
    -------
    frames : np.ndarray [shape=(n_frames, frame_length)]
        2-D array of frames.
    """
    # Use C implementation if available for better performance
    if C_EXTENSIONS_AVAILABLE:
        return frame_signal_c(signal, frame_length, hop_length)

    # Fall back to numpy implementation
    num_frames = 1 + int((len(signal) - frame_length) / hop_length)
    shape = (num_frames, frame_length)
    strides = (hop_length * signal.strides[0], signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        signal, shape=shape, strides=strides
    ).copy()
    return frames
