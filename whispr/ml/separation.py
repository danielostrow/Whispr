import logging
import os
from typing import Dict, List, Optional

import numpy as np

# Try to import C extensions
try:
    from ..c_ext import C_EXTENSIONS_AVAILABLE, separate_by_segmentation_c
except ImportError:
    C_EXTENSIONS_AVAILABLE = False

# Note: we rely solely on segment clustering for speaker separation


# Try to lazily import torch and asteroid; code will fallback if unavailable
try:
    import torch  # noqa: F401
    from asteroid.models import ConvTasNet  # noqa: F401

    _ASTEROID_AVAILABLE = True
except (ImportError, OSError):
    _ASTEROID_AVAILABLE = False


log = logging.getLogger(__name__)


def _separate_by_segmentation(
    signal: np.ndarray, sr: int, segments: List[tuple], labels: List[int]
) -> List[Dict]:
    """Separate speakers by concatenating their assigned segments.

    This is a simpler fallback method used when advanced separation (like
    Asteroid) is unavailable or fails. It does not handle overlapping speech.

    Args:
        signal: The input mono audio signal.
        sr: The sample rate of the signal.
        segments: List of (start_frame, end_frame) tuples from VAD.
        labels: List of integer speaker labels corresponding to each segment.

    Returns:
        A list of speaker dictionaries, each with their concatenated audio.
    """
    # Use C implementation if available
    if C_EXTENSIONS_AVAILABLE:
        return separate_by_segmentation_c(signal, sr, segments, labels)

    # Fall back to Python implementation
    # Collect frame index segments per speaker
    speakers: Dict[str, List[tuple]] = {}
    for (start_frame, end_frame), label in zip(segments, labels):
        spk_key = f"Speaker_{label + 1}"
        speakers.setdefault(spk_key, []).append((start_frame, end_frame))

    hop_len = int(sr * 0.01)  # 10 ms as per Config default
    frame_len = int(sr * 0.025)  # 25 ms

    out: List[Dict] = []
    for spk, segs in speakers.items():
        audio_slices = []
        time_segments = []
        for start_f, end_f in segs:
            start_smp = start_f * hop_len
            end_smp = end_f * hop_len + frame_len
            audio_slices.append(signal[start_smp:end_smp])

            # convert to seconds for metadata
            start_t = start_smp / sr
            end_t = end_smp / sr
            time_segments.append((start_t, end_t))

        if not audio_slices:
            continue

        concatenated = np.concatenate(audio_slices)
        out.append(
            {
                "id": spk,
                "signal": concatenated,
                "sr": sr,
                "segments": time_segments,
            }
        )

    return out


def _find_asteroid_model():
    """Find the Asteroid model file.

    Returns:
        Path to the model file or None if not found
    """
    # Paths to check for our custom model
    paths_to_check = [
        "models/whispr_asteroid_model.pth",
        os.path.join(
            os.path.dirname(__file__), "../../models/whispr_asteroid_model.pth"
        ),
        os.path.expanduser("~/.whispr/models/whispr_asteroid_model.pth"),
    ]

    for path in paths_to_check:
        if os.path.exists(path):
            log.info(f"Found custom Asteroid model at {path}")
            return path

    log.info("No custom Asteroid model found, will use default pretrained model")
    return None


def _asteroid_separate(
    signal: np.ndarray, sr: int, max_speakers: int = 2
) -> Optional[List[Dict]]:
    """Use Asteroid's Conv-TasNet to perform blind speech separation.

    This function attempts to use a pre-trained source separation model to
    isolate speakers. It requires `torch` and `asteroid` to be installed.
    The model is downloaded from HuggingFace Hub on first run.

    Args:
        signal: Mono 1-D waveform at `sr` Hz.
        sr: Sampling rate (must match the model, 16 kHz for most weights).
        max_speakers: Number of speakers the model will try to extract.

    Returns:
        A list of separated speaker tracks, or None if separation fails or
        dependencies are not met.
    """
    if not _ASTEROID_AVAILABLE:
        log.warning(
            "Asteroid not available. Skipping blind source separation. "
            "Install with: pip install asteroid"
        )
        return None

    if sr != 16_000:
        log.warning(
            f"Asteroid model requires 16kHz sample rate, but got {sr}. "
            "Skipping separation."
        )
        return None

    # Try to load our custom trained model first
    custom_model_path = _find_asteroid_model()

    # Load model (custom if available, otherwise pretrained)
    try:
        if custom_model_path:
            model = ConvTasNet(n_src=2)  # Initialize with same architecture
            model.load_state_dict(torch.load(custom_model_path, map_location="cpu"))
            log.info("Using custom trained Asteroid model")
        else:
            # Fall back to pretrained model
            model = ConvTasNet.from_pretrained(
                "mpariente/ConvTasNet_LibriMix_sepclean_16k"
            )
            log.info("Using default pretrained Asteroid model")
    except (RuntimeError, OSError) as e:
        log.error(f"Failed to load Asteroid model: {e}. Cannot perform separation.")
        return None

    # Asteroid expects float32 tensor (batch, time)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    wav_tensor = torch.tensor(signal, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        est_sources = model(wav_tensor)

    est_sources = est_sources.squeeze(0).cpu().numpy()

    tracks = []
    for idx in range(min(max_speakers, est_sources.shape[0])):
        src = est_sources[idx]
        tracks.append(
            {
                "id": f"Speaker_{idx + 1}",  # Changed from "Asteroid_Source_" to match segmentation naming
                "signal": src.astype(np.float32),
                "sr": sr,
                "segments": [(0.0, len(src) / sr)],
            }
        )

    log.info(f"Successfully separated {len(tracks)} sources using Asteroid.")
    return tracks


def separate(
    signal: np.ndarray, sr: int, segments: List[tuple], labels: List[int]
) -> List[Dict]:
    """Separate audio into per-speaker tracks.

    This function acts as a dispatcher for speaker separation. It first
    attempts to use a sophisticated blind source separation model (Asteroid)
    if the number of speakers is low (<= 2) and the necessary libraries are
    installed.

    If the conditions for Asteroid are not met or it fails, it will fall back
    to a simpler method of concatenating VAD segments based on clustering
    results.

    Args:
        signal: The input mono audio signal.
        sr: The sample rate of the signal.
        segments: List of (start_frame, end_frame) tuples from VAD.
        labels: List of integer speaker labels corresponding to each segment.

    Returns:
        A list of speaker dictionaries, each containing the separated signal.
    """
    num_speakers = len(set(labels))
    log.info(f"Attempting to separate {num_speakers} speakers.")

    # Use Asteroid only for simple 2-speaker mixtures; otherwise clustering-based
    if num_speakers > 0 and num_speakers <= 2:
        tracks = _asteroid_separate(signal, sr, max_speakers=num_speakers)
        if tracks is not None:
            return tracks
        else:
            log.warning("Asteroid separation failed, falling back to segmentation.")

    # Fallback / multi-speaker case
    log.info("Using segmentation-based separation.")
    return _separate_by_segmentation(signal, sr, segments, labels)
