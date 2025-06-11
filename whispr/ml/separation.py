from typing import List, Dict, Optional

# Note: we rely solely on segment clustering for speaker separation

import numpy as np

# Try to lazily import torch and asteroid; code will fallback if unavailable
try:
    import torch  # noqa: F401
    from asteroid.models import ConvTasNet  # noqa: F401

    _ASTEROID_AVAILABLE = True
except (ImportError, OSError):
    _ASTEROID_AVAILABLE = False


def _dummy_separate(signal: np.ndarray, sr: int, segments: List[tuple], labels: List[int]) -> List[Dict]:
    """Fallback segmentation-based separation (no overlap handling)."""
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
        out.append({
            "id": spk,
            "signal": concatenated,
            "sr": sr,
            "segments": time_segments,
        })

    return out


def _asteroid_separate(signal: np.ndarray, sr: int, max_speakers: int = 2) -> Optional[List[Dict]]:
    """Use Asteroid Conv-TasNet to perform blind speech separation.

    Parameters
    ----------
    signal : np.ndarray
        Mono 1-D waveform at `sr` Hz.
    sr : int
        Sampling rate (must match the model, 16 kHz for most pre-trained weights).
    max_speakers : int, optional
        Number of speakers the model will try to extract (default 2).

    Returns
    -------
    list[dict] or None
        Separated tracks or None if Asteroid/torch unavailable.
    """
    if not _ASTEROID_AVAILABLE:
        return None

    # Asteroid expects float32 tensor (batch, time)
    import torch
    from asteroid.models import ConvTasNet

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-trained model (2-speaker LibriMix). Works reasonably for generic speech.
    try:
        model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_LibriMix_sepclean_16k")
    except RuntimeError:
        # Fallback to local random-init if HF download fails
        return None

    model.to(device).eval()

    if sr != 16_000:
        # Ideally resample; for now, return None to fallback
        return None

    wav_tensor = torch.tensor(signal, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        est_sources = model(wav_tensor)  # (batch, n_src, time)

    est_sources = est_sources.squeeze(0).cpu().numpy()  # (n_src, time)

    tracks = []
    for idx in range(min(max_speakers, est_sources.shape[0])):
        src = est_sources[idx]
        tracks.append(
            {
                "id": f"Asteroid_Source_{idx + 1}",
                "signal": src.astype(np.float32),
                "sr": sr,
                "segments": [(0.0, len(src) / sr)],
            }
        )

    return tracks


def separate(signal: np.ndarray, sr: int, segments: List[tuple], labels: List[int]) -> List[Dict]:
    """Speaker-centric separation.

    The project's goal is to isolate *speakers*.

    If Asteroid (Conv-TasNet) is available, we attempt blind speech separation to
    handle overlapping voices.  Otherwise, we revert to concatenating VAD
    segments per clustered speaker.
    """

    # Use Asteroid only for simple 2-speaker mixtures; otherwise clustering-based
    if len(set(labels)) <= 2:
        tracks = _asteroid_separate(signal, sr, max_speakers=len(set(labels)))
        if tracks is not None:
            return tracks

    # Fallback / multi-speaker case
    return _dummy_separate(signal, sr, segments, labels) 