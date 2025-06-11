from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config
from .io.loader import load_audio
from .dsp.features import extract_features
from .ml.vad import simple_energy_vad
from .ml.clustering import cluster_speakers
from .ml.separation import separate
from .localization import assign_locations
from .io.writer import write_speaker_clips


def run_pipeline(audio_path: str, cfg: Optional[Config] = None) -> Path:
    """Run Whispr pipeline on `audio_path` and return metadata JSON path."""
    cfg = cfg or Config()

    # 1. Load audio
    mono, sr = load_audio(Path(audio_path), cfg)

    # 2. Feature extraction
    mfcc, energies = extract_features(mono, cfg)

    # 3. VAD
    segments = simple_energy_vad(energies, cfg)
    if not segments:
        raise RuntimeError("No speech detected.")

    # 4. Clustering
    labels = cluster_speakers(mfcc, segments)

    # 5. Assemble per-speaker signals (Demucs if available)
    speakers = separate(mono, sr, segments, labels)

    # 6. Localization (placeholder)
    assign_locations(speakers)

    # 7. Export clips + metadata
    meta_path = write_speaker_clips(speakers, cfg)
    return meta_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Whispr pipeline on audio file")
    parser.add_argument("audio", type=str, help="Path to audio (wav/mp3)")
    parser.add_argument("--sr", type=int, default=16_000, help="Target sampling rate")
    args = parser.parse_args()

    cfg = Config(sample_rate=args.sr)
    out_meta = run_pipeline(args.audio, cfg)
    print(f"Processing complete. Metadata saved to: {out_meta}") 