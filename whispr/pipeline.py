from pathlib import Path
from typing import Optional

from .config import Config
from .dsp.features import extract_features
from .io.loader import load_audio
from .io.writer import write_speaker_clips
from .localization import assign_locations
from .ml.clustering import cluster_speakers
from .ml.separation import separate
from .ml.vad import simple_energy_vad


def run_pipeline(audio_path: Path, cfg: Config) -> Path:
    """Run Whispr pipeline on a single audio file.

    This function orchestrates the entire speaker diarization process:
    1. Loads audio
    2. Extracts features (MFCCs, energy)
    3. Performs Voice Activity Detection (VAD)
    4. Clusters speech segments into speakers
    5. Separates audio for each speaker
    6. Assigns spatial locations (placeholder)
    7. Writes speaker clips and metadata to disk.

    Args:
        audio_path: Path to the input audio file.
        cfg: Configuration object.

    Returns:
        Path to the output metadata JSON file.

    Raises:
        RuntimeError: If no speech is detected in the audio.
    """
    # 1. Load audio
    mono, sr = load_audio(audio_path, cfg)

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


def main():
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run Whispr pipeline on an audio file.")
    parser.add_argument("audio", type=Path, help="Path to input audio file (wav, mp3, etc.)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Directory to save results."
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Target sample rate for processing."
    )
    args = parser.parse_args()

    cfg = Config(
        sample_rate=args.sample_rate,
        output_dir=args.output_dir,
    )
    out_meta = run_pipeline(args.audio, cfg)
    print(f"✅ Processing complete. Metadata saved to: {out_meta}")


if __name__ == "__main__":
    import argparse

    main()
