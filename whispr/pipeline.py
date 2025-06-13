import logging
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

log = logging.getLogger(__name__)


def run_pipeline(audio_path: Path, cfg: Config) -> Path:
    """Run Whispr pipeline on a single audio file.

    This function orchestrates the entire speaker diarization process:
    1. Loads audio
    2. Extracts features (MFCCs, energy)
    3. Performs Voice Activity Detection (VAD)
    4. Clusters speech segments into speakers
    5. Separates audio for each speaker
    6. Assigns spatial locations using trained model
    7. Writes speaker clips and metadata to disk.

    Args:
        audio_path: Path to the input audio file.
        cfg: Configuration object.

    Returns:
        Path to the output metadata JSON file.

    Raises:
        RuntimeError: If no speech is detected in the audio.
    """
    log.info(f"Processing audio file: {audio_path}")

    # 1. Load audio
    mono, sr = load_audio(audio_path, cfg)
    log.info(f"Loaded audio: {len(mono)/sr:.2f}s at {sr}Hz")

    # 2. Feature extraction
    mfcc, energies = extract_features(mono, cfg)
    log.info(f"Extracted features: {mfcc.shape[0]} frames")

    # 3. VAD
    segments = simple_energy_vad(energies, cfg)
    if not segments:
        raise RuntimeError("No speech detected.")
    log.info(f"Detected {len(segments)} speech segments")

    # 4. Clustering
    labels = cluster_speakers(mfcc, segments)
    num_speakers = len(set(labels))
    log.info(f"Identified {num_speakers} speakers")

    # 5. Assemble per-speaker signals (using trained Asteroid model if available)
    speakers = separate(mono, sr, segments, labels)
    log.info(f"Separated {len(speakers)} speaker tracks")

    # 6. Localization (using trained model if available)
    assign_locations(speakers)
    log.info("Assigned spatial locations to speakers")

    # 7. Export clips + metadata
    meta_path = write_speaker_clips(speakers, cfg)
    log.info(f"Wrote speaker clips and metadata to {meta_path}")
    return meta_path


def main():
    """Parse CLI arguments and run the pipeline."""
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run Whispr pipeline on an audio file."
    )
    parser.add_argument(
        "audio", type=Path, help="Path to input audio file (wav, mp3, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save results.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for processing.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained models.",
    )
    args = parser.parse_args()

    cfg = Config(
        sample_rate=args.sample_rate,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
    out_meta = run_pipeline(args.audio, cfg)
    print(f"✅ Processing complete. Metadata saved to: {out_meta}")


if __name__ == "__main__":
    main()
