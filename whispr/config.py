from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    sample_rate: int = 16_000
    frame_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    vad_energy_threshold: float = 0.01  # relative energy
    min_speech_duration_s: float = 0.3  # merge small segments
    output_dir: Path = Path("output") 