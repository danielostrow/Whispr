import json
from pathlib import Path
from typing import List

import soundfile as sf

from ..config import Config


def write_speaker_clips(speakers: List[dict], cfg: Config) -> Path:
    """Write wav files for each speaker and produce metadata JSON.

    Args:
        speakers: Each dict has keys: id, signal (np.ndarray), sr,
            segments (list of (start, end)).
        cfg: The configuration object.

    Returns:
        Path to metadata JSON file.
    """
    out_root = cfg.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    meta = {"speakers": []}

    for spk in speakers:
        spk_id = spk["id"]
        wav_path = out_root / f"{spk_id}.wav"
        sf.write(wav_path, spk["signal"], spk["sr"])

        meta["speakers"].append(
            {
                "id": spk_id,
                "location": spk.get("location", [0.0, 0.0]),
                "audio_file": wav_path.name,
                "segments": [
                    {"start": float(s), "end": float(e)}
                    for s, e in spk.get("segments", [])
                ],
            }
        )

    meta_path = out_root / "metadata.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return meta_path
