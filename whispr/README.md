# Whispr

Whispr is a minimal prototype that detects, clusters and maps speakers from a **single audio file** (mono or stereo) and lets you play each speaker's audio inside an interactive web UI.

## Quick-start

```bash
python -m pip install -r requirements.txt

# Run analysis on your audio file (wav / mp3)
python -m whispr.pipeline path/to/meeting.wav

# Launch the Dash UI
python -m whispr.ui.app output/metadata.json
```

After the pipeline finishes you will find `output/` containing:

* `Speaker_1.wav`, `Speaker_2.wav`, … – isolated tracks
* `metadata.json` – speaker positions + segment timings

Open the UI, click a node to listen to that speaker only.

## Notes

* The current pipeline uses a very naïve energy-VAD, MFCC clustering and dummy localisation. It works best when speakers take turns and background noise is low.
* For overlap handling, install `torch` and `asteroid`; the pipeline will auto-use Asteroid's Conv-TasNet when available.
* Likewise, swap in a stronger VAD (WebRTC, pyannote) or clustering (HDBSCAN) without changing the outer API.
