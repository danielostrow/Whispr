# Machine Learning (ML)

This module contains the core machine learning logic for speaker diarization.

## Modules

- `vad.py`: Performs Voice Activity Detection (VAD) to find speech segments.
- `clustering.py`: Clusters speech segments into distinct speakers.
- `separation.py`: Separates audio sources to isolate individual speakers. It uses `asteroid` if available for superior performance with overlapping speech. 