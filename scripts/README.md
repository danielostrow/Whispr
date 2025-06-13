# Whispr Training Scripts

This directory contains scripts for preparing data and training models for the Whispr speaker diarization and localization system.

## Overview

Whispr is a system designed to identify individual speakers from audio recordings and determine their spatial locations. The system uses:

1. **Asteroid** for source separation (isolating individual speakers)
2. **Clustering** for speaker diarization (identifying who spoke when)
3. **Spatial audio processing** for speaker localization (determining where speakers are located)

## Prerequisites

Install the required dependencies:

```bash
pip install torch torchaudio asteroid librosa soundfile numpy scikit-learn
```

## Data Preparation

The `prepare_training_data.py` script processes raw audio files and annotations into a format suitable for training:

```bash
python scripts/prepare_training_data.py --input-dir data/training --output-dir data
```

This will:
- Process multi-speaker audio files and their annotations
- Extract individual speaker segments
- Process spatial audio files with their location information
- Process iPhone recordings (even without metadata)
- Create metadata files for training

### Handling iPhone Recordings Without Metadata

The script includes special handling for iPhone recordings that don't have spatial metadata:

1. For **mono recordings**, it:
   - Uses energy-based voice activity detection to find speech segments
   - Groups segments into speakers based on timing
   - Assigns estimated spatial positions to each speaker

2. For **stereo recordings**, it:
   - Estimates left-right position based on channel energy differences
   - Maps these differences to spatial coordinates

No manual annotation is required - the system will automatically generate estimated metadata.

## Training Models

### 1. Train Asteroid Model for Source Separation

```bash
python scripts/train_asteroid.py --data-dir data/processed/multi_speaker --output-dir models --epochs 10
```

This trains a ConvTasNet model from Asteroid to separate speakers in mixed audio.

### 2. Train Speaker Localization Model

```bash
python scripts/train_localization.py --data-dir data/processed/spatial_audio --output-dir models --epochs 20
```

This trains a neural network to predict speaker locations from multi-channel audio features.

### Training with iPhone Recordings

To train using iPhone recordings:

```bash
python scripts/train_localization.py --data-dir data/processed/iphone_samples --output-dir models --epochs 20
```

The training script will automatically handle files with or without metadata.

## Using the Trained Models with Whispr

After training, the models will be saved in the `models` directory. To use them with the Whispr pipeline:

1. For source separation, modify `whispr/ml/separation.py` to load your trained Asteroid model:

```python
# In _asteroid_separate() function
model = ConvTasNet.from_pretrained("PATH_TO_YOUR_MODEL/whispr_asteroid_model.pth")
```

2. For speaker localization, update `whispr/localization.py` to use your trained localization model.

## Directory Structure

The expected directory structure for training data is:

```
data/
├── training/
│   ├── multi_speaker/          # Multi-speaker audio files
│   ├── single_speaker/         # Single-speaker audio files
│   ├── annotations/            # RTTM annotation files
│   ├── spatial_audio/          # Multi-channel audio files
│   ├── spatial_info/           # JSON files with speaker location information
│   └── iphone_samples/         # iPhone recordings (with or without metadata)
└── processed/                  # Created by prepare_training_data.py
    ├── multi_speaker/          # Processed multi-speaker data
    ├── spatial_audio/          # Processed spatial audio data
    └── iphone_samples/         # Processed iPhone recordings with estimated metadata
```

## Notes

- The scripts handle both audio files with and without metadata
- For iPhone recordings without metadata, the system estimates speaker positions
- For best results with iPhone recordings, use stereo mode if available 