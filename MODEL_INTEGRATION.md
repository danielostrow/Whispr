# Model Integration in Whispr

This document explains the changes made to integrate trained models into the Whispr speaker diarization and localization system.

## Overview of Changes

1. **Enhanced 3D Spatial Localization**
   - Implemented sophisticated spatial localization in 3D space
   - Added advanced feature extraction for multi-channel audio
   - Incorporated time delay of arrival (TDOA) estimation
   - Added phase difference analysis across frequency bands
   - Implemented room dimension estimation based on audio characteristics
   - Added microphone array configuration detection

2. **Localization Module**
   - Replaced the placeholder localization function with a proper implementation
   - Added a `SpatialLocalizationModel` class with deeper neural network architecture
   - Added batch normalization for better training stability
   - Implemented triangulation for multi-channel recordings
   - Added fallback to circular speaker positioning when model is unavailable

3. **Separation Module**
   - Updated the Asteroid separation function to use our custom trained model
   - Added a function to find the model file in various locations
   - Improved speaker ID naming for consistency

4. **Training Improvements**
   - Added data augmentation for more robust training
   - Implemented validation split and early stopping
   - Added learning rate scheduling
   - Enhanced feature extraction with more sophisticated audio analysis
   - Added speaker position confidence estimation

5. **Pipeline Module**
   - Added logging for better visibility into the pipeline steps
   - Updated the pipeline to use the model directory from configuration
   - Added command-line argument for specifying the model directory

6. **UI Updates**
   - Enhanced the UI to display 3D speaker locations
   - Added a room outline visualization
   - Added detailed speaker information display
   - Improved the styling with a new CSS file

7. **iPhone Recording Processing**
   - Enhanced processing of iPhone recordings to extract spatial information
   - Added detection of stereo vs. pseudo-stereo recordings
   - Implemented estimation of room dimensions based on audio length
   - Added speaker position confidence metrics

## Directory Structure

```
models/                           # Directory for trained models
├── whispr_asteroid_model.pth     # Trained ConvTasNet model for speaker separation
├── whispr_localization_model.pth # Trained model for speaker localization
├── feature_params.json           # Parameters for feature extraction
└── README.md                     # Documentation for the models

scripts/                          # Training scripts
├── prepare_training_data.py      # Script to prepare data for training
├── train_asteroid.py             # Script to train the Asteroid model
├── train_localization.py         # Script to train the localization model
└── README.md                     # Documentation for the scripts
```

## How to Use Trained Models

1. **Place your trained models in the `models/` directory**:
   - `whispr_asteroid_model.pth` - Your trained Asteroid model
   - `whispr_localization_model.pth` - Your trained localization model
   - `feature_params.json` - Parameters used during training

2. **Run the pipeline**:
   ```bash
   python -m whispr.pipeline your_audio_file.wav --model-dir models
   ```

3. **Or use the test script**:
   ```bash
   python test_pipeline.py --audio your_audio_file.wav
   ```

## 3D Speaker Localization

The enhanced localization system uses multiple techniques to accurately position speakers in 3D space:

1. **For multi-channel recordings**:
   - Time delay of arrival (TDOA) estimation between channels
   - Phase difference analysis across frequency bands
   - Energy distribution across channels
   - Triangulation for precise 3D positioning

2. **For mono recordings**:
   - Voice activity detection to identify speech segments
   - Speaker clustering based on temporal patterns
   - Intelligent distribution of speakers in the room space

3. **Room estimation**:
   - Automatic estimation of room dimensions based on audio characteristics
   - Detection of microphone array configuration
   - Adaptation to different recording environments

## Fallback Mechanisms

If trained models are not available, the system will fall back to algorithmic methods:

1. **Localization**: 
   - For multi-channel audio: Uses triangulation based on channel differences
   - For mono audio: Distributes speakers in a circle around the room

2. **Separation**: 
   - First tries to use the default pretrained Asteroid model
   - If that fails, falls back to simple segmentation-based separation

## Next Steps

1. **Train the models** using the scripts in the `scripts/` directory
2. **Evaluate the models** on various audio recordings
3. **Fine-tune the models** for better performance with iPhone recordings
4. **Collect more spatial audio data** to improve localization accuracy 