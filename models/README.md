# Whispr Trained Models

This directory contains trained models for the Whispr speaker diarization and localization system.

## Expected Models

1. **whispr_asteroid_model.pth** - ConvTasNet model for speaker separation
2. **whispr_localization_model.pth** - Neural network model for speaker localization
3. **feature_params.json** - Parameters for feature extraction used during localization

## Training

These models are trained using the scripts in the `scripts/` directory:

- `train_asteroid.py` - For training the speaker separation model
- `train_localization.py` - For training the speaker localization model

## Usage

The Whispr pipeline will automatically look for these models in this directory.
If the models are not found, Whispr will fall back to default implementations:

- For separation: Use pretrained Asteroid model or simple segmentation-based separation
- For localization: Use simple spatial positioning along the x-axis

## Model Format

- The speaker separation model is a PyTorch ConvTasNet model from Asteroid
- The localization model is a simple PyTorch neural network with 3D output (x, y, z) 