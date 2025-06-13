#!/usr/bin/env python3
"""
Train a model for speaker localization using spatial audio data.

This script sets up and trains a neural network model to predict
speaker locations from multi-channel audio features.
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpatialFeatureExtractor:
    """Extract spatial features from multi-channel audio."""
    
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_features(self, audio, sr=None):
        """Extract spatial features from audio.
        
        Args:
            audio: Audio signal (mono or multi-channel)
            sr: Sample rate (if None, use self.sample_rate)
            
        Returns:
            Feature vector for localization
        """
        sr = sr or self.sample_rate
        
        # Resample if necessary
        if sr != self.sample_rate:
            if audio.ndim == 1:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            else:
                # Resample each channel
                resampled = []
                for ch in range(audio.shape[0]):
                    resampled.append(librosa.resample(audio[ch], orig_sr=sr, target_sr=self.sample_rate))
                audio = np.array(resampled)
        
        if audio.ndim == 1:
            # Mono audio
            return self._extract_mono_features(audio)
        else:
            # Multi-channel audio
            return self._extract_multichannel_features(audio)
    
    def _extract_mono_features(self, audio):
        """Extract features from mono audio."""
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude)
        spectral_contrast = librosa.feature.spectral_contrast(S=magnitude)
        
        # Extract temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        # Extract rhythm features
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        
        # Flatten and concatenate features
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.mean(spectral_contrast, axis=1),
            np.mean(zero_crossing_rate),
            np.array([tempo]),
        ])
        
        return features
    
    def _extract_multichannel_features(self, audio):
        """Extract spatial features from multi-channel audio."""
        num_channels = audio.shape[0]
        
        # Extract mono features for each channel
        channel_features = []
        for ch in range(num_channels):
            channel_features.append(self._extract_mono_features(audio[ch]))
        
        # Compute inter-channel features for spatial localization
        spatial_features = []
        
        # Time delay of arrival (TDOA) estimation between channels
        for i in range(num_channels):
            for j in range(i+1, num_channels):
                # Cross-correlation to estimate delay
                corr = np.correlate(audio[i], audio[j], mode='full')
                max_idx = np.argmax(corr)
                delay = max_idx - len(audio[i]) + 1  # Convert to sample delay
                delay_ms = delay * 1000 / self.sample_rate  # Convert to milliseconds
                spatial_features.append(delay_ms)
                
                # Phase difference in different frequency bands
                stft_i = librosa.stft(audio[i], n_fft=self.n_fft, hop_length=self.hop_length)
                stft_j = librosa.stft(audio[j], n_fft=self.n_fft, hop_length=self.hop_length)
                phase_i = np.angle(stft_i)
                phase_j = np.angle(stft_j)
                phase_diff = phase_i - phase_j
                
                # Average phase difference in low, mid, and high frequency bands
                n_bands = 3
                band_size = stft_i.shape[0] // n_bands
                for band in range(n_bands):
                    start_bin = band * band_size
                    end_bin = (band + 1) * band_size if band < n_bands - 1 else stft_i.shape[0]
                    band_phase_diff = np.mean(np.abs(phase_diff[start_bin:end_bin]))
                    spatial_features.append(band_phase_diff)
                
                # Energy ratio between channels
                energy_i = np.sum(np.abs(audio[i])**2)
                energy_j = np.sum(np.abs(audio[j])**2)
                energy_ratio = energy_i / (energy_j + 1e-8)  # Avoid division by zero
                spatial_features.append(energy_ratio)
        
        # Combine all features
        features = np.concatenate([
            np.concatenate(channel_features),
            np.array(spatial_features)
        ])
        
        return features

class SpatialLocalizationDataset(Dataset):
    """Dataset for training speaker localization models."""
    
    def __init__(self, data_dir, feature_extractor, augment=True):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing processed spatial audio data
            feature_extractor: Instance of SpatialFeatureExtractor
            augment: Whether to use data augmentation
        """
        self.data_dir = Path(data_dir)
        self.feature_extractor = feature_extractor
        self.augment = augment
        
        # Find all processed metadata files
        self.metadata_files = list(self.data_dir.glob("*_processed.json"))
        logger.info(f"Found {len(self.metadata_files)} metadata files")
        
        # Find all audio files (for cases without metadata)
        self.audio_files = list(self.data_dir.glob("*.wav"))
        logger.info(f"Found {len(self.audio_files)} audio files")
        
        # Load all metadata
        self.examples = []
        for meta_file in self.metadata_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            audio_file = self.data_dir / metadata["filename"]
            if not audio_file.exists():
                logger.warning(f"Audio file {audio_file} not found, skipping")
                continue
            
            # Add example with metadata
            self.examples.append({
                "metadata": metadata,
                "audio_file": audio_file,
                "has_metadata": True
            })
        
        # Add audio files without metadata
        for audio_file in self.audio_files:
            # Skip files that already have metadata
            if any(ex["audio_file"] == audio_file for ex in self.examples):
                continue
                
            # Add example without metadata
            self.examples.append({
                "metadata": None,
                "audio_file": audio_file,
                "has_metadata": False
            })
        
        logger.info(f"Loaded {len(self.examples)} examples total")
        logger.info(f"- {sum(1 for ex in self.examples if ex['has_metadata'])} with metadata")
        logger.info(f"- {sum(1 for ex in self.examples if not ex['has_metadata'])} without metadata")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        audio_file = example["audio_file"]
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=None, mono=False)
        
        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment_audio(audio)
        
        # Extract features
        features = self.feature_extractor.extract_features(audio, sr)
        
        # Handle case with metadata
        if example["has_metadata"]:
            metadata = example["metadata"]
            # Get speaker locations
            speaker_locations = []
            for speaker in metadata["speakers"]:
                if "position" in speaker:
                    pos = speaker["position"]
                    # Normalize coordinates to [0, 1] range based on room dimensions
                    room_dims = metadata.get("room_dimensions", {"width": 5.0, "length": 5.0, "height": 3.0})
                    x_norm = pos["x"] / room_dims["width"]
                    y_norm = pos["y"] / room_dims["length"]
                    z_norm = pos["z"] / room_dims["height"]
                    speaker_locations.append([x_norm, y_norm, z_norm])
            
            # If no speaker locations in metadata, estimate from audio
            if not speaker_locations:
                speaker_locations = [self._estimate_location_from_audio(audio)]
        else:
            # No metadata, estimate location from audio
            speaker_locations = [self._estimate_location_from_audio(audio)]
        
        # Use the first speaker's location for simplicity
        location = speaker_locations[0]
        
        # Convert to torch tensors
        features = torch.tensor(features, dtype=torch.float32)
        location = torch.tensor(location, dtype=torch.float32)
        
        return {"features": features, "location": location}
    
    def _augment_audio(self, audio):
        """Apply data augmentation to audio.
        
        Args:
            audio: Audio array (mono or multi-channel)
            
        Returns:
            Augmented audio
        """
        # Skip augmentation with 20% probability
        if np.random.rand() < 0.2:
            return audio
        
        # Apply random gain
        gain = np.random.uniform(0.8, 1.2)
        audio = audio * gain
        
        # Add small amount of noise
        if np.random.rand() < 0.5:
            noise_level = np.random.uniform(0.001, 0.005)
            if audio.ndim == 1:
                noise = np.random.randn(len(audio)) * noise_level * np.max(np.abs(audio))
                audio = audio + noise
            else:
                for ch in range(audio.shape[0]):
                    noise = np.random.randn(audio.shape[1]) * noise_level * np.max(np.abs(audio[ch]))
                    audio[ch] = audio[ch] + noise
        
        # Apply random time shift
        if np.random.rand() < 0.5:
            shift_amount = int(np.random.uniform(0, 0.1) * (audio.shape[-1]))
            if audio.ndim == 1:
                audio = np.roll(audio, shift_amount)
            else:
                audio = np.roll(audio, shift_amount, axis=1)
        
        return audio
    
    def _estimate_location_from_audio(self, audio):
        """Estimate speaker location from audio features when no metadata is available.
        
        This is a fallback method that uses channel differences to roughly estimate position.
        In a real implementation, you would use a more sophisticated algorithm.
        
        Args:
            audio: Multi-channel audio array
            
        Returns:
            Estimated [x, y, z] normalized position
        """
        # Default to center if mono audio
        if audio.ndim == 1:
            return [0.5, 0.5, 0.5]
        
        num_channels = audio.shape[0]
        
        # With stereo or more channels, estimate position based on channel energy
        channel_energy = np.array([np.sum(np.abs(audio[ch])) for ch in range(num_channels)])
        
        if num_channels == 2:
            # For stereo, estimate left-right position based on channel balance
            left_right_balance = channel_energy[0] / (np.sum(channel_energy) + 1e-8)
            # Convert to 0-1 range (0 = left, 1 = right)
            x_pos = 1.0 - left_right_balance
            return [x_pos, 0.5, 0.5]  # Assume centered in y and z
        
        elif num_channels >= 4:
            # For 4+ channels, assume a standard layout (front-left, front-right, rear-left, rear-right)
            # and estimate 2D position
            
            # Normalize energies
            channel_energy = channel_energy / (np.sum(channel_energy) + 1e-8)
            
            # Estimate x position (left-right)
            left_energy = channel_energy[0] + channel_energy[2]  # front-left + rear-left
            right_energy = channel_energy[1] + channel_energy[3]  # front-right + rear-right
            x_pos = right_energy / (left_energy + right_energy + 1e-8)
            
            # Estimate y position (front-back)
            front_energy = channel_energy[0] + channel_energy[1]  # front-left + front-right
            rear_energy = channel_energy[2] + channel_energy[3]  # rear-left + rear-right
            y_pos = rear_energy / (front_energy + rear_energy + 1e-8)
            
            return [x_pos, y_pos, 0.5]  # Assume centered in z
        
        else:
            # For other channel configurations, use a simple heuristic
            # based on relative channel energies
            max_channel = np.argmax(channel_energy)
            positions = []
            
            # Distribute positions evenly in a circle
            for ch in range(num_channels):
                angle = 2 * np.pi * ch / num_channels
                x = 0.5 + 0.5 * np.cos(angle)
                y = 0.5 + 0.5 * np.sin(angle)
                positions.append([x, y])
            
            # Return position of the loudest channel
            return positions[max_channel] + [0.5]  # x, y from position + z=0.5

class SpatialLocalizationModel(nn.Module):
    """Neural network model for speaker localization in 3D space."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3)  # Output: (x, y, z) coordinates
        )
    
    def forward(self, x):
        return self.model(x)

def train_localization_model(data_dir, output_dir, epochs=30, batch_size=16, learning_rate=0.001, val_split=0.2):
    """Train a model for speaker localization in 3D space.
    
    Args:
        data_dir: Directory containing processed spatial audio data
        output_dir: Directory to save trained models
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        val_split: Fraction of data to use for validation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create feature extractor
    feature_extractor = SpatialFeatureExtractor()
    
    # Create dataset
    dataset = SpatialLocalizationDataset(data_dir, feature_extractor, augment=True)
    
    # If dataset is empty, exit
    if len(dataset) == 0:
        logger.error("No valid examples found for training")
        return None
    
    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    ) if val_size > 0 else None
    
    # Get input dimension from the first example
    first_example = dataset[0]
    input_dim = first_example["features"].shape[0]
    
    # Create model
    model = SpatialLocalizationModel(input_dim)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train the model
    logger.info(f"Training model for {epochs} epochs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            features = batch["features"].to(device)
            locations = batch["location"].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_func(outputs, locations)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * features.size(0)
        
        avg_train_loss = total_loss / len(train_dataset)
        
        # Validation phase
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].to(device)
                    locations = batch["location"].to(device)
                    
                    outputs = model(features)
                    loss = loss_func(outputs, locations)
                    val_loss += loss.item() * features.size(0)
            
            avg_val_loss = val_loss / len(val_dataset)
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = output_dir / f"whispr_localization_model_best.pth"
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model with validation loss: {avg_val_loss:.4f}")
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    
    # Save final model
    final_model_path = output_dir / "whispr_localization_model.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # If we have a best model, copy it to the standard name
    if best_model_path is not None and best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, final_model_path)
        logger.info(f"Copied best model (val loss: {best_val_loss:.4f}) to {final_model_path}")
    
    logger.info(f"Model saved to {final_model_path}")
    
    # Save feature extractor parameters
    feature_params = {
        "sample_rate": feature_extractor.sample_rate,
        "n_fft": feature_extractor.n_fft,
        "hop_length": feature_extractor.hop_length
    }
    
    with open(output_dir / "feature_params.json", 'w') as f:
        json.dump(feature_params, f, indent=2)
    
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description="Train speaker localization model for Whispr")
    parser.add_argument("--data-dir", type=str, default="data/processed/spatial_audio",
                        help="Directory containing processed spatial audio data")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    args = parser.parse_args()
    
    train_localization_model(
        args.data_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split
    )
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 