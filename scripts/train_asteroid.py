#!/usr/bin/env python3
"""
Train an Asteroid model for speaker separation using our prepared data.

This script sets up and trains a ConvTasNet model from Asteroid for
speaker separation, which is used in the Whispr pipeline.
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

from asteroid.models import ConvTasNet
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WhisprDataset(Dataset):
    """Dataset for training Asteroid models with our processed data."""
    
    def __init__(self, data_dir, sample_rate=16000, segment_length=3.0):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing processed multi-speaker data
            sample_rate: Target sample rate
            segment_length: Length of audio segments in seconds
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        
        # Find all metadata files
        self.metadata_files = list(self.data_dir.glob("*_metadata.json"))
        logger.info(f"Found {len(self.metadata_files)} metadata files")
        
        # Load all metadata
        self.examples = []
        for meta_file in self.metadata_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Only use examples with at least 2 speakers
            if len(metadata["speakers"]) >= 2:
                self.examples.append({
                    "metadata": metadata,
                    "meta_file": meta_file,
                    "directory": meta_file.parent
                })
        
        logger.info(f"Loaded {len(self.examples)} valid examples with 2+ speakers")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        metadata = example["metadata"]
        directory = example["directory"]
        
        # Load audio for each speaker
        speaker_audios = []
        for speaker in metadata["speakers"]:
            audio_file = directory / speaker["audio_file"]
            audio, sr = sf.read(audio_file)
            
            # Resample if necessary
            if sr != self.sample_rate:
                # In a real implementation, you'd resample here
                pass
            
            speaker_audios.append(audio)
        
        # Ensure we have exactly 2 sources (for simplicity in this example)
        # In a real implementation, you might handle variable numbers of speakers
        if len(speaker_audios) > 2:
            # Use only the first two speakers
            speaker_audios = speaker_audios[:2]
        elif len(speaker_audios) < 2:
            # Duplicate the speaker to create a second source
            speaker_audios.append(speaker_audios[0])
        
        # Ensure both sources have the same length
        min_length = min(len(audio) for audio in speaker_audios)
        speaker_audios = [audio[:min_length] for audio in speaker_audios]
        
        # If too short, pad with zeros
        if min_length < self.segment_samples:
            speaker_audios = [
                np.pad(audio, (0, self.segment_samples - len(audio)))
                for audio in speaker_audios
            ]
        
        # If too long, randomly select a segment
        elif min_length > self.segment_samples:
            start = np.random.randint(0, min_length - self.segment_samples)
            speaker_audios = [
                audio[start:start + self.segment_samples]
                for audio in speaker_audios
            ]
        
        # Mix the sources to create the input mixture
        mixture = sum(speaker_audios)
        
        # Normalize
        mixture = mixture / np.max(np.abs(mixture))
        speaker_audios = [audio / np.max(np.abs(audio)) for audio in speaker_audios]
        
        # Convert to torch tensors
        mixture = torch.tensor(mixture, dtype=torch.float32)
        sources = torch.tensor(np.array(speaker_audios), dtype=torch.float32)
        
        return {"mixture": mixture, "sources": sources}

def train_asteroid_model(data_dir, output_dir, epochs=10, batch_size=8):
    """Train an Asteroid ConvTasNet model for speaker separation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = WhisprDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # Create model
    model = ConvTasNet(n_src=2)
    
    # Define optimizer and loss function
    optimizer = make_optimizer(model.parameters(), lr=1e-3)
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    
    # Create system for training
    system = System(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        train_loader=dataloader,
        val_loader=None,  # In a real implementation, you'd have a validation set
    )
    
    # Train the model
    logger.info(f"Training model for {epochs} epochs")
    system.fit(epochs=epochs)
    
    # Save the model
    model_path = output_dir / "whispr_asteroid_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Train Asteroid model for Whispr")
    parser.add_argument("--data-dir", type=str, default="data/processed/multi_speaker",
                        help="Directory containing processed multi-speaker data")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size")
    args = parser.parse_args()
    
    train_asteroid_model(
        args.data_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 