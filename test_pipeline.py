#!/usr/bin/env python3
"""
Test script for the Whispr pipeline with our model integration changes.

This script downloads a sample audio file and runs it through the Whispr pipeline.
"""

import os
import argparse
import subprocess
from pathlib import Path

import requests


def download_sample_audio(output_dir):
    """Download a sample audio file for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for a sample multi-speaker audio file
    url = "https://github.com/joonson/voxconverse/raw/master/samples/sample1.wav"
    output_path = os.path.join(output_dir, "sample1.wav")
    
    if os.path.exists(output_path):
        print(f"Sample file already exists at {output_path}")
        return output_path
    
    print(f"Downloading sample audio from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded sample audio to {output_path}")
    return output_path


def run_pipeline(audio_path, output_dir="output", model_dir="models"):
    """Run the Whispr pipeline on the sample audio."""
    from whispr.config import Config
    from whispr.pipeline import run_pipeline
    
    print(f"Processing audio file: {audio_path}")
    
    # Create configuration
    cfg = Config(
        sample_rate=16000,
        output_dir=Path(output_dir),
        model_dir=Path(model_dir)
    )
    
    # Run the pipeline
    meta_path = run_pipeline(Path(audio_path), cfg)
    
    print(f"Processing complete. Metadata saved to: {meta_path}")
    return meta_path


def run_ui(metadata_path):
    """Launch the UI to visualize the results."""
    print(f"Launching UI with metadata: {metadata_path}")
    subprocess.run(["python", "-m", "whispr.ui.app", str(metadata_path)])


def main():
    parser = argparse.ArgumentParser(description="Test the Whispr pipeline")
    parser.add_argument("--audio", type=str, help="Path to input audio file (if not provided, a sample will be downloaded)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing trained models")
    parser.add_argument("--skip-ui", action="store_true", help="Skip launching the UI")
    args = parser.parse_args()
    
    # Get audio file (download sample if not provided)
    audio_path = args.audio
    if not audio_path:
        audio_path = download_sample_audio("samples")
    
    # Run the pipeline
    metadata_path = run_pipeline(audio_path, args.output_dir, args.model_dir)
    
    # Launch UI if not skipped
    if not args.skip_ui:
        run_ui(metadata_path)


if __name__ == "__main__":
    main() 