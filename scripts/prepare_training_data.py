#!/usr/bin/env python3
"""
Prepare training data for Whispr speaker diarization and localization.

This script processes the downloaded audio files and annotations to create
a dataset suitable for training the Whispr system, which uses Asteroid for
source separation and custom clustering for speaker diarization.
"""

import os
import json
import shutil
import argparse
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_multi_speaker_audio(input_dir, output_dir):
    """Process multi-speaker audio files and their annotations."""
    multi_speaker_dir = Path(input_dir) / "multi_speaker"
    annotations_dir = Path(input_dir) / "annotations"
    output_multi_dir = Path(output_dir) / "processed" / "multi_speaker"
    output_multi_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each audio file in the multi_speaker directory
    for audio_file in multi_speaker_dir.glob("*.wav"):
        logger.info(f"Processing {audio_file.name}")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        # Find corresponding annotation file
        base_name = audio_file.stem
        rttm_file = annotations_dir / f"{base_name}.rttm"
        
        if not rttm_file.exists():
            logger.warning(f"No annotation file found for {audio_file.name}, skipping")
            continue
        
        # Parse RTTM file to get speaker segments
        segments = parse_rttm(rttm_file)
        
        # Create output metadata
        metadata = {
            "filename": audio_file.name,
            "sample_rate": sr,
            "duration": len(audio) / sr,
            "speakers": []
        }
        
        # Extract segments for each speaker
        for speaker_id, speaker_segments in segments.items():
            speaker_audio = []
            time_segments = []
            
            for start_time, end_time in speaker_segments:
                # Convert time to samples
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Extract segment
                if end_sample <= len(audio):
                    segment = audio[start_sample:end_sample]
                    speaker_audio.append(segment)
                    time_segments.append((start_time, end_time))
            
            if speaker_audio:
                # Concatenate all segments for this speaker
                concatenated = np.concatenate(speaker_audio)
                
                # Save speaker audio
                speaker_file = output_multi_dir / f"{base_name}_{speaker_id}.wav"
                sf.write(speaker_file, concatenated, sr)
                
                # Add to metadata
                metadata["speakers"].append({
                    "id": speaker_id,
                    "segments": time_segments,
                    "audio_file": speaker_file.name
                })
        
        # Save metadata
        with open(output_multi_dir / f"{base_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

def parse_rttm(rttm_file):
    """Parse RTTM file format to extract speaker segments."""
    segments = {}
    
    with open(rttm_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 8:
                    # RTTM format: SPEAKER file_id channel start_time duration <NA> <NA> speaker_id <NA> <NA>
                    _, _, _, start_time, duration, _, _, speaker_id, _, _ = parts + ['<NA>', '<NA>']
                    
                    start_time = float(start_time)
                    end_time = start_time + float(duration)
                    
                    if speaker_id not in segments:
                        segments[speaker_id] = []
                    
                    segments[speaker_id].append((start_time, end_time))
    
    return segments

def process_spatial_audio(input_dir, output_dir):
    """Process spatial audio files with their location information."""
    spatial_audio_dir = Path(input_dir) / "spatial_audio"
    spatial_info_dir = Path(input_dir) / "spatial_info"
    output_spatial_dir = Path(output_dir) / "processed" / "spatial_audio"
    output_spatial_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each audio file in the spatial_audio directory
    for audio_file in spatial_audio_dir.glob("*.wav"):
        logger.info(f"Processing spatial audio: {audio_file.name}")
        
        # Load audio (keeping all channels)
        audio, sr = librosa.load(audio_file, sr=16000, mono=False)
        
        # Find corresponding spatial info file
        base_name = audio_file.stem
        info_file = spatial_info_dir / f"{base_name}_locations.json"
        
        if not info_file.exists():
            logger.warning(f"No spatial info file found for {audio_file.name}, skipping")
            continue
        
        # Load spatial information
        with open(info_file, 'r') as f:
            spatial_info = json.load(f)
        
        # Copy audio file to output directory
        output_audio_file = output_spatial_dir / audio_file.name
        shutil.copy(audio_file, output_audio_file)
        
        # Copy spatial info to output directory
        output_info_file = output_spatial_dir / info_file.name
        shutil.copy(info_file, output_info_file)
        
        # Create a simplified metadata file for easier processing
        metadata = {
            "filename": audio_file.name,
            "sample_rate": sr,
            "channels": audio.shape[0] if audio.ndim > 1 else 1,
            "duration": audio.shape[1] / sr if audio.ndim > 1 else len(audio) / sr,
            "room_dimensions": spatial_info.get("room_dimensions", {}),
            "microphone_array": spatial_info.get("microphone_array", {}),
            "speakers": spatial_info.get("speakers", [])
        }
        
        # Save simplified metadata
        with open(output_spatial_dir / f"{base_name}_processed.json", 'w') as f:
            json.dump(metadata, f, indent=2)

def process_iphone_recordings(input_dir, output_dir):
    """Process iPhone recordings without spatial metadata.
    
    This function handles audio files that don't have accompanying
    spatial metadata by creating estimated metadata based on
    audio characteristics.
    
    Args:
        input_dir: Directory containing iPhone recordings
        output_dir: Directory to save processed data
    """
    iphone_dir = Path(input_dir) / "iphone_samples"
    output_iphone_dir = Path(output_dir) / "processed" / "iphone_samples"
    output_iphone_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each audio file in the iPhone directory
    for audio_file in iphone_dir.glob("*.wav"):
        logger.info(f"Processing iPhone recording: {audio_file.name}")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000, mono=False)
        
        # Check if audio is mono or multi-channel
        is_mono = audio.ndim == 1 or audio.shape[0] == 1
        
        # Copy audio file to output directory
        output_audio_file = output_iphone_dir / audio_file.name
        shutil.copy(audio_file, output_audio_file)
        
        # Estimate room dimensions based on audio length
        audio_length_seconds = len(audio) / sr if is_mono else audio.shape[1] / sr
        room_dims = estimate_room_dimensions(audio_length_seconds)
        
        # Create estimated metadata based on audio analysis
        metadata = {
            "filename": audio_file.name,
            "sample_rate": sr,
            "channels": 1 if is_mono else audio.shape[0],
            "duration": audio_length_seconds,
            "room_dimensions": room_dims,
            "microphone_array": estimate_microphone_array(audio, sr, room_dims),
            "speakers": []
        }
        
        # Attempt to detect speakers and their locations
        if is_mono:
            # For mono recordings, use voice activity detection to find speakers
            estimated_speakers = estimate_speakers_from_mono(audio, sr, room_dims)
        else:
            # For stereo/multi-channel, use channel differences to estimate positions
            estimated_speakers = estimate_speakers_from_multichannel(audio, sr, room_dims)
        
        metadata["speakers"] = estimated_speakers
        
        # Save metadata
        with open(output_iphone_dir / f"{audio_file.stem}_processed.json", 'w') as f:
            json.dump(metadata, f, indent=2)

def estimate_room_dimensions(audio_length_seconds):
    """Estimate reasonable room dimensions based on audio characteristics.
    
    Args:
        audio_length_seconds: Length of the audio in seconds
        
    Returns:
        Dictionary with room dimensions
    """
    # Default dimensions for a medium-sized room
    default_width = 5.0
    default_length = 5.0
    default_height = 2.7
    
    # For very short clips, assume smaller room
    if audio_length_seconds < 10:
        return {
            "width": 3.0,
            "length": 3.0,
            "height": 2.5
        }
    # For very long clips, assume larger room
    elif audio_length_seconds > 60:
        return {
            "width": 8.0,
            "length": 8.0,
            "height": 3.0
        }
    else:
        return {
            "width": default_width,
            "length": default_length,
            "height": default_height
        }

def estimate_microphone_array(audio, sr, room_dims):
    """Estimate microphone array configuration based on audio characteristics.
    
    Args:
        audio: Audio array
        sr: Sample rate
        room_dims: Dictionary with room dimensions
        
    Returns:
        Dictionary with microphone array information
    """
    # Default to center of the room
    center_x = room_dims["width"] / 2
    center_y = room_dims["length"] / 2
    center_z = 1.2  # Typical height for handheld device
    
    # Check if audio is mono or multi-channel
    is_mono = audio.ndim == 1 or audio.shape[0] == 1
    
    if is_mono:
        # For mono, assume single microphone
        return {
            "position": {
                "x": center_x,
                "y": center_y,
                "z": center_z
            },
            "type": "single",
            "channels": 1,
            "orientation": "omnidirectional"
        }
    else:
        # For multi-channel, estimate array type
        num_channels = audio.shape[0]
        
        # Check if it's likely to be a stereo recording
        if num_channels == 2:
            # Analyze correlation between channels to estimate if it's true stereo
            corr = np.corrcoef(audio[0], audio[1])[0, 1]
            
            if corr > 0.9:
                # High correlation suggests it's not true stereo but mono recorded on two channels
                array_type = "pseudo-stereo"
                orientation = "omnidirectional"
            else:
                array_type = "stereo"
                orientation = "bidirectional"
        elif num_channels >= 4:
            # Likely a spatial microphone array
            array_type = "spatial"
            orientation = "3D"
        else:
            # Other configurations
            array_type = f"{num_channels}-channel"
            orientation = "directional"
        
        return {
            "position": {
                "x": center_x,
                "y": center_y,
                "z": center_z
            },
            "type": array_type,
            "channels": num_channels,
            "orientation": orientation,
            "spacing": 0.1  # Assume 10cm between microphones
        }

def estimate_speakers_from_mono(audio, sr, room_dims):
    """Estimate speaker segments and positions from mono audio.
    
    This is a simple heuristic approach. In a real implementation,
    you would use a more sophisticated speaker diarization algorithm.
    
    Args:
        audio: Mono audio array
        sr: Sample rate
        room_dims: Dictionary with room dimensions
        
    Returns:
        List of estimated speaker dictionaries with positions and segments
    """
    # Simple energy-based segmentation
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    # Compute energy
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Simple threshold-based VAD
    threshold = 0.1 * np.max(energy)
    speech_frames = energy > threshold
    
    # Find speech segments
    speech_segments = []
    in_speech = False
    start_frame = 0
    
    for i, is_speech in enumerate(speech_frames):
        if is_speech and not in_speech:
            # Speech start
            in_speech = True
            start_frame = i
        elif not is_speech and in_speech:
            # Speech end
            in_speech = False
            # Only keep segments longer than 0.5 seconds
            if (i - start_frame) * hop_length / sr > 0.5:
                speech_segments.append((start_frame * hop_length / sr, i * hop_length / sr))
    
    # If still in speech at the end
    if in_speech:
        speech_segments.append((start_frame * hop_length / sr, len(speech_frames) * hop_length / sr))
    
    # Simple clustering based on segment proximity
    # In a real implementation, you would use speaker embeddings
    speakers = []
    current_speaker = 1
    last_end = 0
    speaker_segments = {}
    
    for start, end in speech_segments:
        # If gap is more than 1 second, assume new speaker
        if start - last_end > 1.0:
            current_speaker += 1
        
        if current_speaker not in speaker_segments:
            speaker_segments[current_speaker] = []
        
        speaker_segments[current_speaker].append((start, end))
        last_end = end
    
    # Create speaker objects with positions distributed throughout the room
    room_width = room_dims["width"]
    room_length = room_dims["length"]
    room_height = room_dims["height"]
    
    # Distribute speakers in a circle around the center of the room
    center_x = room_width / 2
    center_y = room_length / 2
    radius = min(room_width, room_length) * 0.4  # 40% of the smallest dimension
    
    for i, (speaker_id, segments) in enumerate(speaker_segments.items()):
        # Calculate position on circle
        angle = 2 * np.pi * i / len(speaker_segments)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        z = 1.6  # Assuming standing height
        
        speakers.append({
            "id": f"SPEAKER_{speaker_id:02d}",
            "position": {
                "x": float(x),
                "y": float(y),
                "z": float(z)
            },
            "segments": segments
        })
    
    return speakers

def estimate_speakers_from_multichannel(audio, sr, room_dims):
    """Estimate speaker segments and positions from multi-channel audio.
    
    Uses channel differences to estimate spatial positions.
    
    Args:
        audio: Multi-channel audio array of shape (channels, samples)
        sr: Sample rate
        room_dims: Dictionary with room dimensions
        
    Returns:
        List of estimated speaker dictionaries with positions and segments
    """
    num_channels = audio.shape[0]
    
    # First, convert to mono for segmentation
    mono_audio = np.mean(audio, axis=0)
    
    # Get speech segments using the mono estimation
    speakers = estimate_speakers_from_mono(mono_audio, sr, room_dims)
    
    # Room dimensions
    room_width = room_dims["width"]
    room_length = room_dims["length"]
    room_height = room_dims["height"]
    
    # Now refine the positions using channel information
    for speaker in speakers:
        segments = speaker["segments"]
        channel_energies = []
        channel_delays = []
        
        # Calculate average channel energies and delays for this speaker's segments
        for start, end in segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            # Extract segment from each channel
            channel_segments = []
            for ch in range(num_channels):
                segment = audio[ch, start_sample:end_sample]
                channel_segments.append(segment)
                
                # Calculate energy
                energy = np.sum(np.abs(segment)**2)
                
                if len(channel_energies) <= ch:
                    channel_energies.append(energy)
                else:
                    channel_energies[ch] += energy
            
            # Calculate delays between channels using cross-correlation
            if num_channels >= 2:
                for i in range(num_channels):
                    for j in range(i+1, num_channels):
                        # Cross-correlation to estimate delay
                        corr = np.correlate(channel_segments[i], channel_segments[j], mode='full')
                        max_idx = np.argmax(corr)
                        delay = max_idx - len(channel_segments[i]) + 1  # Convert to sample delay
                        delay_ms = delay * 1000 / sr  # Convert to milliseconds
                        
                        delay_key = f"{i}-{j}"
                        if delay_key not in channel_delays:
                            channel_delays[delay_key] = []
                        channel_delays[delay_key].append(delay_ms)
        
        # Normalize energies
        total_energy = sum(channel_energies) + 1e-8
        channel_energies = [e / total_energy for e in channel_energies]
        
        # Average delays
        avg_delays = {}
        for key, delays in channel_delays.items():
            avg_delays[key] = np.mean(delays)
        
        # Estimate position based on channel configuration
        if num_channels == 2:
            # Stereo: estimate left-right position
            left_right_balance = channel_energies[0] / (channel_energies[0] + channel_energies[1] + 1e-8)
            # Convert to room coordinates (0 = left, 1 = right)
            x_pos = room_width * (1.0 - left_right_balance)  # Map to room width
            y_pos = room_length / 2  # Center of room
            z_pos = room_height * 0.6  # Typical speaking height
            
        elif num_channels >= 4:
            # Quad or more: estimate 2D position
            left_energy = channel_energies[0] + channel_energies[2]  # front-left + rear-left
            right_energy = channel_energies[1] + channel_energies[3]  # front-right + rear-right
            front_energy = channel_energies[0] + channel_energies[1]  # front-left + front-right
            rear_energy = channel_energies[2] + channel_energies[3]  # rear-left + rear-right
            
            # Convert to room coordinates
            lr_ratio = right_energy / (left_energy + right_energy + 1e-8)
            fb_ratio = rear_energy / (front_energy + rear_energy + 1e-8)
            
            x_pos = room_width * lr_ratio
            y_pos = room_length * fb_ratio
            z_pos = room_height * 0.6  # Typical speaking height
            
        else:
            # Other configurations: use simple heuristic
            max_channel = np.argmax(channel_energies)
            angle = 2 * np.pi * max_channel / num_channels
            
            # Convert polar to cartesian coordinates in room space
            x_pos = room_width/2 + (room_width/2 * 0.8) * np.cos(angle)  # 80% of half room width
            y_pos = room_length/2 + (room_length/2 * 0.8) * np.sin(angle)  # 80% of half room length
            z_pos = room_height * 0.6  # Typical speaking height
        
        # Update speaker position
        speaker["position"]["x"] = float(x_pos)
        speaker["position"]["y"] = float(y_pos)
        speaker["position"]["z"] = float(z_pos)
        
        # Add confidence level based on energy difference
        if num_channels >= 2:
            # Calculate energy difference between channels
            energy_diff = max(channel_energies) - min(channel_energies)
            # Higher difference = higher confidence in localization
            speaker["position_confidence"] = float(min(1.0, energy_diff * 5.0))  # Scale to 0-1
        else:
            speaker["position_confidence"] = 0.5  # Medium confidence for mono
    
    return speakers

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Whispr")
    parser.add_argument("--input-dir", type=str, default="data/training",
                        help="Directory containing the downloaded training data")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directory to save processed data")
    args = parser.parse_args()
    
    # Process multi-speaker audio
    process_multi_speaker_audio(args.input_dir, args.output_dir)
    
    # Process spatial audio
    process_spatial_audio(args.input_dir, args.output_dir)
    
    # Process iPhone recordings
    process_iphone_recordings(args.input_dir, args.output_dir)
    
    logger.info("Data preparation complete!")

if __name__ == "__main__":
    main() 