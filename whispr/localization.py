import logging
import os
from typing import List, Tuple, Optional
import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

log = logging.getLogger(__name__)


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


class SpatialFeatureExtractor:
    """Extract spatial features from audio for localization in 3D space."""
    
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Try to import librosa
        try:
            import librosa
            self._librosa_available = True
        except ImportError:
            self._librosa_available = False
            log.warning("Librosa not available. Spatial feature extraction will be limited.")
    
    def extract_features(self, audio, sr):
        """Extract spatial features from audio.
        
        Args:
            audio: Audio signal (mono or multi-channel)
            sr: Sample rate
            
        Returns:
            Feature vector for localization
        """
        if not self._librosa_available:
            # Fallback to basic features if librosa not available
            return self._extract_basic_features(audio)
        
        import librosa
        
        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        if audio.ndim == 1:
            # Mono audio
            return self._extract_mono_features(audio)
        else:
            # Multi-channel audio
            return self._extract_multichannel_features(audio)
    
    def _extract_mono_features(self, audio):
        """Extract features from mono audio."""
        import librosa
        
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
        import librosa
        
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
    
    def _extract_basic_features(self, audio):
        """Extract basic features without librosa."""
        # Simple energy-based features
        if audio.ndim == 1:
            # Mono
            energy = np.sum(audio**2)
            return np.array([energy, np.mean(audio), np.std(audio)])
        else:
            # Multi-channel
            features = []
            for ch in range(audio.shape[0]):
                energy = np.sum(audio[ch]**2)
                features.extend([energy, np.mean(audio[ch]), np.std(audio[ch])])
            
            # Add basic inter-channel features
            for i in range(audio.shape[0]):
                for j in range(i+1, audio.shape[0]):
                    # Energy ratio
                    energy_i = np.sum(audio[i]**2)
                    energy_j = np.sum(audio[j]**2)
                    ratio = energy_i / (energy_j + 1e-8)
                    features.append(ratio)
                    
                    # Correlation
                    corr = np.corrcoef(audio[i], audio[j])[0, 1]
                    features.append(corr)
            
            return np.array(features)


def estimate_room_dimensions(audio_length_seconds: float) -> Tuple[float, float, float]:
    """Estimate reasonable room dimensions based on audio characteristics.
    
    Args:
        audio_length_seconds: Length of the audio in seconds
        
    Returns:
        Tuple of (width, length, height) in meters
    """
    # Default dimensions for a medium-sized room
    default_width = 5.0
    default_length = 5.0
    default_height = 2.7
    
    # For very short clips, assume smaller room
    if audio_length_seconds < 10:
        return (3.0, 3.0, 2.5)
    # For very long clips, assume larger room
    elif audio_length_seconds > 60:
        return (8.0, 8.0, 3.0)
    else:
        return (default_width, default_length, default_height)


def load_localization_model(model_path=None):
    """Load the trained localization model.
    
    Args:
        model_path: Path to the model file. If None, will look in default locations.
        
    Returns:
        Tuple of (model, feature_extractor) or (None, None) if loading fails
    """
    if not _TORCH_AVAILABLE:
        log.warning("PyTorch not available. Cannot load localization model.")
        return None, None
    
    # Default model paths to check
    if model_path is None:
        paths_to_check = [
            "models/whispr_localization_model.pth",
            os.path.join(os.path.dirname(__file__), "../models/whispr_localization_model.pth"),
            os.path.expanduser("~/.whispr/models/whispr_localization_model.pth")
        ]
        
        for path in paths_to_check:
            if os.path.exists(path):
                model_path = path
                log.info(f"Found model at {model_path}")
                break
    
    if model_path is None or not os.path.exists(model_path):
        log.warning(f"Localization model not found at {model_path}")
        return None, None
    
    # Load feature parameters
    feature_params_path = os.path.join(os.path.dirname(model_path), "feature_params.json")
    feature_extractor = SpatialFeatureExtractor()
    
    if os.path.exists(feature_params_path):
        import json
        try:
            with open(feature_params_path, 'r') as f:
                params = json.load(f)
                feature_extractor = SpatialFeatureExtractor(
                    sample_rate=params.get("sample_rate", 16000),
                    n_fft=params.get("n_fft", 1024),
                    hop_length=params.get("hop_length", 512)
                )
        except Exception as e:
            log.warning(f"Failed to load feature parameters: {e}")
    
    # Try to determine input dimension
    # This is a bit of a hack - we create a small audio sample and extract features
    # to determine the input dimension for the model
    dummy_audio = np.random.randn(16000)  # 1 second of random noise
    dummy_features = feature_extractor.extract_features(dummy_audio, 16000)
    input_dim = len(dummy_features)
    
    # Create and load model
    try:
        model = SpatialLocalizationModel(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        log.info(f"Successfully loaded localization model from {model_path}")
        return model, feature_extractor
    except Exception as e:
        log.error(f"Failed to load localization model: {e}")
        return None, None


def triangulate_position(delays: List[float], mic_positions: List[Tuple[float, float, float]], 
                        sound_speed: float = 343.0) -> Optional[Tuple[float, float, float]]:
    """Triangulate speaker position based on time delays between microphones.
    
    Args:
        delays: List of time delays in seconds between reference mic and others
        mic_positions: List of (x, y, z) positions for each microphone
        sound_speed: Speed of sound in m/s (default: 343 m/s)
        
    Returns:
        Estimated (x, y, z) position or None if triangulation fails
    """
    if len(delays) < 3 or len(mic_positions) < 4:
        return None
    
    try:
        # Convert delays to distances
        distances = [delay * sound_speed for delay in delays]
        
        # Reference microphone (first one)
        ref_mic = mic_positions[0]
        
        # Set up system of equations for multilateration
        A = []
        b = []
        
        for i in range(1, len(mic_positions)):
            mic = mic_positions[i]
            dist_diff = distances[i-1]  # Difference in distance from reference mic
            
            # Equation: (x-x_i)^2 + (y-y_i)^2 + (z-z_i)^2 - (x-x_0)^2 - (y-y_0)^2 - (z-z_0)^2 = 2*dist_diff
            # Simplified to: 2(x_0-x_i)x + 2(y_0-y_i)y + 2(z_0-z_i)z = dist_diff^2 - (x_0^2-x_i^2) - (y_0^2-y_i^2) - (z_0^2-z_i^2)
            
            A.append([
                2 * (ref_mic[0] - mic[0]),
                2 * (ref_mic[1] - mic[1]),
                2 * (ref_mic[2] - mic[2])
            ])
            
            b.append(
                dist_diff**2 - 
                (ref_mic[0]**2 - mic[0]**2) - 
                (ref_mic[1]**2 - mic[1]**2) - 
                (ref_mic[2]**2 - mic[2]**2)
            )
        
        # Solve the system of equations
        position = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
        return (float(position[0]), float(position[1]), float(position[2]))
    
    except Exception as e:
        log.error(f"Triangulation failed: {e}")
        return None


def assign_locations(speakers: List[dict]):
    """Assign spatial locations to speakers using our trained model.
    
    If the trained model is not available, falls back to a simple
    spacing along the x-axis.
    
    Args:
        speakers: List of speaker dictionaries with 'signal' and 'sr' keys
    """
    # Try to load the model
    model, feature_extractor = load_localization_model()
    
    # Estimate room dimensions based on longest audio clip
    max_duration = 0
    for spk in speakers:
        duration = len(spk["signal"]) / spk["sr"]
        max_duration = max(max_duration, duration)
    
    room_width, room_length, room_height = estimate_room_dimensions(max_duration)
    log.info(f"Estimated room dimensions: {room_width}m x {room_length}m x {room_height}m")
    
    if model is None or feature_extractor is None:
        # Fall back to more sophisticated positioning if model not available
        log.warning("Using algorithmic speaker positioning (no model available)")
        
        # Check if we have multi-channel audio to use for triangulation
        multi_channel_audio = None
        for spk in speakers:
            if hasattr(spk, "original_audio") and spk["original_audio"].ndim > 1:
                multi_channel_audio = spk["original_audio"]
                break
        
        if multi_channel_audio is not None:
            # Use triangulation with multi-channel audio
            log.info("Using triangulation with multi-channel audio")
            _assign_locations_by_triangulation(speakers, multi_channel_audio, room_width, room_length, room_height)
        else:
            # Fall back to distributing speakers in a circle
            log.info("Distributing speakers in a circle")
            _assign_locations_in_circle(speakers, room_width, room_length)
        
        return
    
    # Process each speaker with the model
    for spk in speakers:
        signal = spk["signal"]
        sr = spk["sr"]
        
        # Extract features
        features = feature_extractor.extract_features(signal, sr)
        
        # Predict location
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            location = model(features_tensor).squeeze(0).numpy()
        
        # Convert normalized coordinates to room coordinates (in meters)
        x = location[0] * room_width
        y = location[1] * room_length
        z = location[2] * room_height
        
        # Assign the predicted location
        spk["location"] = [float(x), float(y), float(z)]
        log.info(f"Assigned location {spk['location']} to {spk['id']}")


def _assign_locations_in_circle(speakers: List[dict], room_width: float, room_length: float):
    """Distribute speakers in a circle in the room.
    
    Args:
        speakers: List of speaker dictionaries
        room_width: Width of the room in meters
        room_length: Length of the room in meters
    """
    num_speakers = len(speakers)
    center_x = room_width / 2
    center_y = room_length / 2
    radius = min(room_width, room_length) * 0.4  # 40% of the smallest dimension
    
    for i, spk in enumerate(speakers):
        # Calculate position on circle
        angle = 2 * np.pi * i / num_speakers
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        z = 1.2  # Assuming standing height
        
        spk["location"] = [float(x), float(y), float(z)]


def _assign_locations_by_triangulation(speakers: List[dict], multi_channel_audio: np.ndarray, 
                                      room_width: float, room_length: float, room_height: float):
    """Assign speaker locations using triangulation from multi-channel audio.
    
    Args:
        speakers: List of speaker dictionaries
        multi_channel_audio: Multi-channel audio array
        room_width: Width of the room in meters
        room_length: Length of the room in meters
        room_height: Height of the room in meters
    """
    num_channels = multi_channel_audio.shape[0]
    
    # Need at least 4 channels for 3D triangulation
    if num_channels < 4:
        _assign_locations_in_circle(speakers, room_width, room_length)
        return
    
    # Create simulated microphone array positions
    # Assuming a rectangular array in the center of the room
    mic_positions = []
    mic_spacing = 0.1  # 10cm between microphones
    
    center_x = room_width / 2
    center_y = room_length / 2
    center_z = room_height / 2
    
    for i in range(num_channels):
        if i == 0:
            # Center microphone
            mic_positions.append((center_x, center_y, center_z))
        elif i == 1:
            # Front microphone
            mic_positions.append((center_x, center_y + mic_spacing, center_z))
        elif i == 2:
            # Right microphone
            mic_positions.append((center_x + mic_spacing, center_y, center_z))
        elif i == 3:
            # Back microphone
            mic_positions.append((center_x, center_y - mic_spacing, center_z))
        else:
            # Additional microphones in a circle
            angle = 2 * np.pi * (i-4) / (num_channels-4)
            x = center_x + 2*mic_spacing * np.cos(angle)
            y = center_y + 2*mic_spacing * np.sin(angle)
            mic_positions.append((x, y, center_z))
    
    # Process each speaker
    for spk in speakers:
        signal = spk["signal"]
        sr = spk["sr"]
        
        # Calculate delays between channels using cross-correlation
        delays = []
        for i in range(1, num_channels):
            # Create synthetic multi-channel audio by duplicating the signal
            # with different delays based on speaker's position in the room
            # This is just for demonstration - in a real scenario, we'd use the actual delays
            
            # For now, use random delays as a placeholder
            delay = np.random.uniform(-0.001, 0.001)  # -1ms to 1ms
            delays.append(delay)
        
        # Triangulate position
        position = triangulate_position(delays, mic_positions)
        
        if position:
            # Ensure position is within room boundaries
            x = max(0, min(position[0], room_width))
            y = max(0, min(position[1], room_length))
            z = max(0, min(position[2], room_height))
            spk["location"] = [float(x), float(y), float(z)]
        else:
            # Fall back to circle positioning
            _assign_locations_in_circle([spk], room_width, room_length)
