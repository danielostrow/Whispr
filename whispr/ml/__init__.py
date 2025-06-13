"""Machine Learning (ML) module for Whispr.

This module contains the core machine learning logic for speaker diarization.
"""

from .clustering import cluster_speakers  # noqa: F401
from .separation import separate  # noqa: F401
from .vad import simple_energy_vad  # noqa: F401
