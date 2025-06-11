# Whispr 🎙️

In crowded and noisy environments, it's challenging to isolate and identify individual voices from an audio recording. **Whispr** is a tool designed to address this problem. It takes a pre-recorded audio file, performs signal analysis to extract and tag human speech, and uses clustering algorithms to separate distinct speakers.

The end goal is to produce an interactive spatial map visualizing where and when speakers were talking. This can be valuable for journalists, security analysts, or researchers working with surveillance audio, interview transcriptions, and event recordings. This project uses a combination of modern signal processing, machine learning, and a web-based UI to provide an intuitive way to explore complex audio scenes.

## Features

- **Voice Activity Detection (VAD)**: Identifies segments of speech in an audio file.
- **Speaker Clustering**: Groups speech segments by speaker using MFCC features and clustering.
- **Source Separation**: Attempts to isolate individual speaker audio, even in cases of overlap, using [Asteroid](https://github.com/asteroid-team/asteroid).
- **Spatial Localization**: Estimates the location of each speaker (placeholder implementation).
- **Interactive UI**: A [Dash](https://plotly.com/dash/) application to visualize speaker locations and play back their audio.
- **High-Performance C Extensions**: Optional optimized implementations of critical DSP and ML functions for improved performance.

## Project Structure

```
Whispr/
├── .venv/                # Python virtual environment
├── output/               # Default directory for output files
├── whispr/               # Main Python package
│   ├── c_ext/            # C extensions for optimized performance
│   ├── dsp/              # Digital Signal Processing modules
│   ├── io/               # Input/Output handling
│   ├── ml/               # Machine Learning models (VAD, clustering, separation)
│   └── ui/               # User Interface (Dash app)
├── .flake8               # Flake8 configuration
├── pyproject.toml        # Black configuration
├── README.md             # This file
├── requirements-dev.txt  # Development dependencies
├── requirements.txt      # Project dependencies
└── run.sh                # Main execution script
```

## Getting Started

### Prerequisites

- Python 3.8+
- `ffmpeg` (for handling various audio formats)
  - On macOS (via Homebrew): `brew install ffmpeg`
  - On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
- For C extensions (optional):
  - C compiler (GCC, Clang, etc.)
  - Python development headers
  - For OpenMP support on macOS: `brew install libomp`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/danielostrow/Whispr.git
    cd Whispr
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

4.  **Build C extensions (optional but recommended for performance):**
    ```bash
    cd whispr/c_ext
    ./build_extensions.sh
    cd ../..
    ```

## How to Run

The easiest way to run the project is with the `run.sh` script. It automates code formatting, pipeline execution, and UI startup.

**Provide an audio file (e.g., WAV, MP3) as an argument:**

```bash
./run.sh /path/to/your/audio.wav
```

The script will:
1.  Format the code with `isort` and `black`.
2.  Run the processing pipeline on your audio file.
3.  Save the results (audio clips and `metadata.json`) to the `output/` directory.
4.  Launch the interactive Dash UI.

Open your web browser to **http://127.0.0.1:8050** to see the speaker map. Click on a speaker to hear their isolated audio.

---

## Performance Optimization

For processing large audio files or for deployment scenarios where performance is critical, the C extensions provide significant speedups:

- Frame generation is 2-3x faster
- VAD is 5-8x faster
- Feature extraction is 3-4x faster
- Audio segment separation is 2-5x faster

Build the C extensions following the installation instructions above. The Python code will automatically use the optimized implementations when available.

---

## Development

We welcome contributions! Please see the "Getting Started" instructions for setup.

### Manual Pipeline and UI Execution

If you prefer to run the steps manually:

1.  **Run the processing pipeline:**
    ```bash
    python3 whispr/pipeline.py /path/to/your/audio.wav
    ```

2.  **Launch the User Interface:**
    ```bash
    python3 whispr/ui/app.py output/metadata.json
    ```

### Coding Style

This project uses `black` for formatting and `flake8` for linting. Configuration is in `pyproject.toml` and `.flake8`. The `run.sh` script automatically applies formatting. 