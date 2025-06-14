# Whispr 🎙️

[![Python Tests](https://github.com/danielostrow/Whispr/actions/workflows/python-tests.yml/badge.svg)](https://github.com/danielostrow/Whispr/actions/workflows/python-tests.yml)
[![Python Lint](https://github.com/danielostrow/Whispr/actions/workflows/python-lint.yml/badge.svg)](https://github.com/danielostrow/Whispr/actions/workflows/python-lint.yml)
[![C Extensions Build](https://github.com/danielostrow/Whispr/actions/workflows/c-extensions-build.yml/badge.svg)](https://github.com/danielostrow/Whispr/actions/workflows/c-extensions-build.yml)
[![Docker Build and Test](https://github.com/danielostrow/Whispr/actions/workflows/docker-build.yml/badge.svg)](https://github.com/danielostrow/Whispr/actions/workflows/docker-build.yml)

In crowded and noisy environments, it's challenging to isolate and identify individual voices from an audio recording. **Whispr** is a tool designed to address this problem. It takes a pre-recorded audio file, performs signal analysis to extract and tag human speech, and uses clustering algorithms to separate distinct speakers.

The end goal is to produce an interactive spatial map visualizing where and when speakers were talking. This can be valuable for journalists, security analysts, or researchers working with surveillance audio, interview transcriptions, and event recordings. This project uses a combination of modern signal processing, machine learning, and a web-based UI to provide an intuitive way to explore complex audio scenes.

## Features

- **Voice Activity Detection (VAD)**: Identifies segments of speech in an audio file.
- **Speaker Clustering**: Groups speech segments by speaker using MFCC features and clustering.
- **Source Separation**: Attempts to isolate individual speaker audio, even in cases of overlap, using [Asteroid](https://github.com/asteroid-team/asteroid).
- **Spatial Localization**: Estimates the location of each speaker (placeholder implementation).
- **Interactive UI**: A [Dash](https://plotly.com/dash/) application to visualize speaker locations and play back their audio.
- **High-Performance C Extensions**: Optional optimized implementations of critical DSP and ML functions for improved performance.
- **Docker Support**: Run the entire application in a containerized environment without local setup.
- **CI/CD Workflows**: Automated testing and validation across multiple platforms.

## Project Structure

```
Whispr/
├── .github/             # GitHub Actions workflows for CI/CD
├── .venv/               # Python virtual environment
├── output/              # Default directory for output files
├── whispr/              # Main Python package
│   ├── c_ext/           # C extensions for optimized performance
│   ├── dsp/             # Digital Signal Processing modules
│   ├── io/              # Input/Output handling
│   ├── ml/              # Machine Learning models (VAD, clustering, separation)
│   └── ui/              # User Interface (Dash app)
├── .flake8              # Flake8 configuration
├── CHANGELOG.md         # Version history and changes
├── Dockerfile           # Docker container definition
├── pyproject.toml       # Black configuration
├── README.md            # This file
├── requirements-dev.txt # Development dependencies
├── requirements.txt     # Project dependencies
└── run.sh               # Main execution script
```

## Getting Started

### Prerequisites

- Python 3.11+
- `ffmpeg` (for handling various audio formats)
  - On macOS (via Homebrew): `brew install ffmpeg`
  - On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
- For C extensions (optional):
  - C compiler (GCC, Clang, etc.)
  - Python development headers
  - For OpenMP support on macOS: `brew install libomp`

### Installation

#### Option 1: Local Installation

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

#### Option 2: Docker Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danielostrow/Whispr.git
   cd Whispr
   ```

2. **Build and run with Docker:**
   ```bash
   docker build -t whispr .
   ```

## How to Run

### Using the run.sh Script

The easiest way to run the project is with the `run.sh` script. It automates code formatting, pipeline execution, and UI startup.

**Provide an audio file (e.g., WAV, MP3) as an argument:**

```bash
./run.sh /path/to/your/audio.wav
```

**Using Docker with run.sh:**

```bash
./run.sh -d /path/to/your/audio.wav
```

The script will:
1.  Format the code with `isort` and `black`.
2.  Run the processing pipeline on your audio file.
3.  Save the results (audio clips and `metadata.json`) to the `output/` directory.
4.  Launch the interactive Dash UI.

Open your web browser to **http://127.0.0.1:8050** to see the speaker map. Click on a speaker to hear their isolated audio.

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

### Using Docker Directly

```bash
# Process an audio file
docker run --rm -v /path/to/your/audio.wav:/app/input/audio.wav -v $(pwd)/output:/app/output whispr:latest /app/input/audio.wav --output-dir /app/output

# Launch UI (on local system)
python whispr/ui/app.py output/metadata.json
```

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

### Continuous Integration

This project uses GitHub Actions for continuous integration and testing:

- **Python Tests**: Runs unit tests on multiple platforms and Python versions
- **Python Lint**: Enforces code quality with flake8, black, and isort
- **C Extensions Build**: Tests C extensions across platforms
- **Integration Tests**: Tests the full pipeline with sample audio
- **Docker Build**: Ensures the Docker container builds and works correctly

### Automated Changelog

The project includes an automated changelog update system that works when pull requests are merged to the main branch.

#### Pull Request Title Format

For automatic changelog updates to work properly, format your PR titles following these conventions:

- **New features**: Start with `feat:` or include `feature` or `add` in the title
  - Example: `feat: add speaker visualization feature`
  - Will be categorized under: **Added**

- **Bug fixes**: Start with `fix:` or include `bug` in the title
  - Example: `fix: resolve audio playback issue in Firefox`
  - Will be categorized under: **Fixed**

- **Breaking changes/removals**: Include `deprecate`, `remove`, or `drop` in the title
  - Example: `remove: drop support for Python 3.7`
  - Will be categorized under: **Removed**

- **Security fixes**: Include `security` in the title
  - Example: `security: update vulnerable dependency`
  - Will be categorized under: **Security**

- **Other changes**: Any other title
  - Example: `improve performance of clustering algorithm`
  - Will be categorized under: **Changed**

The PR number will be automatically appended to each changelog entry for reference.

### Coding Style

This project uses `black` for formatting and `flake8` for linting. Configuration is in `pyproject.toml` and `.flake8`. The `run.sh` script automatically applies formatting.

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. To set up pre-commit:

```bash
pip install pre-commit
pre-commit install
```

This will automatically format your code with black and isort, and check for other issues before each commit. 