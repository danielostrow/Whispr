# Whispr

In crowded and noisy environments, it's challenging to isolate and identify individual voices from an audio recording. This project addresses that problem by developing Whispr, a tool that takes pre-recorded audio files and performs signal analysis to extract and tag human speech from background noise.

## How it Works

Using GNU Radio for preprocessing and Fast Fourier Transform for frequency analysis, the program identifies voice patterns and applies basic clustering algorithms to separate and profile distinct speakers. The end goal is to produce a spatial map of where, when, and what the speakers were saying in the space from the audio files alone.

## Potential Applications

This tool could be valuable for journalists, security analysts, or researchers working with surveillance audio, interview transcriptions, and event recordings.

## Project Structure

- `whispr/`: Main Python package
  - [`dsp/`](whispr/dsp/README.md): Digital Signal Processing modules
  - [`io/`](whispr/io/README.md): Input/Output modules
  - [`ml/`](whispr/ml/README.md): Machine Learning modules
  - [`ui/`](whispr/ui/README.md): User Interface modules
- `output/`: Default directory for output files
- `.venv/`: Python virtual environment
- `requirements.txt`: Project dependencies


1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

The project has two main parts: a processing pipeline that analyzes an audio file and a web-based UI to explore the results.

1.  **Run the processing pipeline:**

    You need an audio file (e.g., WAV, MP3) with speech to process. Place it somewhere accessible. Then, run the pipeline:

    ```bash
    python -m whispr.pipeline path/to/your/audio.wav
    ```

    This will create an `output/` directory containing the processed audio segments and a `metadata.json` file.

2.  **Launch the User Interface:**

    Once the pipeline has finished, you can start the Dash UI to visualize the speaker locations:

    ```bash
    python -m whispr.ui.app output/metadata.json
    ```

    Open your web browser and navigate to `http://127.0.0.1:8050/` to see the speaker map. Click on a speaker to hear their isolated audio.

## Development and Contribution

We welcome contributions to Whispr! If you'd like to help improve the tool, please follow these guidelines.

### Development Setup

Follow the "Getting Started" instructions to set up your environment.

### Contribution Workflow

1.  Fork the repository on GitHub.
2.  Create a new branch for your feature or bug fix: `git checkout -b feature/my-new-feature` or `bugfix/issue-number`.
3.  Make your changes and commit them with clear, descriptive messages.
4.  Push your branch to your fork: `git push origin feature/my-new-feature`.
5.  Create a Pull Request from your fork to the main Whispr repository.

### Coding Style

This project uses standard Python style guides (PEP 8). We recommend using formatters and linters like `black` and `flake8` to maintain code quality. 