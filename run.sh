#!/bin/bash
set -e

# --- Configuration ---
AUDIO_FILE=$1
OUTPUT_DIR="output"
VENV_DIR=".venv"

# --- OS Detection ---
VENV_ACTIVATE_PATH=""
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    VENV_ACTIVATE_PATH="$VENV_DIR/bin/activate"
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    VENV_ACTIVATE_PATH="$VENV_DIR/Scripts/activate"
else
    echo "Unsupported OS: $OSTYPE. Cannot determine venv activation script."
    exit 1
fi

# --- Validation ---
if [ -z "$AUDIO_FILE" ]; then
    echo "Error: No audio file provided."
    echo "Usage: ./run.sh <path_to_audio_file>"
    exit 1
fi

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found at '$AUDIO_FILE'"
    exit 1
fi

# --- Activate Virtual Environment ---
if [ ! -f "$VENV_ACTIVATE_PATH" ]; then
    echo "Python virtual environment not found. Please run:"
    echo "python3 -m venv .venv"
    echo "Then install dependencies with:"
    echo "pip install -r requirements.txt -r requirements-dev.txt"
    exit 1
fi
source "$VENV_ACTIVATE_PATH"
echo "✅ Virtual environment activated."

# --- Main Logic ---
echo "▶️  Starting Whispr process for: $AUDIO_FILE"

# 1. Format and Lint
echo "🎨 Formatting and linting code..."
python3 -m isort whispr/
python3 -m black whispr/
echo "✅ Code formatted."
echo "🔬 Linting with flake8..."
python3 -m flake8 whispr/
echo "✅ Linting complete."


# 2. Run Processing Pipeline
echo "🚀 Running processing pipeline..."
METADATA_PATH="$OUTPUT_DIR/metadata.json"
python3 -u whispr/pipeline.py "$AUDIO_FILE" --output-dir "$OUTPUT_DIR"

echo "✅ Pipeline complete. Metadata saved to: $METADATA_PATH"

# 3. Launch UI
echo "📊 Launching UI..."
# The -u flag is for unbuffered output
python3 -u whispr/ui/app.py "$METADATA_PATH"

echo "✅ All done." 