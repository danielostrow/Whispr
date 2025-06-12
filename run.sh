#!/bin/bash
set -e

# --- Configuration ---
AUDIO_FILE=$1
OUTPUT_DIR="output"
VENV_DIR=".venv"
FORCE_REBUILD=""
USE_DOCKER=""

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--rebuild)
            FORCE_REBUILD="yes"
            shift
            ;;
        -f|--file)
            AUDIO_FILE="$2"
            shift 2
            ;;
        -d|--docker)
            USE_DOCKER="yes"
            shift
            ;;
        -h|--help)
            echo "Usage: ./run.sh [OPTIONS] <path_to_audio_file>"
            echo ""
            echo "Options:"
            echo "  -o, --output-dir DIR    Set custom output directory (default: output)"
            echo "  -r, --rebuild           Force rebuild of C extensions"
            echo "  -f, --file FILE         Specify audio file (alternative to positional arg)"
            echo "  -d, --docker            Use Docker container instead of local environment"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example: ./run.sh -r -o custom_output audio.wav"
            echo "Example with Docker: ./run.sh -d -f audio.wav"
            exit 0
            ;;
        *)
            # First non-option arg is the audio file (if not already set)
            if [[ -z "$AUDIO_FILE" ]]; then
                AUDIO_FILE="$1"
            fi
            shift
            ;;
    esac
done

# --- Validation ---
if [ -z "$AUDIO_FILE" ]; then
    echo "⚠️ Error: No audio file provided."
    echo "Usage: ./run.sh [OPTIONS] <path_to_audio_file>"
    echo "Try './run.sh --help' for more information."
    exit 1
fi

if [ ! -f "$AUDIO_FILE" ]; then
    echo "⚠️ Error: Audio file not found at '$AUDIO_FILE'"
    exit 1
fi

# --- Ensure Output Directory Exists ---
mkdir -p "$OUTPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"

# --- Docker Execution Path ---
if [[ "$USE_DOCKER" == "yes" ]]; then
    echo "🐳 Using Docker to run Whispr"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed or not in PATH."
        echo "Please install Docker or run without the --docker flag."
        exit 1
    fi
    
    # Build the Docker image if needed
    echo "🔨 Building Docker image..."
    docker build -t whispr:latest .
    
    # Get absolute paths for mounting
    AUDIO_FILE_ABS=$(realpath "$AUDIO_FILE")
    OUTPUT_DIR_ABS=$(realpath "$OUTPUT_DIR")
    
    echo "🚀 Running Whispr in Docker container..."
    docker run --rm -v "$AUDIO_FILE_ABS:/app/input/$(basename "$AUDIO_FILE")" \
               -v "$OUTPUT_DIR_ABS:/app/output" \
               whispr:latest "/app/input/$(basename "$AUDIO_FILE")" --output-dir /app/output
    
    echo "✅ Processing complete. Results saved to $OUTPUT_DIR"
    
    # Launch UI if we're on a system with a display
    if [[ -n "$DISPLAY" || "$OSTYPE" == "darwin"* ]]; then
        echo "📊 Launching UI..."
        python -m whispr.ui.app "$OUTPUT_DIR/metadata.json"
    else
        echo "⚠️ No display detected. UI not launched."
        echo "To view results, run: python -m whispr.ui.app $OUTPUT_DIR/metadata.json"
    fi
    
    exit 0
fi

# --- OS Detection ---
VENV_ACTIVATE_PATH=""
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    VENV_ACTIVATE_PATH="$VENV_DIR/bin/activate"
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    VENV_ACTIVATE_PATH="$VENV_DIR/Scripts/activate"
else
    echo "⚠️ Unsupported OS: $OSTYPE. Cannot determine venv activation script."
    exit 1
fi

# --- Activate Virtual Environment ---
if [ ! -f "$VENV_ACTIVATE_PATH" ]; then
    echo "⚠️ Python virtual environment not found. Setting up environment..."
    
    # Create venv
    echo "📦 Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_ACTIVATE_PATH"
    
    # Install dependencies
    echo "📥 Installing dependencies..."
    pip install --upgrade pip
    pip install -e .
    pip install -r requirements.txt -r requirements-dev.txt
else
    source "$VENV_ACTIVATE_PATH"
    echo "✅ Virtual environment activated."
fi

# --- Build C Extensions ---
EXTENSIONS_BUILD_SCRIPT="whispr/c_ext/build_extensions.sh"
if [ -f "$EXTENSIONS_BUILD_SCRIPT" ]; then
    # Fix missing stdbool.h in vad.c
    if ! grep -q "#include <stdbool.h>" whispr/c_ext/src/vad.c; then
        echo "🔧 Adding stdbool.h include to vad.c..."
        sed -i '3a #include <stdbool.h>' whispr/c_ext/src/vad.c
    fi
    
    if [[ "$FORCE_REBUILD" == "yes" ]]; then
        echo "🔨 Forcing rebuild of C extensions..."
        chmod +x "$EXTENSIONS_BUILD_SCRIPT"
        (cd whispr/c_ext && ./build_extensions.sh)
    else
        # Check if C extensions are already built
        python3 -c "from whispr.c_ext import C_EXTENSIONS_AVAILABLE; exit(0 if C_EXTENSIONS_AVAILABLE else 1)" 2>/dev/null || {
            echo "🔧 Building C extensions for improved performance..."
            chmod +x "$EXTENSIONS_BUILD_SCRIPT"
            (cd whispr/c_ext && ./build_extensions.sh)
        }
    fi
    
    # Verify C extensions were built successfully
    if python3 -c "from whispr.c_ext import C_EXTENSIONS_AVAILABLE; exit(0 if C_EXTENSIONS_AVAILABLE else 1)" 2>/dev/null; then
        echo "✅ C extensions are available and will be used for performance."
    else
        echo "⚠️ C extensions could not be built. Using slower Python implementations."
        echo "   Run with --rebuild flag to try again or check for compile errors."
    fi
fi

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
python3 -m whispr.pipeline "$AUDIO_FILE" --output-dir "$OUTPUT_DIR"

echo "✅ Pipeline complete. Metadata saved to: $METADATA_PATH"

# 3. Launch UI
echo "📊 Launching UI..."
# The -u flag is for unbuffered output
python3 -u -m whispr.ui.app "$METADATA_PATH"

echo "✅ All done." 