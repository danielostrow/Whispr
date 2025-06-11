#!/bin/bash
set -e

# Ensure we're in the c_ext directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "📦 Building Whispr C extensions..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    echo "🐧 Detected Linux OS"
    # Check if we need to install development tools
    if ! command -v gcc &> /dev/null; then
        echo "⚠️  gcc not found. Please install build essentials:"
        echo "sudo apt-get update && sudo apt-get install -y build-essential python3-dev"
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    echo "🍎 Detected macOS"
    # Check for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo "🔍 Detected Apple Silicon (M1/M2)"
        # Check for libomp
        if ! brew list libomp &>/dev/null; then
            echo "⚠️  OpenMP not found. Installing with Homebrew..."
            brew install libomp
        else
            echo "✅ OpenMP found."
        fi
    else
        echo "🔍 Detected Intel Mac"
        # Check for libomp
        if ! brew list libomp &>/dev/null; then
            echo "⚠️  OpenMP not found. Installing with Homebrew..."
            brew install libomp
        else
            echo "✅ OpenMP found."
        fi
    fi
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="Windows"
    echo "🪟 Detected Windows"
    # Ensure Visual C++ Build Tools or MinGW is installed
    # (This is challenging to check automatically)
    echo "⚠️  Please ensure you have installed Visual C++ Build Tools or MinGW."
else
    OS="Unknown"
    echo "⚠️  Unknown OS: $OSTYPE"
    echo "Proceeding with generic build settings."
fi

# Ensure numpy is installed
echo "🔧 Checking for NumPy..."
if ! pip show numpy &>/dev/null; then
    echo "📥 Installing NumPy..."
    pip install numpy
else
    echo "✅ NumPy already installed."
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ *.so *.pyd

# Build and install the extensions
echo "🚀 Building C extensions..."
pip install -e .

# Verify installation
echo "🔍 Verifying installation..."
if python -c "from whispr.c_ext import C_EXTENSIONS_AVAILABLE; print('✅ C extensions available!' if C_EXTENSIONS_AVAILABLE else '❌ C extensions not available!')"; then
    echo "✅ C extensions built successfully!"
    
    # Run tests if available
    if [ -f "test_extensions.py" ]; then
        echo ""
        echo "🧪 Running extension tests..."
        echo "------------------------------"
        chmod +x test_extensions.py
        ./test_extensions.py
        echo "------------------------------"
    fi
    
    echo "You can now import them in your code from whispr.c_ext"
else
    echo "❌ Error: C extensions could not be imported after building."
    echo "Check the error messages above for more information."
    exit 1
fi 