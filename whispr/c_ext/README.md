# Whispr C Extensions

This module contains high-performance C implementations of core audio processing functions used in Whispr. These optimized versions provide significant speed improvements for CPU-intensive operations.

## Optimized Functions

The following functions have been optimized:

1. **Frame Signal** (`framing_c.frame_signal_c`) - Creates overlapping frames from an audio signal with optimized memory access patterns.

2. **VAD** (`vad_c.simple_energy_vad_c`) - Energy-based Voice Activity Detection with optimized median calculation and segment detection.

3. **Frame Energy** (`features_c.compute_frame_energy`) - Fast windowing and energy calculation with OpenMP parallelization where available.

4. **Audio Separation** (`separation_c.separate_by_segmentation_c`) - Memory-efficient speaker separation by segments.

## Building the Extensions

### Prerequisites

The build process is OS-specific but has been designed to work on all major platforms:

#### Linux
- GCC or compatible C compiler
- Python development headers (`python3-dev` or `python-devel` package)
- OpenMP for parallelization

#### macOS
- Clang compiler (comes with Xcode Command Line Tools)
- OpenMP support (install via Homebrew: `brew install libomp`)
- Different paths are automatically detected for Intel and Apple Silicon Macs

#### Windows
- Visual Studio Build Tools with C/C++ compiler
- Or MinGW-w64 for GCC on Windows
- Python development headers (installed with Python)

#### Common Requirements
- NumPy (will be installed automatically if missing)

### Quick Build

Use the provided script which handles OS-specific requirements:

```bash
./build_extensions.sh
```

### Manual Build

```bash
pip install -e .
```

## Automatic Usage in Whispr

The extensions are automatically used when available. The Python implementations will be used as a fallback if the C extensions cannot be built or imported. 

To force a rebuild of the extensions, you can run:

```bash
./run.sh --rebuild your_audio_file.wav
```

This will ensure you're using the latest compiled versions.

## Performance Benefits

The C extensions provide several advantages:

1. **Memory efficiency** - Careful allocation and deallocation of resources
2. **Parallelization** - OpenMP support for multi-threading where appropriate
3. **Cache-friendly access patterns** - Optimized memory access
4. **Reduced GIL contention** - C code releases the Python GIL during computationally intensive operations
5. **Cross-platform optimization** - Performance enhancements on all major operating systems

For large audio files, these optimizations can lead to 2-10x speedups in processing time, with the best performance on multi-core systems with OpenMP support. 