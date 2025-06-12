import os
import platform
import sys
from setuptools import setup, Extension
import numpy as np

# Initialize compiler flags
extra_compile_args = []
extra_link_args = []
define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

# Handle different operating systems
system = platform.system()

if system == "Linux":
    # Linux - use standard OpenMP flags
    extra_compile_args = ['-fopenmp', '-O3']
    extra_link_args = ['-fopenmp']
elif system == "Darwin":  # macOS
    # Check if using Apple Silicon
    if platform.machine() == "arm64":
        # Apple Silicon may need special handling
        if os.path.exists("/opt/homebrew/opt/libomp"):
            # Homebrew on Apple Silicon
            extra_compile_args = ['-Xpreprocessor', '-fopenmp', '-O3']
            extra_link_args = ['-lomp']
            os.environ["CC"] = "clang"
            os.environ["CXX"] = "clang++"
            os.environ["CFLAGS"] = "-I/opt/homebrew/opt/libomp/include"
            os.environ["LDFLAGS"] = "-L/opt/homebrew/opt/libomp/lib"
        else:
            # Without OpenMP installed, just use optimization
            extra_compile_args = ['-O3']
    else:
        # Intel Mac
        if os.path.exists("/usr/local/opt/libomp"):
            # Homebrew on Intel Mac
            extra_compile_args = ['-Xpreprocessor', '-fopenmp', '-O3']
            extra_link_args = ['-lomp']
            os.environ["CC"] = "clang"
            os.environ["CXX"] = "clang++"
            os.environ["CFLAGS"] = "-I/usr/local/opt/libomp/include"
            os.environ["LDFLAGS"] = "-L/usr/local/opt/libomp/lib"
        else:
            # Without OpenMP installed, just use optimization
            extra_compile_args = ['-O3']
elif system == "Windows":
    # Windows with MSVC
    if sys.platform == 'win32':
        extra_compile_args = ['/openmp', '/O2']
        # No need for extra link args with MSVC
    else:
        # MinGW
        extra_compile_args = ['-fopenmp', '-O3']
        extra_link_args = ['-fopenmp']
else:
    # Default to just optimization for other platforms
    extra_compile_args = ['-O3']

print(f"Building extensions for {system} platform with:")
print(f"  Compile args: {extra_compile_args}")
print(f"  Link args: {extra_link_args}")
print(f"  Define macros: {define_macros}")

# Define extensions
extensions = [
    Extension(
        'framing_c',
        sources=['src/framing.c'],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros
    ),
    Extension(
        'vad_c',
        sources=['src/vad.c'],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros
    ),
    Extension(
        'features_c',
        sources=['src/features.c'],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros
    ),
    Extension(
        'separation_c',
        sources=['src/separation.c'],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros
    )
]

setup(
    name="whispr_c_ext",
    version="0.1",
    description="C extensions for Whispr",
    ext_modules=extensions,
    setup_requires=['numpy>=1.20.0'],
    install_requires=['numpy>=1.20.0'],
) 