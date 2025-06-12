FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Fix the missing stdbool.h include in vad.c
RUN sed -i '3a #include <stdbool.h>' whispr/c_ext/src/vad.c

# Install the package in development mode
RUN pip install -e .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Build C extensions
RUN cd whispr/c_ext && chmod +x build_extensions.sh && ./build_extensions.sh

# Create output directory
RUN mkdir -p /app/output

# Default command: run the processing pipeline on the provided audio file
ENTRYPOINT ["python", "-m", "whispr.pipeline"]
CMD ["--help"] 