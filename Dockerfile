FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy source code
COPY . .

# Build C extensions
RUN cd whispr/c_ext && chmod +x build_extensions.sh && ./build_extensions.sh

# Create output directory
RUN mkdir -p /app/output

# Default command: run the processing pipeline on the provided audio file
ENTRYPOINT ["python", "whispr/pipeline.py"]
CMD ["--help"] 