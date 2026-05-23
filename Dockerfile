# Heed studio, self-hosted in a container.
#
# Build:  docker build -t heed .
# Run:    docker run --rm -p 7777:7777 -v "$PWD/workspace:/workspace" heed
# Then open http://127.0.0.1:7777 and record, train, test, and export.
#
# The image is CPU-only. Training a tiny model on CPU is fine. If you want GPU
# training, run heed natively with a CUDA torch instead (see the README).

FROM python:3.11-slim

# libsndfile backs soundfile (wav read/write). That is all the system audio
# stack the studio needs; the browser captures the microphone, not the server.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only torch wheel first so the heed install reuses it instead
# of pulling the much larger default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install heed from the repo source (no PyPI round-trip needed to build).
COPY . /app
RUN pip install --no-cache-dir ".[ui,export]"

# Optional: bake in multi-speaker TTS augmentation (adds ~1 GB of voices on
# first download, not at build). Uncomment to enable, then run `heed
# download-tts` inside the container.
# RUN pip install --no-cache-dir "kokoro-onnx>=0.4" "piper-tts>=1.2"

EXPOSE 7777

# Projects, recordings, and trained models live here. Mount a host volume so
# they survive container restarts.
VOLUME ["/workspace"]

CMD ["heed", "ui", "--host", "0.0.0.0", "--port", "7777", "--workspace", "/workspace"]
