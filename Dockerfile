# Heed studio, self-hosted in a container.
#
# Build:  docker build -t heed .
# Run:    docker run --rm -p 7777:7777 -v "$PWD/workspace:/workspace" heed
#   or:   docker compose up
# Then open http://127.0.0.1:7777 and record, train, test, and export.
#
# The image is CPU-only and bundles both multi-speaker TTS families plus their
# voice files, so training and the cross-speaker / cross-TTS evaluations work out
# of the box with no extra downloads. That makes it a large image (a few GB). For
# a lean build, drop `,kokoro` from the install and remove the download line.
#
# For GPU training, run heed natively with a CUDA build of torch instead.

FROM python:3.11-slim

# libsndfile backs soundfile (wav read/write). The browser captures the
# microphone, so the server needs no other audio stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first so the heed install reuses it instead of the much larger
# default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install heed from the repo source with the studio, export, and both TTS
# families (Piper for cross-speaker, Kokoro for cross-family validation).
COPY . /app
RUN pip install --no-cache-dir ".[ui,export,tts,kokoro]"

# Bake the voices into the image so multi-speaker training and the cross-speaker
# and cross-TTS tests work immediately. Piper LibriTTS-R is about 78 MB; Kokoro
# is about 340 MB.
RUN heed download-tts && heed download-kokoro

EXPOSE 7777

# Projects, recordings, and trained models live here. Mount a host volume so
# they survive container restarts.
VOLUME ["/workspace"]

CMD ["heed", "ui", "--host", "0.0.0.0", "--port", "7777", "--workspace", "/workspace"]
