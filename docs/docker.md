# Self-host with Docker

Run the [studio](studio.md) in a container, with no local Python setup. The
browser still captures the microphone, so recording works from any machine that
can open the page.

## Quick start

```bash
docker compose up        # builds the image, then serves http://127.0.0.1:7777
```

Recordings, trained models, and exports persist in `./workspace` on the host, so
they survive restarts.

## Or build and run by hand

```bash
docker build -t heed .
docker run --rm -p 7777:7777 -v "$PWD/workspace:/workspace" heed
```

## Notes

- The image is CPU-only. Training a tiny model on CPU is fine, just a little
  slower. For GPU training, run heed natively with a CUDA build of torch instead.
- The image installs `heed-wakeword[ui,export]` from the repo source, so it does
  not depend on PyPI.
- To bake in multi-speaker TTS augmentation, uncomment the TTS line in the
  `Dockerfile`, rebuild, and run `heed download-tts` inside the container once.
