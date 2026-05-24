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
- The image installs from the repo source (`.[ui,export,tts,kokoro]`), so it does
  not depend on PyPI.
- It bundles both TTS families (Piper and Kokoro) and bakes in their voice files,
  so multi-speaker training and the cross-speaker and cross-TTS evaluations work
  with no extra downloads. That makes the image a few GB; for a lean build, drop
  `,kokoro` from the install line and the download line in the `Dockerfile`.
