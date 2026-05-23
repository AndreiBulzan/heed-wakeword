# Changelog

All notable changes to heed are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] (unreleased)

First public release.

### Training

- Custom wake-word training from 8 to 30 user recordings plus a handful of
  distractors.
- Three model sizes: small (about 10K params, 41 KB), medium (about 27K,
  108 KB), large (about 60K, 235 KB). Model inference runs in 1 to 15 ms on a
  phone CPU.
- 16 kHz audio, 40-bin log-mel features, a depthwise-separable 1D CNN,
  focal-loss training, and a threshold calibrated to the user's voice.
- An energy gate skips the model during silence, which saves a lot of compute
  for always-on listening.
- Trains on GPU when present, otherwise on CPU.

### Preprocessing (streaming, on-device)

- Causal high-pass at 100 Hz (8th-order Butterworth) plus 50/60 Hz mains
  notches, applied the same way at train and inference time.
- A 25 ms Hann window, a 512-point FFT (a power of two, so the transform stays
  a fast radix-2 in any language), 40 mel bins, log-mel, then cepstral mean
  normalization for mic invariance.
- Runs incrementally. The high-pass filters each incoming chunk once with
  retained state, and the STFT recomputes only the frames that new audio
  touched.
- The Python (`heed/audio.py`) and JS (`examples/*/preprocessing.js`)
  implementations agree bit-for-bit, checked in CI by `verify-preprocessing.mjs`.

### Augmentation and robustness

- Parametric room-impulse-response pool, a parametric noise pool (white, pink,
  brown, hum, fan, babble), SpecAugment, a VTLP-style speaker warp, and gain
  jitter.
- Random-position positives so the phrase is learned at every alignment in the
  streaming buffer.
- Silence-class negatives (zeros, ambient, non-babble noise) and
  partial-utterance negatives that force a full-phrase match.
- A speaker-prototype regularizer that discourages sensitivity to the trainer's
  own voice.

### TTS augmentation

- Piper-TTS (en_US-libritts_r-medium, 904 speakers) for cross-speaker
  generalization.
- Kokoro-onnx (27 voices) as a second TTS family, sampled round-robin, for
  cross-family generalization.
- Per-engine spectral envelope matching pulls TTS audio toward the user's mic
  spectrum.
- Phonetic-neighbor distractors and a per-project TTS cache.

### Evaluation

- A self-test on the user's own positives and negatives, the direct measure of
  "does it trigger when I say it".
- A cross-speaker held-out test (Piper voices the model never saw) and a
  cross-TTS held-out test (Kokoro voices).
- A synthetic `heed smoke` self-test, judged on AUC so it is deterministic and
  CI-safe.

### Deployment

- `heed export` produces ONNX float32, ONNX INT8, and TFLite, plus a
  `wake.json` preprocessing contract and a deployment README, with numerical
  verification against PyTorch (bit-exact for fp32, a small delta for int8).
  Conv and ReLU are fused at export for lower CPU latency.
- Python inference via onnxruntime.
- A browser demo (`examples/inference_browser/`) that runs fully client-side
  with onnxruntime-web, so it is static-hostable with no server.
- React Native demos for iOS and Android (`examples/inference_react_native/`)
  with ONNX fp32/INT8 and TFLite, and live switching between words and runtimes.
- The `wake.json` contract plus the Python and JS references let any platform
  (Flutter, native Swift, native Kotlin) reproduce the preprocessing.

### Tooling

- CLI: `heed init / record / train / test / listen / eval / export / smoke /
  doctor / cross-tts-test`.
- `heed download-tts` and `heed download-kokoro` for the one-time voice
  downloads.
- `heed ui`, a Flask studio for recording, training, testing, exporting, and
  pushing a model to the mobile demo.

### Known limitations

- A curated pretrained pack is not bundled yet. v0.1 ships "hey doc" and
  "activate x" as example models, and everyone else trains their own phrase.
- Embedded and Cortex-M deployment is plausible through INT8 ONNX or
  TFLite-Micro but is not validated yet.
- A Colab trainer and a hosted browser demo are planned.
