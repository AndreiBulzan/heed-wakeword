# Heed Wake Word

Train your own wake word in seconds, or grab a ready-made one, then run it
fully on-device. No cloud, no telemetry, no per-call fees. A wake word here is a
40 to 235 KB model that runs in Python, in the browser, and on iOS and Android.

Heed is Apache-2.0 licensed, so it is free for commercial use with no copyleft.
It is an open option where wake-word SDKs are usually paid and closed.

Try it with no install: [train your own in Colab](https://colab.research.google.com/github/AndreiBulzan/heed-wakeword/blob/main/notebooks/heed_train_colab.ipynb), or run the static [browser demo](examples/inference_browser/).

## Two ways to use it

1. **Train a custom word.** Record a phrase a few times, or let TTS synthesize
   it across hundreds of voices, then train on CPU or GPU in seconds and export.
2. **Use a pretrained word.** The mobile demo bundles four example words (hey
   doc, activate x, hey jarvis, hey fetch) and an open "custom" slot for a model
   you train and push from the studio. hey doc and activate x are the solid
   ones. hey jarvis and hey fetch are quick placeholders that show off live
   multi-word switching. A curated, speaker-independent pack is planned.

Both paths produce the same artifact. You get an ONNX or TFLite model plus a
`wake.json` preprocessing contract, and it runs the same way on every platform.

## Why Heed

| | Heed | Picovoice | openWakeWord |
|---|---|---|---|
| License | Apache-2.0, commercial OK | Paid, closed | Code Apache, weights CC-BY-NC |
| Train your own | Yes, studio plus CLI | Paid console | Yes, heavier |
| On-device, no cloud | Yes | Yes | Yes |
| Browser, client-side | Yes | No | No |
| iOS and Android demos | Yes, ONNX and TFLite | Yes | Community |

## Quickstart (about a minute)

```bash
pip install "heed-wakeword[ui]"   # base plus the browser studio
heed ui                            # opens http://127.0.0.1:7777
```

Record a few positives and negatives in the browser, press Train, then
Live-test. A GPU is optional and gets used when present. If you prefer the
terminal:

```bash
heed init my_phrase --phrase "hey computer"
heed train my_phrase                                  # quick, tuned to your voice
heed train my_phrase --tts-pos 400 --kokoro-pos 200   # cross-speaker, works for anyone
heed export my_phrase                                 # wake.onnx, wake.int8.onnx, wake.tflite, wake.json
```

The package is `heed-wakeword` on PyPI. You import it as `heed`, and the command
is `heed`.

## What you get

- **Tiny and fast.** small is about 10K params (41 KB), medium about 27K
  (108 KB), large about 60K (235 KB). INT8 is roughly 40% of that. Model
  inference takes 1 to 15 ms on a phone CPU.
- **Many runtimes, every platform.** ONNX (fp32 and INT8) and TFLite, on
  Python, the browser (onnxruntime-web), and React Native iOS and Android.
- **A streaming preprocessor we wrote ourselves.** A causal high-pass with
  50/60 Hz notches feeds a 25 ms Hann window, a 512-point FFT (a power of two,
  so it stays fast in any language), a 40-bin log-mel, and CMN. It runs
  incrementally, recomputing only the frames that new audio touched, and it
  agrees with Python bit-for-bit in JS (CI checks this). On a phone, prep is
  about 15 to 20 ms per 100 ms of audio, and an energy gate skips the model
  during silence.
- **Quality you can measure.** A cross-speaker held-out eval and a
  cross-TTS-family eval tell you whether a model works beyond the trainer's own
  voice, before you ship it.
- **A permissive stack.** torch, numpy, scipy, soundfile, click, with optional
  piper-tts, kokoro-onnx, flask, and onnxruntime. No CC-BY-NC weights and no
  research-only data.

## Deploy anywhere

The model consumes log-mel features, so any runtime reproduces the same
preprocessing chain. `wake.json` specifies it in full, and there are reference
implementations in Python (`heed/audio.py`) and JS
(`examples/*/preprocessing.js`) that agree bit-for-bit.

| Target | How |
|---|---|
| Python | `onnxruntime` on CPU. See `export/README.md`. |
| Browser | `onnxruntime-web` with `examples/inference_browser/`. Fully client-side and static-hostable on Vercel, Netlify, or GitHub Pages; ships a `vercel.json`. |
| iOS and Android | `examples/inference_react_native/`, with ONNX fp32/INT8 and TFLite, plus live word and runtime switching. |
| Other native (Flutter, Swift, Kotlin) | Run the ONNX or TFLite model, then port the preprocessing from the Python or JS reference (about 250 lines). |

Deployment needs none of the training dependencies. A 3 MB runtime and your
sub-250 KB model cover it.

## Install

```bash
pip install heed-wakeword              # core: train and the model
pip install "heed-wakeword[ui]"        # plus the browser studio (Flask)
pip install "heed-wakeword[tts]"       # plus piper-tts, then: heed download-tts
pip install "heed-wakeword[kokoro]"    # plus kokoro-onnx, then: heed download-kokoro
pip install "heed-wakeword[export]"    # plus onnx and onnxruntime (export, verify)
pip install "heed-wakeword[all]"       # everything
heed doctor                            # check torch, onnxruntime, and TTS
heed smoke                             # synthetic end-to-end self-test, no mic
```

## Self-host the studio (Docker)

Run the browser studio in a container, with no local Python setup:

```bash
docker compose up      # builds the image, then serves http://127.0.0.1:7777
```

Recordings and trained models persist in `./workspace`. The image is CPU-only,
which is fine for training a tiny model; for GPU training, run heed natively. See
`Dockerfile`.

## Recording good data

This is the biggest lever on quality.

- **Positives.** 8 to 30 recordings of the phrase. Vary your prosody, distance
  from the mic, and room. Variety beats raw count.
- **Negatives.** Distractor phrases in your own voice ("good morning", "the
  weather is nice") make precious hard negatives. Add similar-sounding phrases
  (for "hey doc", add "hey John") so the model learns the boundary.
- **Cross-speaker.** Turn on TTS (`--tts-pos`, `--kokoro-pos`) to synthesize the
  phrase across hundreds of voices, so the model is not tied to you. Confirm
  with the cross-speaker eval before you ship.

## CLI reference

```
heed ui              [--host 127.0.0.1] [--port 7777] [--workspace DIR]
heed init            <name> --phrase "..."
heed record          <name> --kind {positive|negative} --count N
heed download-tts / download-kokoro
heed train           <name> [--epochs N] [--tts-pos N] [--kokoro-pos N]
                            [--target-fpr X] [--model-size {small|medium|large}] ...
heed test            <name> <audio.wav>
heed listen          <name>
heed eval            <name> [--positive-dir P] [--negative-dir N]
heed cross-tts-test  <name>
heed export          <name>
heed smoke / doctor
```

Run `heed <cmd> --help` for the full options.

## Design, in one paragraph

Log-mel spectrograms (40 bins, a 25 ms window, a 10 ms hop, a 512-point FFT)
feed a small depthwise-separable 1D CNN over time, with a stride-2 stem, a few
DS-conv blocks, a global average pool, and a linear head. Training builds a
per-user set from a handful of real positives, signal-processing augmentation (a
VTLP-style speaker warp, reverb, noise, gain), and optional multi-speaker TTS,
with a speaker-prototype regularizer that discourages sensitivity to the
trainer's own voice. The high-pass is causal and state-retaining, so the exact
same filtering streams chunk by chunk on-device, and the STFT is computed
incrementally so only the frames that new audio touched get recomputed. The
threshold is calibrated to a target false-positive rate, and inference is a
sliding window with an RMS and voice-band energy gate in front of the model.
See `notes/` for the design rationale and a comparison with prior work.

## GPU and CPU

Training auto-detects CUDA and uses it when present, otherwise it runs on CPU.
The model is small, so CPU training works fine and is only a little slower.
Model inference is CPU-only by design, because the model is far too small for
GPU offload to beat the data-transfer cost. The one place a GPU pays off is TTS
synthesis during training. See the install notes for the optional
`onnxruntime-gpu` swap.

## Documentation

Deeper guides for each tool live in [`docs/`](docs/): the
[studio UI](docs/studio.md), [export and deploy](docs/export-and-deploy.md),
[Colab](docs/colab.md), [Docker](docs/docker.md), [mobile](docs/mobile.md), and
[browser and JS](docs/browser-and-js.md).

## Roadmap

Working today: custom training (studio and CLI), GPU and CPU, multi-speaker TTS,
cross-speaker eval, ONNX and TFLite export with verified equivalence, browser and
iOS and Android inference, live multi-word switching, a zero-install Colab
trainer, a static client-side browser demo, and a Docker studio image.

Ideas for later, not promises: a curated pretrained pack of generic phrases, more
reference preprocessing ports, folding the preprocessing into the model graph so
raw audio can go straight in, and embedded targets such as TFLite-Micro. Whether
any of these land depends on interest and time.

This is v0.1. The "working today" list is the honest scope.

## License

Apache-2.0. You can use Heed commercially, in closed-source products, with no
obligation to open your own code. Keep the license and NOTICE file. The license
includes a patent grant. Every dependency is MIT, BSD, or Apache-2.0.
