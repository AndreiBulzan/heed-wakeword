# Heed Wake Word

[![PyPI](https://img.shields.io/pypi/v/heed-wakeword.svg)](https://pypi.org/project/heed-wakeword/) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/AndreiBulzan/heed-wakeword/blob/main/LICENSE) [![CI](https://github.com/AndreiBulzan/heed-wakeword/actions/workflows/ci.yml/badge.svg)](https://github.com/AndreiBulzan/heed-wakeword/actions/workflows/ci.yml) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndreiBulzan/heed-wakeword/blob/main/notebooks/heed_train_colab.ipynb) [![Docker](https://img.shields.io/badge/Docker-self--host-2496ED?logo=docker&logoColor=white)](https://github.com/AndreiBulzan/heed-wakeword/blob/main/docs/docker.md)

Train your own wake word in seconds, or grab a ready-made one, then run it
fully on-device. The model is a 40 to 235 KB file that runs in Python, in the
browser, and on iOS and Android. Everything runs locally, so the audio never
leaves the device and there are no usage fees.

Heed is Apache-2.0 licensed, so commercial and closed-source use are fine, with no
copyleft.

Try it with no install. Train in Colab on a free GPU, either the
[quick generic trainer](https://colab.research.google.com/github/AndreiBulzan/heed-wakeword/blob/main/notebooks/heed_train_colab.ipynb)
(type a phrase, done) or the
[train-on-your-own-voice notebook](https://colab.research.google.com/github/AndreiBulzan/heed-wakeword/blob/main/notebooks/heed_train_your_voice_colab.ipynb)
(record or upload a few clips, then test it in the notebook). Or try the live
[browser demo](https://heed.solenvo.com), which runs entirely client-side (source:
[examples/inference_browser](https://github.com/AndreiBulzan/heed-wakeword/tree/main/examples/inference_browser)).

## Two ways to use it

1. **Train a custom word.** Record a phrase a few times, or let TTS synthesize
   it across hundreds of voices, then train on CPU or GPU in seconds and export.
2. **Use a pretrained word.** The mobile demo bundles four example words (hey
   doc, activate x, hey jarvis, hey scout) and an open "custom" slot for a model
   you train and push from the studio. hey doc and activate x are the solid
   ones. hey jarvis and hey scout are quick placeholders that show off live
   multi-word switching. Slightly better pretrained defaults are planned.

Both paths produce the same artifact. You get an ONNX or TFLite model plus a
`wake.json` preprocessing contract, and it runs the same way on every platform.

## Where Heed fits

Tools like Picovoice (Porcupine), openWakeWord, and LiveKit are the established
options today, and they are all good. Heed is for a specific gap: a fully
permissive (Apache-2.0), train-your-own wake word that also runs client-side in
the browser.

In practice that means you train a custom word in seconds from the studio or the
CLI, with multi-speaker TTS and a cross-speaker evaluation so the model works for
people other than you. The result is a sub-250 KB model that runs the same way in
Python, in the browser, and on iOS and Android, as ONNX (float32 or int8) or
TFLite. You can self-host the studio in Docker or train in Colab with no setup at
all, and commercial and closed-source use carry no per-call fees.

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
  piper-tts, kokoro-onnx, flask, and onnxruntime, all under MIT, BSD, or
  Apache-2.0. The models you train are yours to ship.

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

Run the studio in a container, no local Python setup. Pull the prebuilt image:

```bash
docker run --rm -p 7777:7777 -v "$PWD/workspace:/workspace" ghcr.io/andreibulzan/heed:latest
```

Then open http://127.0.0.1:7777. Or build from source with `docker compose up`.
Recordings and trained models persist in `./workspace`, and the image bundles the
TTS voices so training works out of the box. See the
[Docker guide](https://github.com/AndreiBulzan/heed-wakeword/blob/main/docs/docker.md).

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

`<name>` is a project you pick with `heed init`; that folder holds your clips, the
trained model, and the export. A full run is just:

```
heed init myword --phrase "hey scout"
heed train myword --tts-pos 400 --kokoro-pos 300 --model-size medium
heed export myword
```

Full command list:

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

Deeper guides for each tool live in [`docs/`](https://github.com/AndreiBulzan/heed-wakeword/tree/main/docs): the
[studio UI](https://github.com/AndreiBulzan/heed-wakeword/blob/main/docs/studio.md),
[export and deploy](https://github.com/AndreiBulzan/heed-wakeword/blob/main/docs/export-and-deploy.md),
[Colab](https://github.com/AndreiBulzan/heed-wakeword/blob/main/docs/colab.md),
[Docker](https://github.com/AndreiBulzan/heed-wakeword/blob/main/docs/docker.md),
[mobile](https://github.com/AndreiBulzan/heed-wakeword/blob/main/docs/mobile.md), and
[browser and JS](https://github.com/AndreiBulzan/heed-wakeword/blob/main/docs/browser-and-js.md).

## Roadmap

Everything below works today: custom training from the studio or the CLI, on GPU
or CPU; multi-speaker TTS augmentation and a cross-speaker evaluation; ONNX and
TFLite export with verified numerical equivalence; inference in the browser and on
iOS and Android, with live multi-word switching; a zero-install Colab trainer; a
static client-side browser demo; and a Docker image for the studio.

A few directions are interesting for later: a curated pack of speaker-independent
phrases, more reference preprocessing ports, folding the preprocessing into the
model graph so raw audio goes straight in, or embedded targets like TFLite-Micro.
None of these are promised. This is a v0.1, and what runs today is the real scope.

## License

Apache-2.0. You can use Heed commercially, in closed-source products, with no
obligation to open your own code. Keep the license and NOTICE file. The license
includes a patent grant. Every dependency is MIT, BSD, or Apache-2.0.
