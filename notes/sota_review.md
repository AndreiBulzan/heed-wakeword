# SOTA review - custom wake word / few-shot KWS / tiny KWS

Compiled 2026-05-21 to ground design v2.

## TL;DR

- The "frozen feature extractor + small per-word classifier" pattern is the
  default for both commercial and OSS custom wake-word systems.
- Phoneme-aware few-shot KWS is an established research thread (PhonMatchNet,
  iPhonMatchNet, prototypical KWS, query-by-example CTC).
- Gradient reversal has been applied to KWS - but for real-vs-synthetic data,
  not speaker invariance (Park et al. 2024).
- Productized **on-device, audio-from-developer, speaker-generalizing**
  personalization in <60 s CPU does NOT appear to exist openly. This is the
  real gap.

## OSS / commercial landscape

### Porcupine (Picovoice) - commercial leader
- "Type-to-train": developer types phrase → cloud trains → .ppn → on-device.
- ~15-20 KB RAM, <1 MB runtime, fully on-device inference.
- Closed source, tier-priced. 2025: ~250K custom wake words deployed.
- Add-on "Personalized Wake Word": speaker-locked enrollment <15 s on-device
  - different product than cross-speaker custom wake words.
- **Beating them requires**: open source, or audio-trained (language-agnostic),
  or no cloud step in setup.

### OpenWakeWord (Scripka) - OSS leader
- Pipeline: `Audio → Mel → Embedding Model → Wake Word Model`.
- Embedding = frozen Google speech embedding (arXiv 2002.01322, Apache-2.0).
- Per-wake-word classifier: 2-layer FC or LSTM.
- Training: ~13K positives via TTS (Piper / Kokoro) + noise + reverb;
  requires GPU, hours per word.
- Output: ~200 KB ONNX/TFLite.
- **Frozen Google embedding is candidate backbone**; training pipeline is the
  weak point.

### microWakeWord (Espressif / ESPHome)
- INT8 TFLite, 100-300 KB, ESP32-S3/P4 (dilated conv) or C3/C5/C6
  (depthwise-separable).
- 30 ms windows, MFCC.
- Validates the 100-300 KB quantized envelope on cheap MCUs.

### Sensory + Cobra VAD
- Sensory: proprietary ultra-low-power wake word.
- Cobra VAD: ~30 KB, mature.
- VAD frontend is solved - reuse Silero or Cobra, don't build.

### EfficientWord-Net
- FaceNet-style Siamese, 4-6 samples, ~95.4 % accuracy.
- Validates few-shot Siamese; accuracy ceiling suggests not production-grade
  alone for low-FAR deployments.

## Research thread: phoneme-aware few-shot KWS

| Work | Year | Key idea | Notes |
|---|---:|---|---|
| **PhonMatchNet** (Lee et al.) | 2023 | Two-stream encoder + self-attention + phoneme-level detection loss | Closest published analogue to our backbone idea - zero-shot user-defined KWS |
| **iPhonMatchNet** | 2024 | + implicit AEC | 95 % MAE reduction at +0.13 % model size |
| **Few-Shot KWS w/ ProtoNets** (Parnami & Lee) | 2020 | Prototypical Networks | Foundation of EfficientWord-Net |
| **DONUT** (Lugosch) | 2018 | CTC posteriorgram QbyE | Phonetic posteriorgram matching, OOV-safe |
| **Few-Shot KWS from Mixed Speech** (Yuan) | 2024 | EfficientNet-B0 + Mix-Training k-hot | Speaker variation not the focus |
| **MT-HuBERT** | 2025 | Self-supervised mix-training on HuBERT | Strong but HuBERT-scale, not edge |

## Research thread: adversarial KWS

### Park et al. 2024 - "Adversarial training of KWS to minimize TTS data overfitting"
- Gradient reversal on **real-vs-synthetic** discriminator.
- SVDF backbone, 320 K params.
- 12 % relative FRR improvement with real positives + adversarial; 6-8 %
  with TTS-only positives.
- **Implication**: adversarial real/synth is published; adversarial speaker
  invariance for custom-KWS backbone is not. Both losses are complementary.

### Speaker-adversarial training in ASR / SV
- Domain-adversarial speaker removal established (Tjandra 2022, A-DA framework,
  flexible gradient reversal at chosen layers).
- Technique is mature in other audio tasks; novelty for us is applying it
  specifically to a custom-KWS phonetic backbone designed to be frozen and
  reused across user-supplied wake words.

## Backbone architecture options

| Family | Params | GSC v2 acc | Streaming | Notes |
|---|---:|---:|---|---|
| BC-ResNet-1 | 9.2 K | 96.6 % | yes | Tiniest, broadcasted residual |
| MatchboxNet | 77-140 K | 97.3-97.6 % | yes | 1D time-channel-separable conv |
| WakeNet9 (Espressif) | ~100 K | n/a | yes | Dilated conv, ESP32 native |
| Google embedding (OpenWakeWord) | ? | n/a | yes | Frozen Apache-2.0 TFHub |
| EfficientNet-B0 | ~5 M | strong | partial | Too big for target |
| HuBERT / Wav2Vec2 | ~94 M | best on FS | no | Research only |

**Choice**: BC-ResNet-1 for backbone (smallest validated). MatchboxNet as
fallback if speaker-adversarial losses need more capacity.

## TTS augmentation tooling

- **Piper-sample-generator** - slerp-weight speaker embedding interpolation,
  LibriTTS 904 speakers. Already standard in OpenWakeWord / microWakeWord.
- **Kokoro** - 67 voices, used in some OpenWakeWord trainers.
- **XTTS-v2 / F5-TTS** - higher quality, heavier; potential follow-up.

**Choice**: piper-sample-generator with slerp-weight interpolation.

## Real gaps we'd fill

1. Audio-trained (not text) custom wake word with cross-speaker generalization.
2. On-device personalization in <60 s CPU (no cloud, no GPU).
3. Structural speaker invariance via adversarial pretraining of the phonetic
   backbone for custom KWS.
4. Speaker-prototype regularizer from owner's own distractor recordings -
   appears unexplored.
5. Fully OSS / local / no telemetry as positioning.

## Reuse vs build

| Component | Source | Reuse / build |
|---|---|---|
| Mel front-end | torch / librosa | reuse |
| VAD gate | Silero VAD | reuse |
| Backbone arch | BC-ResNet-1 | reuse, retrain weights |
| Pretraining data | LibriSpeech + CommonVoice + VoxCeleb | reuse |
| Phoneme labels | Montreal Forced Aligner | reuse |
| Speaker labels | VoxCeleb metadata | reuse |
| Gradient reversal layer | standard PyTorch | reuse |
| Speaker-prototype regularizer | ours | **build** |
| TTS augmentation | piper-sample-generator | reuse |
| Per-user head training | ours | **build** |
| ONNX export | torch.onnx | reuse |

The genuinely new code is small: backbone retraining script with our loss
combination, the regularizer term, the personalization CLI. Most heavy
infrastructure (TTS, datasets, alignment) is off-the-shelf.

## Honest novelty self-grade

- Architecture pattern (frozen backbone + tiny head): **not novel** - standard.
- Phoneme-aware backbone for custom KWS: **not novel** - PhonMatchNet.
- Gradient reversal for KWS: **not novel** - Park et al. (different target).
- Speaker-adversarial *frozen-and-reused* phonetic backbone for custom KWS:
  **product-novel**, marginally research-novel.
- Speaker-prototype regularizer from owner's distractor pool: **appears novel**.
- On-device <60 s CPU personalization, audio-trained, fully OSS: **product-novel**,
  no direct competitor.

The pitch is engineering + product, not research breakthrough. The
combination fills a real, named gap (also flagged as a gap by the SF-KWS
review paper arxiv:2506.11169).
