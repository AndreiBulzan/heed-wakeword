# Heed wake word, React Native / Expo demo

Sample app that runs on-device wake-word detection on Android and iOS. It uses
Expo with a development build, because Expo Go cannot load native modules like
`onnxruntime-react-native`, so we ship a full dev build.

The preprocessing JS is identical to `examples/inference_browser/`: the same
HPF, STFT, mel, and CMN code, running in the Hermes engine. Inference is ONNX
via `onnxruntime-react-native`, which picks the best execution provider per
platform (CPU, NNAPI on Android, Core ML on iOS when the delegate is enabled).

The app ships five word slots (hey doc, activate x, hey jarvis, hey scout, and a
custom slot) with live word switching and on-device runtime switching between
ONNX fp32, ONNX int8, and TFLite.

## Prerequisites

- Node 18+ and `npm` (or `pnpm` / `yarn`)
- An Android device plus a USB cable (Android testing works from any OS,
  including Windows)
- For iOS: an EAS account (`npm install -g eas-cli`, then `eas login`). No Mac
  required, iOS dev builds happen in the EAS cloud.

## Step 1: install dependencies

```bash
cd examples/inference_react_native
npm install
```

## Step 2: get a model into a slot

The repo already ships five working slots in `assets/`, so you can build and run
without training anything. To add your own word, the easiest path is the studio:
run `heed ui`, train a model, and press "Send to mobile", which exports and
copies the files into a slot for you (slot 5, the custom slot, by default).

To do it by hand, train and export with the CLI, then copy the four output files
into a slot. For example, to replace slot 4 (the custom slot):

```bash
heed init my_phrase --phrase "hey computer"
heed train my_phrase --tts-pos 400 --kokoro-pos 200
heed export my_phrase
cp my_phrase/export/wake.onnx       assets/slot4.onnx
cp my_phrase/export/wake.int8.onnx  assets/slot4.int8.onnx
cp my_phrase/export/wake.tflite     assets/slot4.tflite
cp my_phrase/export/wake.json       assets/slot4.json
```

The app bundles whatever is in `assets/` at build time, so rebuild after
swapping a model.

## Step 3: build and run on Android

Connect your Android device over USB with USB debugging enabled, then:

```bash
npx expo run:android
```

The first build takes about 5 to 10 minutes (it downloads Gradle deps and builds
native modules). Later rebuilds are much faster. The app installs and opens
automatically. Tap Start, grant the mic permission, and say a wake phrase.

## Step 4: build and run on iOS (no Mac required)

```bash
# One-time: link the project to your own Expo account
eas init

# Build a development client in the cloud (about 10 to 15 min)
eas build --profile development --platform ios
```

When the build finishes, EAS gives you a QR code and install link. Open it on
the iOS device. You need to be in the device's provisioning profile, or use the
internal distribution path with a personal Apple Developer account. For
TestFlight builds use `--profile preview` instead.

Note: this repo has no `extra.eas.projectId` baked in, so `eas init` creates a
fresh project under your account. That is the intended flow for a clone.

## How it is wired

```
LiveAudioStream (16 kHz, int16 PCM, ~100ms chunks via "data" event)
        |
        v  decodePcm16 -> Float32Array
StreamingPreprocessor (HPF + STFT + mel + log + CMN, same code as browser)
        |
        v  (1, 40, 101) log-mel
ONNX session (slotN.onnx, loaded from a bundled asset)
        |
        v  logit
WakeWordDetector (sigmoid + EMA + hysteresis + refractory)
        |
        v
UI state (probability bar, trigger log, latency stat)
```

Audio enters and leaves the app entirely on-device. There is no network
component anywhere in the chain.

## Performance

On a recent flagship Android (Pixel 7 / Galaxy S22 class):
- median inference latency: about 3 to 6 ms per 100 ms audio chunk
- effective CPU usage: about 0.5 to 1 percent of one core
- memory: under 30 MB resident, most of it ONNX runtime overhead

On a low-end phone (about the $150 class):
- median inference latency: about 10 to 20 ms per 100 ms chunk
- still well within the 100 ms budget, comfortable real-time

To drop CPU further for always-on use, enable the NNAPI execution provider on
Android or the Core ML execution provider on iOS (see the next section).

## NPU and hardware acceleration

`onnxruntime-react-native` ships with NNAPI (Android) and Core ML (iOS)
delegates. To enable them:

```js
const session = await InferenceSession.create(modelPath, {
  executionProviders: Platform.OS === "ios"
    ? ["coreml", "cpu"]
    : ["nnapi", "cpu"],
});
```

The delegate falls back to CPU silently on unsupported ops, so this is safe to
enable unconditionally. Expect roughly 5 to 10 times lower power consumption for
always-on use when the delegate is engaged.

We left this off by default because some older devices have quirks with NNAPI on
specific ONNX ops, and the CPU path is the most reliable starting point. Switch
to a delegate once you have validated on your target devices.

## Troubleshooting

- "Cannot find module" for a slot asset: you removed a model file from
  `assets/`. The five slot sets ship with the repo; restore them or rebuild.
- `metro` complains about an unknown extension: `metro.config.js` already adds
  `.onnx` and `.tflite` to `assetExts`. If you built from a stale cache, run
  `npx expo start --clear`.
- Android build fails with `RECORD_AUDIO` errors: `app.json` already declares the
  permission. If you forked and changed the package id, run
  `npx expo prebuild --clean`.
- iOS build fails at EAS with a code-signing error: run `eas credentials` to set
  up the certificate, then rebuild. Expo's docs walk first-time iOS publishers
  through this.
- Live audio is silent or never triggers: confirm the mic permission was granted
  (Android sometimes asks twice on first install) and that the phone is not muted
  at the system level.

## Limitations

- Expo Go cannot load this project, it requires a dev build for native modules.
  Use `expo run:android` or `eas build` as described above.
- iOS distribution outside TestFlight is more involved (provisioning profile plus
  an ad-hoc UDID list). For a public demo, plan a TestFlight external testing
  track.
- The demo defaults to the CPU execution provider for portability. For always-on
  production use, enable the platform delegate (see "NPU and hardware
  acceleration") and validate on your target devices.
