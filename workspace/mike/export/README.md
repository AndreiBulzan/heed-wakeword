# Wake-word model export

Trained for phrase: **"hey mike"**. Sigmoid output threshold: **0.682**.

## Files

- `wake.onnx`: float32 ONNX model, **108.3 KB**, 27089 parameters. Graph-optimized (Conv+ReLU fused).
- `wake.int8.onnx`: INT8-quantized ONNX, **40.0 KB**. Use for size-constrained mobile / edge.
- `wake.json`: metadata sidecar: threshold, mel params, preprocessing chain.

## Choose your runtime / format

| Target | Runtime | Model file | Notes |
|---|---|---|---|
| Desktop Python | `onnxruntime` | `wake.onnx` | Cross-platform, simplest |
| Browser | `onnxruntime-web` | `wake.onnx` | WASM + SIMD; ~5-10 ms/inference |
| Android (CPU) | `onnxruntime-android` (.aar) | `wake.onnx` | ~3 MB runtime |
| Android (NPU via NNAPI) | `tflite` with NNAPI delegate | `wake.tflite` | Dispatches to NPU/DSP when available; falls back to CPU. **Lower power.** |
| iOS (CPU) | `onnxruntime-objc` (CocoaPod) | `wake.onnx` | ~3 MB framework |
| iOS (Neural Engine via Core ML) | `tflite` with Core ML delegate | `wake.tflite` | Routes graph to Apple Neural Engine where supported; falls back to GPU / CPU. **Lower power.** |
| Embedded MCU | TFLite Micro (separate workstream) | `wake.tflite` + INT8 conversion | Cortex-M / ESP32 territory |

INT8 (`wake.int8.onnx`) is smaller on disk and **lower-power on NPUs** (which run INT8 natively).
On desktop x86 CPU it can actually be *slower* than fp32 due to quantize/dequantize overhead. That
isn't an issue on mobile NPUs / DSPs where INT8 is the native path.

## Usage (Python)

```python
import json
import numpy as np
import onnxruntime as ort
from heed.audio import load_wav, prepare_clip, log_mel

meta = json.load(open("wake.json"))
sess = ort.InferenceSession("wake.onnx")

audio = load_wav("test.wav")        # any sample rate; load_wav resamples to 16k
clip = prepare_clip(audio)          # HPF + peak_normalize + trim + center
mel = log_mel(clip).numpy()         # log-mel + CMN, shape (1, 40, 101)
logit = sess.run(None, {"mel": mel})[0][0]
prob = 1.0 / (1.0 + np.exp(-logit)) # sigmoid
print(f"prob = {prob:.3f}  triggered = {prob > meta['threshold']}")
```

## Usage (Android, Kotlin): TFLite + NNAPI delegate

```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate

val options = Interpreter.Options().apply {
    // NNAPI dispatches to NPU/DSP when the device supports it.
    // Falls back to CPU silently on unsupported devices.
    addDelegate(NnApiDelegate())
    setNumThreads(2)
}
val interpreter = Interpreter(loadModelFile(context, "wake.tflite"), options)

// Input: float32 mel features, shape (1, 40, 101). See preprocessing below
val input = Array(1) { Array(40) { FloatArray(101) } }
val output = Array(1) { FloatArray(1) }
interpreter.run(input, output)
val prob = 1.0f / (1.0f + Math.exp(-output[0][0].toDouble()))
val triggered = prob > THRESHOLD
```

## Usage (iOS, Swift): TFLite + Core ML delegate

```swift
import TensorFlowLite

let coreMLOptions = CoreMLDelegate.Options()
coreMLOptions.coreMLVersion = 3  // route to Neural Engine on supported chips
let coreMLDelegate = CoreMLDelegate(options: coreMLOptions)!

var options = Interpreter.Options()
options.threadCount = 2
let interpreter = try Interpreter(
    modelPath: Bundle.main.path(forResource: "wake", ofType: "tflite")!,
    options: options,
    delegates: [coreMLDelegate]
)
try interpreter.allocateTensors()
try interpreter.copy(melData, toInputAt: 0)
try interpreter.invoke()
let output = try interpreter.output(at: 0)
let logit = output.data.withUnsafeBytes { $0.load(as: Float32.self) }
let prob = 1.0 / (1.0 + exp(-Double(logit)))
let triggered = prob > THRESHOLD
```

## Preprocessing chain (REQUIRED, apply in this exact order)

The model expects log-mel features, NOT raw audio. Implement these four
steps in your target language; `wake.json` has every constant.

1. **High-pass filter** at 100 Hz (8th-order Butterworth, causal single-pass)
   + notch filters at 50 Hz and 60 Hz (mains hum). Critical for cross-mic
   robustness. Being causal, it streams: filter each new chunk once with
   retained biquad state (see `examples/*/preprocessing.js`).
2. **Peak normalize** to -3 dBFS (≈ 0.707 linear). Optional at inference;
   CMN (step 4) already makes log-mel invariant to constant audio scaling.
3. **Log-mel spectrogram**: STFT with a 25 ms (400-sample) Hann window,
   n_fft=512 (window zero-padded to the FFT size; a power-of-two FFT is fast
   on every runtime), hop=160, 40 mel bins, then `log(power)`.
4. **CMN**: subtract per-clip mean across time per mel bin. **CRITICAL**:
   model trained with CMN expects CMN'd input; omitting it makes the model
   wildly inaccurate.

For streaming inference (continuous wake-word detection):
- Run preprocessing every ~100 ms hop on the latest 1-second audio buffer.
- Feed mel features to the model, apply sigmoid to the logit.
- Trigger when probability exceeds threshold for `consecutive_frames` consecutive frames (default 2), then suppress for `refractory_seconds` (default 0.7s).
- Use the energy gate (`energy_gate` in wake.json) to skip preprocessing+model entirely during silence. Major power saving.

A reference streaming implementation in JavaScript ships in `examples/inference_browser/` (works in any modern browser, doubles as a deployment template for Swift/Kotlin/C).
