# Export and deploy

A trained model is a tiny neural net that takes log-mel features and returns one
number (a logit). To run it anywhere you need two things: the model file, and the
`wake.json` contract that says how to turn audio into the features the model
expects. Heed exports both, and every runtime reproduces the same preprocessing.

## Export

From the CLI:

```bash
heed export my_phrase            # writes into my_phrase/export/
```

Or press Export in the studio. Either way you get:

| File | What it is |
|---|---|
| `wake.onnx` | float32 ONNX model, graph-optimized (Conv+ReLU fused). About 40 to 235 KB depending on model size. |
| `wake.int8.onnx` | INT8-quantized ONNX, roughly 40 percent of the float32 size. Lower power on NPUs, which run INT8 natively. |
| `wake.tflite` | TFLite (LiteRT) model, for the Android NNAPI and iOS Core ML delegate paths. Written only if `litert-torch` is installed. |
| `wake.json` | The preprocessing contract: phrase, threshold, mel parameters, the filter and CMN steps, the trigger logic, and the energy gate. |
| `README.md` | A deployment readme generated for that specific model, with copy-paste snippets. |

The export step verifies that the ONNX output matches the source PyTorch model
within a tight tolerance and refuses to ship a model that fails.

## Which file to use

- **Python, browser, or quick start anywhere**: `wake.onnx` with an ONNX runtime.
- **Android NPU / iOS Neural Engine, lowest power**: `wake.tflite` with the NNAPI
  or Core ML delegate.
- **Smallest download**: `wake.int8.onnx`. On phone NPUs it is also lower power.
  On desktop x86 it can be slightly slower than float32, so prefer float32 there.

## The preprocessing chain

The model consumes log-mel features, not raw audio. Apply these four steps in
order; `wake.json` carries every constant.

1. High-pass filter at 100 Hz (8th-order Butterworth, causal) plus 50 and 60 Hz
   notches. Streams chunk by chunk with retained filter state.
2. Peak normalize (optional at inference; CMN below makes the model invariant to
   constant scaling).
3. Log-mel spectrogram: 25 ms (400-sample) Hann window, FFT size 512, hop 160, 40
   mel bins, then log of the power.
4. CMN: subtract the per-clip mean across time per mel bin. Required. A model
   trained with CMN is wildly inaccurate without it.

Reference implementations that agree bit-for-bit live in `heed/audio.py` (Python)
and `examples/inference_browser/preprocessing.js` (JavaScript). The JS file is the
template for a Swift, Kotlin, or C port, about 250 lines.

## Run it: Python

```python
import json, numpy as np, onnxruntime as ort
from heed.audio import load_wav, prepare_clip, log_mel

meta = json.load(open("export/wake.json"))
sess = ort.InferenceSession("export/wake.onnx")

audio = load_wav("test.wav")          # any sample rate; resampled to 16 kHz
clip  = prepare_clip(audio)           # HPF, normalize, trim, center
mel   = log_mel(clip).numpy()         # log-mel + CMN, shape (1, 40, 101)
logit = sess.run(None, {"mel": mel})[0][0]
prob  = 1.0 / (1.0 + np.exp(-logit))
print("triggered" if prob > meta["threshold"] else "no", f"({prob:.3f})")
```

## Run it: browser

Copy `wake.onnx` and `wake.json` into `examples/inference_browser/` and serve the
folder. It loads ONNX Runtime Web, runs the same preprocessing in JS, and shows a
live probability bar. It is fully client-side. See
[Browser and JavaScript](browser-and-js.md).

## Run it: mobile

Copy the four files into a slot in `examples/inference_react_native/assets/`
(or use the studio's Send to mobile button). The app runs ONNX float32, ONNX
int8, or TFLite, with the platform delegate available for NPU acceleration. See
[Mobile](mobile.md).

## Run it: streaming, in any language

For always-on detection, run preprocessing every 100 ms on the last 1 second of
audio, feed the mel features to the model, apply a sigmoid, and trigger when the
probability stays above the threshold for `consecutive_frames` frames in a row
(then hold off for `refractory_seconds`). Use the `energy_gate` block in
`wake.json` to skip the model during silence, the single biggest power saving on
an always-on device. The browser reference in `examples/inference_browser/`
implements exactly this and doubles as a porting template.
