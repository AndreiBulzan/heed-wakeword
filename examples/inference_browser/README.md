# Heed wake word, browser inference reference

A working in-browser wake-word detector. It doubles as the reference
implementation of the preprocessing chain for mobile ports (Swift, Kotlin, C).
All audio stays on-device.

This folder ships with a working "hey doc" model (`wake.onnx` + `wake.json`), so
you can serve it and try it immediately. Replace those two files to run your own.

## Quick start

1. Serve this directory from a local HTTP server (browser fetch and Worklets
   require an origin, so opening the file directly will not work):

   ```bash
   cd code/examples/inference_browser
   python -m http.server 8000
   ```

2. Open `http://localhost:8000/` in any modern browser (WebAssembly plus
   AudioWorklet). Click Start listening, allow the mic, and say "hey doc".

The page shows the live probability bar, the trigger threshold marker, and a log
of every trigger event with timestamp and scores.

To use your own word, train and export a model, then copy the files in:

```bash
heed init mywake --phrase "hey andre"
heed record mywake --kind positive --count 10
heed record mywake --kind negative --count 10
heed train  mywake --tts-pos 400
heed export mywake
cp mywake/export/wake.onnx mywake/export/wake.json .
```

## Deploy as a hosted demo (Vercel or any static host)

The page is fully client-side: ONNX Runtime Web loads from a CDN, and the model
loads from a relative path. There is no server and no build step, so you can host
this folder on any static host (Vercel, Netlify, GitHub Pages, your own domain).
HTTPS is required for microphone access, which all of those provide.

A `vercel.json` is included. To deploy on Vercel:

```bash
npm i -g vercel
cd code/examples/inference_browser
vercel deploy --prod
```

Or in the Vercel dashboard, import the repo and set the project Root Directory to
`examples/inference_browser`. There is no build command and no server cost; it
ships as static files.

## Files

| File | What it does |
|---|---|
| `index.html` | UI, mic capture, model load, visualization |
| `audio-worklet.js` | Off-main-thread audio buffering (~100 ms chunks) |
| `preprocessing.js` | Reference preprocessing chain: HPF (state-preserving biquad cascade), STFT, mel, log, CMN |
| `wakeword.js` | Combines preprocessor, ONNX session, and trigger logic (EMA, hysteresis, refractory) |
| `filter-coeffs.js` | Precomputed IIR coefficients and mel filterbank (generated from `scipy.signal`, hardcoded here) |
| `wake.onnx` | The bundled "hey doc" model (replace with your own) |
| `wake.json` | Metadata sidecar (preprocessing contract) |
| `vercel.json` | Static hosting config for the Vercel deploy above |

## Streaming preprocessing notes

`preprocessing.js` is a port of `heed/audio.py`'s `log_mel(highpass_filter(...))`.
Differences and trade-offs that any mobile port needs to know:

1. **HPF state carries across chunks.** The `BiquadCascade` class stores `s1` and
   `s2` per biquad section between calls. This makes the filter behavior on
   chunked input equivalent to filtering a continuous stream, with no edge
   transients at chunk boundaries. The Python reference uses `sosfiltfilt`
   (zero-phase forward-backward) on the full 1-second buffer each step; the JS
   version uses causal `sosfilt` with state for streaming. These produce slightly
   different low-mel-bin magnitudes (a factor of 2 in dB rolloff), absorbed by CMN
   to within numerical noise. Tested in `verify-preprocessing.mjs`.

2. **Peak normalization is omitted.** It is mathematically a no-op under CMN:
   scaling audio by `k` adds `2*log(k)` to every log-mel bin uniformly across
   frames, and CMN subtracts the per-bin mean which absorbs it. Saves a buffer
   scan per step.

3. **STFT uses a reference DFT (O(N^2)).** Clean and dependency-free, fast enough
   for 10 frames per second on any laptop or modern phone (about 80K mul-adds per
   frame). For production on slower hardware, swap in a real-input FFT library
   (`fft.js`, `kissfft.js`); only `powerSpectrum()` in `preprocessing.js` needs to
   change.

4. **Mel filterbank is sparse.** About 90 percent of the 40 by 201 weights are
   zero, so we ship a sparse representation in `filter-coeffs.js`, roughly 3 to 5
   times faster than a dense matmul.

5. **Energy gate runs before everything.** `wakeword.js` checks RMS and voice-band
   fraction on the raw chunk and short-circuits when the gate fails. This is the
   single biggest power saver on always-on devices: most of the time there is no
   speech and the whole pipeline is skipped.

## Mobile deployment template

The preprocessing chain in `preprocessing.js` is intentionally close to what you
would write in Swift, Kotlin, or C. To port:

| JS construct | Swift | Kotlin | C |
|---|---|---|---|
| `Float32Array` | `[Float]` / `UnsafeMutablePointer<Float>` | `FloatArray` | `float[]` |
| `BiquadCascade` (Direct Form II) | `Accelerate.vDSP_biquad` | `org.tensorflow.lite.support.audio` | handwritten loop |
| `powerSpectrum()` DFT loop | `vDSP_DFT_zrop_CreateSetup` | `nayuki` FFT or kissfft port | kissfft / pocketfft |
| Sparse mel matmul | same | same | same |
| `Math.log()` | `simd.log` | `Math.log` | `logf` |
| ONNX session | `ORTSession` | `OrtSession` | `OrtRun` |

Or skip ONNX and use `wake.tflite` (also produced by `heed export`) with the
TFLite NNAPI delegate (Android) or Core ML delegate (iOS) for native NPU
acceleration. The preprocessing chain is identical regardless of model format.

## Verifying the JS preprocessing matches Python

```bash
# from the repo root, with heed installed:
node examples/inference_browser/verify-preprocessing.mjs
```

This generates a known audio signal, runs both the Python reference
(`heed/audio.py`) and the JS preprocessor on it, and asserts the log-mel outputs
match within tolerance. If you change the JS code, run this before opening a PR.
