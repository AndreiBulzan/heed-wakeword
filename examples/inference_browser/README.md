# Heed wake word, browser inference reference

A working in-browser wake-word detector. It doubles as the reference
implementation of the preprocessing chain for mobile ports (Swift, Kotlin, C).
All audio stays on-device.

This folder ships with four ready-made words you can switch between in the page
(hey doc, activate x, hey jarvis, hey scout), plus a drop zone to load your own
`wake.onnx` + `wake.json` from `heed export`. Serve it and try it immediately.

## Quick start

1. Serve this directory from a local HTTP server (browser fetch and Worklets
   require an origin, so opening the file directly will not work):

   ```bash
   cd examples/inference_browser
   python -m http.server 8000
   ```

2. Open `http://localhost:8000/` in any modern browser (WebAssembly plus
   AudioWorklet). Pick a word, click Start listening, allow the mic, and say it.

The page shows the live probability bar, the trigger threshold marker, and a log
of every trigger event with timestamp and scores.

To use your own word, train and export a model, then copy the files in:

```bash
heed init mywake --phrase "hey jasper"
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
cd examples/inference_browser
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

`preprocessing.js` is a port of `heed/audio.py`'s `log_mel(highpass_filter(...))`,
and it matches the Python reference bit-for-bit (checked in CI by
`verify-preprocessing.mjs`). Things any mobile port needs to know:

1. **HPF state carries across chunks.** The `BiquadCascade` stores `s1` and `s2`
   per biquad section between calls, so filtering chunk by chunk is identical to
   filtering the continuous stream, with no edge transients at chunk boundaries.
   The Python reference (`StreamingHighpass` in `heed/audio.py`) is the same causal
   filter, which is why the two agree to about 1e-5.

2. **Peak normalization is omitted.** It is a no-op under CMN: scaling audio by
   `k` adds `2*log(k)` to every log-mel bin uniformly across frames, and CMN
   subtracts the per-bin mean which absorbs it. Saves a buffer scan per step.

3. **The STFT is a built-in radix-2 FFT.** `n_fft` is 512 (a power of two), so the
   transform is a plain dependency-free radix-2 FFT (`fftRadix2` in
   `preprocessing.js`), no external library needed. The streaming preprocessor also
   recomputes only the frames new audio touched (about 14 of 101 per 100 ms chunk).

4. **Mel filterbank is sparse.** Most of the 40 by 257 weights are zero (257 =
   `n_fft`/2 + 1), so we ship a sparse representation in `filter-coeffs.js`, faster
   than a dense matmul.

5. **Energy gate runs before the model.** `wakeword.js` checks RMS and the
   voice-band power fraction on the high-pass-filtered 1-second window (the same
   window the model sees, matching `heed.gate`) and short-circuits when the gate
   fails. This is the single biggest power saver on always-on devices: most of the
   time there is no speech, so the whole model run is skipped.

## Mobile deployment template

The preprocessing chain in `preprocessing.js` is intentionally close to what you
would write in Swift, Kotlin, or C. To port:

| JS construct | Swift | Kotlin | C |
|---|---|---|---|
| `Float32Array` | `[Float]` / `UnsafeMutablePointer<Float>` | `FloatArray` | `float[]` |
| `BiquadCascade` (Direct Form II) | `Accelerate.vDSP_biquad` | `org.tensorflow.lite.support.audio` | handwritten loop |
| `fftRadix2()` (radix-2, n_fft 512) | `vDSP_DFT_zrop_CreateSetup` | `nayuki` FFT or kissfft port | kissfft / pocketfft |
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
