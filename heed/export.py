"""ONNX export for trained wake-word models.

What we export:
  * `wake.onnx`: float32 model. Input: log-mel features shape (B, n_mels, T),
    already CMN-normalized. Output: raw logit (apply sigmoid in your runtime).
  * `wake.int8.onnx` (optional): INT8-quantized variant via onnxruntime's
    dynamic quantization. ~25% the size, typically <0.01 accuracy delta on
    the user's positives/negatives at the calibrated threshold.
  * `wake.json`: metadata sidecar: phrase, threshold, sample_rate, mel
    params, HPF cutoff, etc. Everything a downstream runtime needs to
    reproduce the preprocessing chain in JS / Swift / Kotlin / C.

What we DON'T export (yet):
  * The audio preprocessing chain (HPF, peak_normalize, log_mel, CMN).
    Reasons: the IIR butterworth HPF + 50/60 Hz notches don't have clean
    ONNX equivalents, and `torch.stft` ONNX export is opset-sensitive. v1
    keeps the model export rock-solid and ships a Python reference
    pipeline that users can translate to their target language. The
    wake.json sidecar contains every constant needed.

Deployment usage (Python reference):
    import onnxruntime as ort
    import json, numpy as np
    from heed.audio import load_wav, prepare_clip, log_mel

    meta = json.load(open("export/wake.json"))
    sess = ort.InferenceSession("export/wake.onnx")
    audio = load_wav("test.wav")
    clip = prepare_clip(audio)         # HPF + normalize + trim + center
    mel = log_mel(clip).numpy()        # log-mel + CMN, shape (1, 40, 101)
    logit = sess.run(None, {"mel": mel})[0]
    prob = 1.0 / (1.0 + np.exp(-logit))
    triggered = prob > meta["threshold"]
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import HOP_SAMPLES, N_FFT, N_MELS, SAMPLE_RATE, WIN_SAMPLES, WINDOW_FRAMES
from .trainer import load_model


@dataclass
class ExportResult:
    """Summary of an export run, returned by export_to_onnx."""
    onnx_path: Path
    int8_path: Path | None
    metadata_path: Path
    n_params: int
    onnx_size_bytes: int
    int8_size_bytes: int | None
    max_abs_error_fp32: float
    max_abs_error_int8: float | None


def export_to_onnx(
    model_pt_path: str | Path,
    output_dir: str | Path,
    *,
    int8: bool = True,
    opset: int = 17,
    verify_atol: float = 1e-4,
    log_fn=print,
) -> ExportResult:
    """Export a trained heed model to ONNX (float32 + optional INT8).

    Args:
        model_pt_path: path to a trained model.pt produced by heed train.
        output_dir: directory to write wake.onnx / wake.int8.onnx / wake.json.
        int8: also produce a dynamic-quantized INT8 ONNX file.
        opset: ONNX opset version to target. 17 is the modern default and
               works with onnxruntime ≥ 1.13.
        verify_atol: max-abs tolerance when comparing ONNX vs PyTorch outputs.
                     We raise if exceeded (fail loudly rather than ship a
                     broken model).

    Returns ExportResult with paths + diagnostics. Raises on verification
    failure.
    """
    model_pt_path = Path(model_pt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_model(model_pt_path)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Dummy input: (batch=1, n_mels, frames). PyTorch's stft with center=True
    # on a 1-second audio at hop=160 produces WINDOW_FRAMES + 1 = 101 frames.
    n_mels = int(payload.get("n_mels", N_MELS))
    n_frames = int(payload.get("window_frames", WINDOW_FRAMES)) + 1
    dummy = torch.randn(1, n_mels, n_frames)

    # --- Export float32 ONNX ---
    onnx_path = output_dir / "wake.onnx"
    log_fn(f"exporting float32 ONNX → {onnx_path}")
    # dynamo=False uses the legacy TorchScript exporter which doesn't
    # require the onnxscript package. The legacy path is fine for a
    # model this simple (depthwise-separable convs + batchnorm + linear)
    # and will keep working for the foreseeable future (deprecated but
    # not removed). Sufficient and stable.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["mel"],
            output_names=["logit"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes={
                "mel":   {0: "batch"},
                "logit": {0: "batch"},
            },
            dynamo=False,
        )

    # --- Verify float32 matches PyTorch numerically ---
    try:
        import onnxruntime as ort
    except Exception as exc:  # ImportError or Windows DLL load failure
        import sys
        raise RuntimeError(
            f"onnxruntime is not usable from this Python.\n"
            f"  underlying error: {type(exc).__name__}: {exc}\n"
            f"  python:           {sys.executable}\n"
            f"\n"
            f"Likely causes:\n"
            f"  1. onnxruntime not installed in THIS conda env (you may have it\n"
            f"     in a different env where piper-tts works). Install with:\n"
            f"       {sys.executable} -m pip install onnxruntime\n"
            f"  2. On Windows: DLL load failure. Install the Microsoft Visual\n"
            f"     C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
            f"     Then `heed doctor` to see the full diagnostic."
        ) from exc

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()
    ort_out = sess.run(None, {"mel": dummy.numpy()})[0]
    err_fp32 = float(np.abs(pt_out - ort_out).max())
    log_fn(f"  float32 verify: max-abs-error = {err_fp32:.6e}")
    if err_fp32 > verify_atol:
        raise RuntimeError(
            f"ONNX float32 export verification failed: "
            f"max abs error {err_fp32:.6e} > tolerance {verify_atol:.1e}. "
            f"Refusing to ship a broken model."
        )

    onnx_size = onnx_path.stat().st_size

    # --- Optional INT8 ---
    int8_path = None
    int8_size = None
    err_int8 = None
    if int8:
        int8_path = output_dir / "wake.int8.onnx"
        log_fn(f"quantizing → {int8_path}")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                str(onnx_path),
                str(int8_path),
                weight_type=QuantType.QInt8,
            )
        except Exception as exc:
            log_fn(f"  [warn] INT8 quantization failed: {exc}")
            int8_path = None
        else:
            # Verify INT8 is close enough to float32 (looser tolerance, INT8
            # is lossy by design, but should be < ~5e-2 on a model this small).
            int8_sess = ort.InferenceSession(
                str(int8_path),
                providers=["CPUExecutionProvider"],
            )
            int8_out = int8_sess.run(None, {"mel": dummy.numpy()})[0]
            err_int8 = float(np.abs(pt_out - int8_out).max())
            int8_size = int8_path.stat().st_size
            log_fn(f"  int8 verify:    max-abs-error = {err_int8:.6e}")
            if err_int8 > 0.2:
                log_fn(f"  [warn] INT8 error is unusually high ({err_int8:.3f}). "
                       f"Test the INT8 model on real positives before using it.")

    # --- Apply ORT_ENABLE_EXTENDED graph optimization to both fp32 and int8 ---
    # Fuses Conv+ReLU into FusedConv ops. Portable (no NCHWc / hardware-specific
    # transforms). Typical gain on this model: ~20% inference latency on CPU EP,
    # free win on every downstream runtime that supports FusedConv (mainstream
    # ones do). Done AFTER quantization because the dynamic quantizer's
    # shape-inference doesn't handle FusedConv outputs.
    # BatchNorm in the source model has been folded into preceding Conv weights
    # already by torch.onnx.export's constant folding; the one BN node that
    # remains is the input-side normalization (no Conv before it to absorb
    # into), kept as-is.
    log_fn("applying ORT graph optimizations (Conv+ReLU fusion)")

    def _optimize_in_place(path: Path, label: str) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        opt_so = ort.SessionOptions()
        opt_so.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        opt_so.optimized_model_filepath = str(tmp_path)
        _ = ort.InferenceSession(
            str(path), opt_so, providers=["CPUExecutionProvider"]
        )
        path.unlink()
        tmp_path.rename(path)
        log_fn(f"  {label} optimized → {path.name} ({path.stat().st_size/1024:.1f} KB)")

    _optimize_in_place(onnx_path, "fp32")
    onnx_size = onnx_path.stat().st_size
    if int8_path is not None and int8_path.exists():
        _optimize_in_place(int8_path, "int8")
        int8_size = int8_path.stat().st_size

    # --- Re-verify both after optimization (the graph rewrite must not move
    # outputs outside the tolerance) ---
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    err_fp32_opt = float(np.abs(pt_out - sess.run(None, {"mel": dummy.numpy()})[0]).max())
    log_fn(f"  fp32 post-opt verify: max-abs-error = {err_fp32_opt:.6e}")
    if err_fp32_opt > verify_atol:
        raise RuntimeError(
            f"Post-optimization fp32 verification failed: {err_fp32_opt:.6e} > {verify_atol:.1e}"
        )
    if int8_path is not None and int8_path.exists():
        int8_sess_opt = ort.InferenceSession(
            str(int8_path), providers=["CPUExecutionProvider"]
        )
        err_int8 = float(np.abs(pt_out - int8_sess_opt.run(None, {"mel": dummy.numpy()})[0]).max())
        log_fn(f"  int8 post-opt verify: max-abs-error = {err_int8:.6e}")

    # --- Metadata sidecar (everything needed to reproduce preprocessing) ---
    metadata = _build_metadata(payload)
    metadata_path = output_dir / "wake.json"
    # Always write as UTF-8. Windows' default cp1252 can't encode unicode
    # characters like '≈' that appear in our notes/docstrings, and the
    # default Path.write_text on Windows uses the system encoding.
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log_fn(f"wrote metadata    → {metadata_path}")

    # --- Friendly README in the export dir ---
    _write_export_readme(output_dir, metadata, onnx_size, int8_size, n_params)

    log_fn("")
    log_fn(f"  float32 size:  {onnx_size/1024:6.1f} KB  ({n_params} params)")
    if int8_size:
        log_fn(f"  int8 size:     {int8_size/1024:6.1f} KB  "
               f"({100 * int8_size / onnx_size:.0f}% of float32)")
    log_fn(f"  metadata:      {metadata_path.name}")
    log_fn(f"  README:        export/README.md  (deployment instructions)")

    return ExportResult(
        onnx_path=onnx_path,
        int8_path=int8_path,
        metadata_path=metadata_path,
        n_params=n_params,
        onnx_size_bytes=onnx_size,
        int8_size_bytes=int8_size,
        max_abs_error_fp32=err_fp32,
        max_abs_error_int8=err_int8,
    )


@dataclass
class TFLiteExportResult:
    """Summary of a TFLite export."""
    tflite_path: Path
    tflite_size_bytes: int
    max_abs_error: float


def export_to_tflite(
    model_pt_path: str | Path,
    output_dir: str | Path,
    *,
    verify_atol: float = 1e-4,
    log_fn=print,
) -> TFLiteExportResult:
    """Export a trained heed model to TFLite (LiteRT) via litert-torch.

    TFLite is the preferred format for mobile NPU delegation:
        - Android: NNAPI delegate, GPU delegate, or vendor-specific (Hexagon
                   on Qualcomm, APU on MediaTek) via TFLite's delegate API.
        - iOS:     Core ML delegate routes the graph to Apple Neural Engine
                   when supported (otherwise falls back to GPU then CPU).

    Requires `pip install litert-torch`. Raises ImportError if unavailable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import litert_torch  # noqa: F401  (heavy import; defer to call site)
    except ImportError as exc:
        raise ImportError(
            "litert-torch not installed (required for TFLite export)"
        ) from exc

    model, payload = load_model(Path(model_pt_path))
    model.eval()

    n_mels = int(payload.get("n_mels", N_MELS))
    n_frames = int(payload.get("window_frames", WINDOW_FRAMES)) + 1
    dummy = torch.randn(1, n_mels, n_frames)

    log_fn(f"exporting TFLite → {output_dir / 'wake.tflite'}")
    edge = litert_torch.convert(model, (dummy,))
    tflite_path = output_dir / "wake.tflite"
    edge.export(str(tflite_path))

    # Numerical equivalence check vs PyTorch source.
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()
    tf_out = np.asarray(edge(dummy)).flatten()
    err = float(np.abs(pt_out.flatten() - tf_out).max())
    log_fn(f"  tflite verify: max-abs-error = {err:.6e}")
    if err > verify_atol:
        raise RuntimeError(
            f"TFLite export verification failed: {err:.6e} > {verify_atol:.1e}"
        )

    size = tflite_path.stat().st_size
    log_fn(f"  tflite size:  {size/1024:6.1f} KB")

    return TFLiteExportResult(
        tflite_path=tflite_path,
        tflite_size_bytes=size,
        max_abs_error=err,
    )


def _build_metadata(payload: dict) -> dict:
    """Construct the wake.json sidecar with everything a downstream runtime
    needs to reproduce preprocessing in any language."""
    return {
        # Identity
        "phrase": payload.get("phrase", ""),
        "threshold": float(payload["threshold"]),
        # Audio I/O
        "sample_rate": int(payload.get("sample_rate", SAMPLE_RATE)),
        "audio_window_samples": int(payload.get("sample_rate", SAMPLE_RATE)),
        # Mel features (must match heed.audio.log_mel exactly)
        "n_mels": int(payload.get("n_mels", N_MELS)),
        "n_fft": int(N_FFT),
        "win_length": int(WIN_SAMPLES),
        "hop_length": int(HOP_SAMPLES),
        "window_frames": int(payload.get("window_frames", WINDOW_FRAMES)),
        "mel_fmin_hz": 0.0,
        "mel_fmax_hz": float(payload.get("sample_rate", SAMPLE_RATE)) / 2.0,
        "window_type": "hann",
        # Preprocessing chain (apply IN ORDER, all required for accuracy)
        "preprocessing": [
            {"op": "highpass_filter", "cutoff_hz": 100.0, "order": 8,
             "mains_notch_hz": [50.0, 60.0],
             "note": "8th-order Butterworth HPF + 50/60 Hz notch filters. "
                     "Killing sub-100Hz rumble before mel is critical for "
                     "robustness across mic profiles."},
            {"op": "peak_normalize", "target_dbfs": -3.0,
             "note": "Scale audio so peak = 10^(-3/20) ≈ 0.707."},
            {"op": "log_mel",
             "note": "Standard log-mel with hann window. See mel params above."},
            {"op": "cmn",
             "note": "Cepstral mean normalization, subtract per-clip mean "
                     "across time per mel bin. CRITICAL: model trained with "
                     "CMN expects CMN'd input; omitting it makes the model "
                     "wildly inaccurate."},
        ],
        # Trigger logic (for streaming runtimes)
        "trigger": {
            "consecutive_frames": 2,
            "refractory_seconds": 0.7,
            "ema_alpha": 0.5,
            "note": "Trigger when raw prob > threshold for `consecutive_frames` "
                    "in a row, then suppress for `refractory_seconds`. ema_alpha "
                    "is for display smoothing only (not the trigger gate; the "
                    "EMA lags and was suppressing short, confident detections).",
        },
        # Energy gate (optional but recommended, saves compute during silence)
        "energy_gate": {
            "rms_threshold_dbfs": -55.0,
            "voice_band_min_fraction": 0.15,
            "voice_band_lo_hz": 100.0,
            "voice_band_hi_hz": 7000.0,
            "note": "Skip the model entirely when gate fails. Prevents "
                    "amplified-mic-noise false readings during silence.",
        },
        # Provenance
        "model_size": payload.get("model_size", "unknown"),
        "channels": int(payload.get("channels", 32)),
        "n_blocks": int(payload.get("n_blocks", 3)),
        "heed_export_version": 1,
    }


def _write_export_readme(output_dir: Path, metadata: dict,
                         onnx_size: int, int8_size: int | None,
                         n_params: int) -> None:
    """Write a deployment README into the export directory with code
    examples for Python, Android, iOS, and browser runtimes."""
    phrase = metadata["phrase"]
    threshold = metadata["threshold"]
    size_kb = onnx_size / 1024
    int8_size_kb = (int8_size / 1024) if int8_size else None
    int8_line = (
        f"- `wake.int8.onnx`: INT8-quantized ONNX, **{int8_size_kb:.1f} KB**. "
        f"Use for size-constrained mobile / edge.\n"
        if int8_size_kb else ""
    )
    tflite_path = output_dir / "wake.tflite"
    tflite_line = ""
    if tflite_path.exists():
        tf_kb = tflite_path.stat().st_size / 1024
        tflite_line = (
            f"- `wake.tflite`: TFLite (LiteRT), **{tf_kb:.1f} KB**. "
            f"Recommended for Android NNAPI / iOS Core ML delegate paths "
            f"(NPU acceleration).\n"
        )

    content = f'''# Wake-word model export

Trained for phrase: **"{phrase}"**. Sigmoid output threshold: **{threshold:.3f}**.

## Files

- `wake.onnx`: float32 ONNX model, **{size_kb:.1f} KB**, {n_params} parameters. Graph-optimized (Conv+ReLU fused).
{int8_line}{tflite_line}- `wake.json`: metadata sidecar: threshold, mel params, preprocessing chain.

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
logit = sess.run(None, {{"mel": mel}})[0][0]
prob = 1.0 / (1.0 + np.exp(-logit)) # sigmoid
print(f"prob = {{prob:.3f}}  triggered = {{prob > meta['threshold']}}")
```

## Usage (Android, Kotlin): TFLite + NNAPI delegate

```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate

val options = Interpreter.Options().apply {{
    // NNAPI dispatches to NPU/DSP when the device supports it.
    // Falls back to CPU silently on unsupported devices.
    addDelegate(NnApiDelegate())
    setNumThreads(2)
}}
val interpreter = Interpreter(loadModelFile(context, "wake.tflite"), options)

// Input: float32 mel features, shape (1, 40, 101). See preprocessing below
val input = Array(1) {{ Array(40) {{ FloatArray(101) }} }}
val output = Array(1) {{ FloatArray(1) }}
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
let logit = output.data.withUnsafeBytes {{ $0.load(as: Float32.self) }}
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
'''
    # UTF-8 explicit. Windows would otherwise crash on unicode chars
    (output_dir / "README.md").write_text(content, encoding="utf-8")
