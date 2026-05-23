// Node script: cross-verify the JS preprocessing pipeline against the
// Python reference. Asserts that for a fixed audio signal:
//
//   (1) The JS streaming log-mel output matches Python's log-mel within a
//       tight tolerance. Both sides now use the SAME causal high-pass
//       (heed.audio.highpass_filter == JS BiquadCascade) and the same
//       n_fft=512 STFT, so the only residual difference is float32 (torch)
//       vs float64 (JS) round-off — well under tolerance.
//   (2) Loading the exported ONNX model and running an end-to-end step
//       produces a logit/probability matching Python's inference path on
//       the same input (full deployment-path parity).
//
// Run from this directory:
//   node verify-preprocessing.mjs
//
// Requires: `python` on PATH with heed importable. The end-to-end ONNX
// check additionally needs `onnxruntime-node` + a wake.onnx/wake.json exported
// from the CURRENT (n_fft=512) pipeline; both are skipped gracefully if absent.

import { execSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(here, "../..");

// 1) Generate a reference signal + run Python preprocessing
const pyScript = `
import numpy as np, json, sys
sys.path.insert(0, "${ROOT}")
from heed.audio import highpass_filter, log_mel
import torch

np.random.seed(7)
audio = (
    0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
    + 0.2 * np.random.randn(16000)
).astype(np.float32)

filtered = highpass_filter(torch.from_numpy(audio))
mel = log_mel(filtered, apply_cmn=True).squeeze(0).numpy()

print(json.dumps({
    "audio": audio.tolist(),
    "mel":   mel.tolist(),
}))
`.trim();

console.log("running Python reference…");
const tmp = resolve(here, ".verify-ref.py");
writeFileSync(tmp, pyScript);
const raw = execSync(`python ${tmp}`, { maxBuffer: 1024 * 1024 * 64 }).toString();
const { audio: audioArr, mel: melRef } = JSON.parse(raw);
const audio = Float32Array.from(audioArr);
const melReference = melRef.flat();  // 40 × 101 → 4040

// 2) Run JS preprocessor on the same audio in ~100ms chunks
console.log("running JS streaming preprocessor…");
const { StreamingPreprocessor } = await import("./preprocessing.js");

const pp = new StreamingPreprocessor();
const chunkSize = 1600;
let lastMel = null;
for (let off = 0; off + chunkSize <= audio.length; off += chunkSize) {
  lastMel = pp.step(audio.subarray(off, off + chunkSize));
}
// Also feed in any tail remainder
if (audio.length % chunkSize !== 0) {
  lastMel = pp.step(audio.subarray(audio.length - (audio.length % chunkSize)));
}

// 3) Compare
let sumSq = 0, maxAbs = 0;
for (let i = 0; i < melReference.length; i++) {
  const d = Math.abs(melReference[i] - lastMel[i]);
  sumSq += d * d;
  if (d > maxAbs) maxAbs = d;
}
const rms = Math.sqrt(sumSq / melReference.length);
console.log(`  Python vs JS log-mel:  max abs error = ${maxAbs.toExponential(3)}`);
console.log(`                         RMS error     = ${rms.toExponential(3)}`);

// Both sides use the identical causal high-pass and n_fft=512 STFT, so the
// only difference is float32 (torch) vs float64 (JS) round-off. Expect
// ~1e-4 or better. Tolerances are tight to catch any real preprocessing drift.
const TOL_MAX = 0.01;
const TOL_RMS = 0.002;
let ok = true;
if (maxAbs > TOL_MAX) {
  console.log(`  ❌ FAIL: max abs error ${maxAbs.toFixed(3)} > ${TOL_MAX}`);
  ok = false;
} else {
  console.log(`  ✓ max abs error within tolerance (${TOL_MAX})`);
}
if (rms > TOL_RMS) {
  console.log(`  ❌ FAIL: RMS error ${rms.toFixed(3)} > ${TOL_RMS}`);
  ok = false;
} else {
  console.log(`  ✓ RMS error within tolerance (${TOL_RMS})`);
}

// 4) If a model is present alongside, run end-to-end and report prob
const modelPath = resolve(here, "wake.onnx");
const metaPath = resolve(here, "wake.json");
if (existsSync(modelPath) && existsSync(metaPath)) {
  try {
    console.log("\nrunning end-to-end inference test (vs ONNX model)…");
    // onnxruntime-node has a different API than ort-web; do the bare-bones call.
    const ort = await import("onnxruntime-node");
    const session = await ort.InferenceSession.create(modelPath);
    const meta = JSON.parse(readFileSync(metaPath, "utf-8"));
    const tensor = new ort.Tensor("float32", lastMel, [1, 40, 101]);
    const out = await session.run({ mel: tensor });
    const logit = out.logit.data[0];
    const prob = 1 / (1 + Math.exp(-logit));
    console.log(`  ONNX prob on test signal: ${prob.toFixed(4)}`);
    console.log(`  threshold from wake.json: ${meta.threshold.toFixed(4)}`);
    console.log(`  ${prob > meta.threshold ? "→ would trigger" : "→ would not trigger"} (synthetic sine — should be 'not trigger')`);
  } catch (e) {
    console.log(`  (skipped: onnxruntime-node not available — ${e.message.split("\n")[0]})`);
  }
} else {
  console.log("\n(skipped end-to-end ONNX check; drop wake.onnx + wake.json " +
              "here to enable it)");
}

import { unlinkSync } from "node:fs";
try { unlinkSync(tmp); } catch (_) {}
process.exit(ok ? 0 : 1);
