// React Native wake-word detector. Same logic as the browser demo's
// wakeword.js, but engine-agnostic: it takes a `runtime` ({ run(mel) -> logit })
// supplied by App.js, which may be ONNX (onnxruntime-react-native) or TFLite
// (react-native-fast-tflite).

import { N_FFT, StreamingPreprocessor } from "./preprocessing.js";

function energyGate(audio, meta) {
  const rmsThreshDbfs = meta.energy_gate?.rms_threshold_dbfs ?? -55.0;
  const voiceFracMin = meta.energy_gate?.voice_band_min_fraction ?? 0.15;

  let sumSq = 0;
  for (let i = 0; i < audio.length; i++) sumSq += audio[i] * audio[i];
  const rms = Math.sqrt(sumSq / audio.length);
  if (rms < 1e-9) return false;
  const rmsDbfs = 20 * Math.log10(rms);
  if (rmsDbfs < rmsThreshDbfs) return false;

  // First-order HPF at ~100 Hz; voice-band proxy
  const alpha = 0.987;
  let xPrev = 0, yPrev = 0;
  let bandSumSq = 0;
  for (let i = 0; i < audio.length; i++) {
    const y = alpha * (yPrev + audio[i] - xPrev);
    bandSumSq += y * y;
    xPrev = audio[i];
    yPrev = y;
  }
  const bandFrac = Math.sqrt(bandSumSq / audio.length) / (rms + 1e-9);
  return bandFrac >= voiceFracMin;
}

export class WakeWordDetector {
  /**
   * @param {{run: (mel: Float32Array) => Promise<number>}} runtime  inference engine
   * @param {object} meta  parsed wake.json
   */
  constructor(runtime, meta) {
    // `runtime` abstracts the inference engine: { run(melFloat32) -> Promise<logit> }.
    // App.js supplies an ONNX (onnxruntime-react-native) or TFLite
    // (react-native-fast-tflite) implementation, so this detector is
    // engine-agnostic — the streaming/gate/trigger logic is identical for both.
    this.runtime = runtime;
    this.meta = meta;
    // Safety: the mel features (and thus n_fft) are baked into the model at
    // training time. Loading a model exported with a different n_fft would
    // silently feed it wrong features and tank accuracy. Fail loudly instead.
    if (meta.n_fft != null && meta.n_fft !== N_FFT) {
      throw new Error(
        `wake.json n_fft=${meta.n_fft} but this preprocessor uses n_fft=${N_FFT}. ` +
        `Re-export the model from the current heed ('heed export <project>').`
      );
    }
    this.threshold = meta.threshold;
    this.preprocessor = new StreamingPreprocessor();

    const t = meta.trigger ?? {};
    this.emaAlpha = t.ema_alpha ?? 0.5;
    this.consecutiveFrames = t.consecutive_frames ?? 2;
    this.refractorySeconds = t.refractory_seconds ?? 0.7;

    this.ema = 0;
    this.aboveCount = 0;
    this.lastTriggerTime = -Infinity;
  }

  reset() {
    this.preprocessor.reset();
    this.ema = 0;
    this.aboveCount = 0;
    this.lastTriggerTime = -Infinity;
  }

  /**
   * Process a new audio chunk. Returns { prob, ema, triggered, gated, latencyMs }.
   * @param {Float32Array} audioChunk
   */
  async step(audioChunk) {
    const t0 = Date.now();
    // Always ingest so the causal filter + ring buffer stay continuous across
    // gated (silent) chunks; ingest is cheap. The gate controls only the
    // expensive STFT + model run below.
    this.preprocessor.ingest(audioChunk);
    const gated = !energyGate(audioChunk, this.meta);
    if (gated) {
      this.ema *= 0.5;
      this.aboveCount = 0;
      return { prob: 0, ema: this.ema, triggered: false, gated: true,
               latencyMs: 0, latencyPrepMs: 0, latencyInferMs: 0 };
    }

    const tPrep0 = Date.now();
    const mel = this.preprocessor.computeMel();
    const tPrep1 = Date.now();
    const logit = await this.runtime.run(mel);
    const tInfer1 = Date.now();
    const prob = 1 / (1 + Math.exp(-logit));

    this.ema = this.emaAlpha * prob + (1 - this.emaAlpha) * this.ema; // for display/smoothing

    // Trigger on consecutive RAW-prob crossings. We previously also required
    // ema > threshold, but the EMA lags by design — a short, confident wake
    // word crosses threshold for 2-3 frames while the smoothed ema is still
    // climbing, so it never fired. The consecutive-frame count is itself the
    // debounce that keeps false-accepts low.
    let triggered = false;
    if (prob > this.threshold) {
      this.aboveCount++;
    } else {
      this.aboveCount = 0;
    }
    const nowSec = Date.now() / 1000;
    if (
      this.aboveCount >= this.consecutiveFrames &&
      nowSec - this.lastTriggerTime > this.refractorySeconds
    ) {
      triggered = true;
      this.lastTriggerTime = nowSec;
      this.aboveCount = 0;
    }
    return {
      prob, ema: this.ema, triggered, gated: false,
      latencyMs: Date.now() - t0,
      latencyPrepMs: tPrep1 - tPrep0,
      latencyInferMs: tInfer1 - tPrep1,
    };
  }
}
