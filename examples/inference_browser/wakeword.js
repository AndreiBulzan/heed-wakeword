// WakeWordDetector — wraps the streaming preprocessor + ONNX session +
// trigger logic (hysteresis + refractory) into a single object the UI
// can drive frame-by-frame.

import {
  N_FFT, N_FRAMES, N_MELS, StreamingPreprocessor,
} from "./preprocessing.js";

/** RMS + voice-band power gate, a lightweight proxy of heed.gate (Python).
 *  Skips the model when the audio is silent or non-speech-shaped. This is the
 *  power saver on always-on devices, and the reason the Python detector stays
 *  quiet on ambient.
 *
 *  Runs on the FILTERED 1-second window (post 100 Hz high-pass + 50/60 Hz
 *  notch), exactly like heed.infer. That matters: a quiet room dominated by
 *  sub-100 Hz rumble (HVAC, desk, handling) sits above the RMS floor on the raw
 *  mic but drops well below it once the rumble is removed, so the model is
 *  skipped instead of being fed near-silence ~10x a second.
 *
 *  Returns { pass, rmsDbfs, bandFrac } so the UI can show why a frame gated.
 */
function energyGate(filtered, meta) {
  const rmsThreshDbfs = meta.energy_gate?.rms_threshold_dbfs ?? -55.0;
  const voiceFracMin = meta.energy_gate?.voice_band_min_fraction ?? 0.15;

  let sumSq = 0;
  for (let i = 0; i < filtered.length; i++) sumSq += filtered[i] * filtered[i];
  const rms = Math.sqrt(sumSq / filtered.length);
  const rmsDbfs = rms < 1e-9 ? -Infinity : 20 * Math.log10(rms);
  if (rmsDbfs < rmsThreshDbfs) return { pass: false, rmsDbfs, bandFrac: 0 };

  // Voice-band power fraction. The 100 Hz lower bound is already enforced by the
  // biquad high-pass that produced `filtered`, so the gate only adds the 7000 Hz
  // upper bound (a one-pole low-pass, drops mic hiss) and takes the in-band
  // POWER fraction (band energy / total energy). This matches heed.gate's
  // |FFT|^2 band ratio and its 0.15 threshold. The previous build divided
  // band-RMS by full-RMS, an amplitude ratio (the square root of this), so it
  // read far higher than 0.15 and almost never gated.
  const aLp = 0.936;  // one-pole low-pass at ~7000 Hz: 1 - exp(-2*pi*7000/16000)
  let yLp = 0, bandSumSq = 0;
  for (let i = 0; i < filtered.length; i++) {
    yLp = yLp + aLp * (filtered[i] - yLp);
    bandSumSq += yLp * yLp;
  }
  const bandFrac = bandSumSq / (sumSq + 1e-12);
  return { pass: bandFrac >= voiceFracMin, rmsDbfs, bandFrac };
}

export class WakeWordDetector {
  /**
   * @param {InferenceSession} session  ONNX Runtime Web session
   * @param {object} meta  parsed wake.json
   */
  constructor(session, meta) {
    this.session = session;
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

    // Smoothing + hysteresis (mirrors python infer.WakeWordDetector)
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

  /** Process a new audio chunk. Returns { prob, ema, triggered, gated }. */
  async step(audioChunk) {
    // Always ingest so the causal filter + ring buffer stay continuous across
    // gated (silent) chunks; ingest is cheap (O(samples)). The energy gate runs
    // on RAW audio and only controls the expensive STFT + model run below.
    this.preprocessor.ingest(audioChunk);
    // Gate on the FILTERED 1-second window the preprocessor just updated (the
    // model's actual input), exactly like heed.infer. Gating on the raw mic
    // buffer instead let sub-100 Hz rumble hold the level above the RMS floor,
    // so the model ran on near-silent-but-rumbly ambient and occasionally
    // false-fired; the Python detector gates post-high-pass and stays quiet.
    const g = energyGate(this.preprocessor.filteredBuffer, this.meta);
    if (!g.pass) {
      this.ema *= 0.5;
      this.aboveCount = 0;
      return { prob: 0, ema: this.ema, triggered: false, gated: true,
               rmsDbfs: g.rmsDbfs, bandFrac: g.bandFrac };
    }

    const mel = this.preprocessor.computeMel();
    // ONNX expects shape [1, 40, 101]
    const tensor = new ort.Tensor(
      "float32", mel, [1, N_MELS, N_FRAMES],
    );
    const result = await this.session.run({ mel: tensor });
    // Output name from torch.onnx.export is "logit" (see export.py)
    const logit = result.logit.data[0];
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
    const now = performance.now() / 1000;
    if (
      this.aboveCount >= this.consecutiveFrames &&
      now - this.lastTriggerTime > this.refractorySeconds
    ) {
      triggered = true;
      this.lastTriggerTime = now;
      this.aboveCount = 0;
    }
    return { prob, ema: this.ema, triggered, gated: false,
             rmsDbfs: g.rmsDbfs, bandFrac: g.bandFrac };
  }
}
