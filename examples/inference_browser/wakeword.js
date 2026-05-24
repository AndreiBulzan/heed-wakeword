// WakeWordDetector — wraps the streaming preprocessor + ONNX session +
// trigger logic (hysteresis + refractory) into a single object the UI
// can drive frame-by-frame.

import {
  N_FFT, N_FRAMES, N_MELS, SAMPLE_RATE, StreamingPreprocessor,
} from "./preprocessing.js";

const VOICE_BAND_LO = 100, VOICE_BAND_HI = 7000;

/** Cheap RMS + voice-band gate. Skips model invocation when audio is
 *  silent or non-speech-shaped — major power saver on always-on devices.
 *  Returns true if the gate passes (audio likely contains speech).
 */
function energyGate(audio, meta) {
  const rmsThreshDbfs = meta.energy_gate?.rms_threshold_dbfs ?? -55.0;
  const voiceFracMin = meta.energy_gate?.voice_band_min_fraction ?? 0.15;

  // RMS in dBFS
  let sumSq = 0;
  for (let i = 0; i < audio.length; i++) sumSq += audio[i] * audio[i];
  const rms = Math.sqrt(sumSq / audio.length);
  if (rms < 1e-9) return false;
  const rmsDbfs = 20 * Math.log10(rms);
  if (rmsDbfs < rmsThreshDbfs) return false;

  // Voice-band (100-7000 Hz) energy fraction: the same idea as heed.gate, done
  // with cheap one-pole filters instead of an FFT. A high-pass at 100 Hz drops
  // rumble (fans, handling); a low-pass at 7000 Hz drops mic hiss; the ratio of
  // the band-passed RMS to the full RMS approximates the spectral voice-band
  // fraction. The upper cutoff is what stops steady fan/hiss noise from passing.
  const aHp = 0.987;  // high-pass, exp(-2*pi*100/16000)
  const aLp = 0.936;  // low-pass,  1 - exp(-2*pi*7000/16000)
  let xPrev = 0, yHp = 0, yLp = 0, bandSumSq = 0;
  for (let i = 0; i < audio.length; i++) {
    const x = audio[i];
    yHp = aHp * (yHp + x - xPrev);   // high-pass above 100 Hz
    yLp = yLp + aLp * (yHp - yLp);   // low-pass below 7000 Hz
    bandSumSq += yLp * yLp;
    xPrev = x;
  }
  const bandFrac = Math.sqrt(bandSumSq / audio.length) / (rms + 1e-9);
  return bandFrac >= voiceFracMin;
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
    // Raw 1-second ring buffer for the energy gate, so it sees the same window
    // the model does (heed.gate runs on the full 1-s frame). Gating on the tiny
    // per-call chunk let brief noise gusts cross the RMS threshold too easily.
    this.gateBuffer = new Float32Array(SAMPLE_RATE);

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
    this.gateBuffer.fill(0);
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
    // Roll the raw chunk into the 1-second gate buffer and gate on the whole
    // second, matching heed.gate's window (not just this ~100 ms chunk).
    const gb = this.gateBuffer, L = gb.length, n = audioChunk.length;
    if (n >= L) {
      for (let i = 0; i < L; i++) gb[i] = audioChunk[n - L + i];
    } else {
      gb.copyWithin(0, n);
      for (let i = 0; i < n; i++) gb[L - n + i] = audioChunk[i];
    }
    const gated = !energyGate(this.gateBuffer, this.meta);
    if (gated) {
      this.ema *= 0.5;
      this.aboveCount = 0;
      return { prob: 0, ema: this.ema, triggered: false, gated: true };
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
    return { prob, ema: this.ema, triggered, gated: false };
  }
}
