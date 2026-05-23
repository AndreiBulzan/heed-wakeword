// Streaming audio preprocessing for heed wake-word inference.
//
// Reference implementation of the preprocessing chain documented in
// wake.json[preprocessing]. The exact same chain must run in any deployment
// language (Swift / Kotlin / C). This JS version is dependency-free so it
// doubles as a porting template, and is byte-identical between the browser
// and React-Native demos (pure DSP, no platform specifics).
//
// Pipeline (per audio chunk, typically ~100 ms):
//   ingest(newAudio):
//     newAudio --> causal Biquad cascade (HPF 100 Hz + 50/60 Hz notch),
//                  stateful: each new sample is filtered exactly ONCE and
//                  its filtered value never changes again
//              --> append to the 1-s filtered ring buffer
//   computeMel():
//              --> compute ONLY the STFT frames the new audio changed
//                  (older frames carry over unchanged from the previous hop)
//              --> mel projection (sparse matmul) + log
//              --> CMN over the 101-frame buffer
//              --> Float32Array[40 * 101] log-mel features for the ONNX model
//
// Why this is fast AND correct: the high-pass is CAUSAL (single-pass biquad
// with retained state), bit-identical to heed.audio.highpass_filter run over
// the whole stream. Because a causal filter's output for a sample never
// depends on future samples, already-filtered audio is immutable, so the STFT
// only needs to recompute the frames touched by new audio (plus the few
// reflect-padding frames at the buffer edges). The assembled 101-frame log-mel
// is therefore bit-equivalent to a from-scratch torch.stft over the buffer —
// see verify-preprocessing.mjs, which checks JS vs Python and streaming vs
// one-shot.
//
// n_fft is a power of two (512), so the transform is a plain radix-2 FFT. The
// 25 ms analysis window (400 samples) is zero-padded to 512 and centered,
// exactly as torch.stft(n_fft=512, win_length=400) does.
//
// Note on peak normalization: omitted at inference. CMN makes log-mel
// invariant to any constant audio scaling (scaling audio by k adds 2·log(k) to
// every log-mel bin uniformly across frames; CMN subtracts the per-bin mean
// which absorbs that offset), so peak_normalize is a no-op here.

import { PARAMS, FILTERS, HANN_WINDOW, MEL_FB_SPARSE } from "./filter-coeffs.js";

export const SAMPLE_RATE = PARAMS.sampleRate;     // 16000
export const N_FFT = PARAMS.nFft;                 // 512 (power of two)
export const WIN_LENGTH = PARAMS.winLength;       // 400 (25 ms analysis window)
export const HOP = PARAMS.hop;                    // 160 (10 ms)
export const N_MELS = PARAMS.nMels;               // 40
export const N_FRAMES = PARAMS.nFrames;           // 101
export const WINDOW_SAMPLES = SAMPLE_RATE;        // 1-second ring buffer
const N_BINS = PARAMS.nBins;                      // 257 = N_FFT/2 + 1

// ----- Causal biquad SOS cascade (stateful, streaming) -------------------
//
// Direct Form II Transposed per section:
//     y  = b0*x + s1
//     s1 = b1*x + s2 - a1*y
//     s2 = b2*x      - a2*y
// State (s1, s2) is retained across calls so chunk-by-chunk filtering is
// bit-identical to filtering the concatenated stream in one shot (LTI). On the
// first chunk each section is seeded with `zi * (stage's first input sample)`,
// matching scipy.signal.sosfilt(sos, x, zi=sosfilt_zi*x[0]) used by
// heed.audio.StreamingHighpass / highpass_filter.

class BiquadCascade {
  constructor(filters) {
    this.stages = filters.map((f) => ({
      sections: f.sections,
      zi: f.zi,
      s1: new Float64Array(f.sections.length),
      s2: new Float64Array(f.sections.length),
      inited: false,
    }));
  }

  reset() {
    for (const st of this.stages) {
      st.s1.fill(0);
      st.s2.fill(0);
      st.inited = false;
    }
  }

  /** Filter x[0..len) in place, retaining state across calls. */
  process(x, len) {
    if (len === 0) return;
    for (const st of this.stages) {
      const secs = st.sections;
      if (!st.inited) {
        const x0 = x[0]; // first input sample seen by THIS stage
        for (let s = 0; s < secs.length; s++) {
          st.s1[s] = st.zi[s][0] * x0;
          st.s2[s] = st.zi[s][1] * x0;
        }
        st.inited = true;
      }
      for (let s = 0; s < secs.length; s++) {
        const sec = secs[s];
        const b0 = sec.b0, b1 = sec.b1, b2 = sec.b2, a1 = sec.a1, a2 = sec.a2;
        let s1 = st.s1[s], s2 = st.s2[s];
        for (let i = 0; i < len; i++) {
          const xi = x[i];
          const y = b0 * xi + s1;
          s1 = b1 * xi + s2 - a1 * y;
          s2 = b2 * xi - a2 * y;
          x[i] = y;
        }
        st.s1[s] = s1;
        st.s2[s] = s2;
      }
    }
  }
}

// ----- Analysis window: 400-sample Hann centered in the 512-point frame ---

const WINDOW = (() => {
  const w = new Float64Array(N_FFT);
  const off = (N_FFT - WIN_LENGTH) >> 1; // 56 — torch centers win_length in n_fft
  for (let n = 0; n < WIN_LENGTH; n++) w[off + n] = HANN_WINDOW[n];
  return w;
})();

// ----- Radix-2 FFT (N = N_FFT, power of two) ------------------------------
//
// In-place iterative Cooley-Tukey (decimation-in-time). Bit-reversal indices
// and twiddle factors are precomputed once. Forward transform (exp(-2πi kn/N)),
// unnormalized — matches torch.stft's convention.

const FFT_LOG2 = Math.round(Math.log2(N_FFT));
const FFT_REV = (() => {
  const r = new Uint16Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    let x = i, y = 0;
    for (let j = 0; j < FFT_LOG2; j++) { y = (y << 1) | (x & 1); x >>= 1; }
    r[i] = y;
  }
  return r;
})();
const FFT_COS = new Float64Array(N_FFT >> 1);
const FFT_SIN = new Float64Array(N_FFT >> 1);
for (let k = 0; k < (N_FFT >> 1); k++) {
  const a = (-2 * Math.PI * k) / N_FFT;
  FFT_COS[k] = Math.cos(a);
  FFT_SIN[k] = Math.sin(a);
}

function fftRadix2(re, im) {
  const N = N_FFT, rev = FFT_REV;
  for (let i = 0; i < N; i++) {
    const j = rev[i];
    if (j > i) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >> 1;
    const step = (N / len) | 0;
    for (let i = 0; i < N; i += len) {
      let tw = 0;
      for (let k = 0; k < half; k++) {
        const wr = FFT_COS[tw], wi = FFT_SIN[tw];
        const a = i + k, b = a + half;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr;        im[a] += xi;
        tw += step;
      }
    }
  }
}

// ----- Streaming preprocessor --------------------------------------------

/** Pushes audio chunks through the causal HPF + STFT + mel + CMN. Emits the
 *  current (40 × 101) log-mel tensor as a Float32Array of length 4040, laid
 *  out bin-major ([bin0_frame0..bin0_frame100, bin1_frame0, …]) — i.e. ONNX
 *  channels-first shape [1, 40, 101].
 *
 *  Usage: call ingest(chunk) for every chunk to keep the filter + buffer
 *  continuous (cheap), then computeMel() when you actually want features (e.g.
 *  only when an energy gate passes). step(chunk) does both for convenience.
 */
export class StreamingPreprocessor {
  constructor() {
    this.cascade = new BiquadCascade([
      FILTERS.hpf_100, FILTERS.notch_50, FILTERS.notch_60,
    ]);
    this.filteredBuffer = new Float32Array(WINDOW_SAMPLES);
    // mel ring buffer, bin-major for cheap per-row ring-shift.
    this.melBuffer = new Float32Array(N_MELS * N_FRAMES);
    this.pendingSamples = 0;
    this.initialized = false;
    // Scratch
    this._re = new Float64Array(N_FFT);
    this._im = new Float64Array(N_FFT);
    this._spec = new Float64Array(N_BINS);
    this._output = new Float32Array(N_MELS * N_FRAMES);
    this._filtTmp = new Float64Array(WINDOW_SAMPLES);
  }

  reset() {
    this.cascade.reset();
    this.filteredBuffer.fill(0);
    this.melBuffer.fill(0);
    this.pendingSamples = 0;
    this.initialized = false;
  }

  /** Filter new audio (causal, stateful) and append to the 1-s ring buffer.
   *  Cheap (O(new samples)); safe to call on every chunk including silence so
   *  the filter state and buffer stay continuous. */
  ingest(audioChunk) {
    const n = audioChunk.length;
    if (n === 0) return;
    if (this._filtTmp.length < n) this._filtTmp = new Float64Array(n);
    const tmp = this._filtTmp;
    for (let i = 0; i < n; i++) tmp[i] = audioChunk[i];
    this.cascade.process(tmp, n); // causal filter in place, retains state

    const fb = this.filteredBuffer;
    if (n >= WINDOW_SAMPLES) {
      const start = n - WINDOW_SAMPLES;
      for (let i = 0; i < WINDOW_SAMPLES; i++) fb[i] = tmp[start + i];
      this.pendingSamples = WINDOW_SAMPLES;
    } else {
      fb.copyWithin(0, n);
      for (let i = 0; i < n; i++) fb[WINDOW_SAMPLES - n + i] = tmp[i];
      this.pendingSamples = Math.min(WINDOW_SAMPLES, this.pendingSamples + n);
    }
  }

  /** Recompute the STFT frames touched since the last call, then CMN + emit.
   *  Returns the (40 × 101) log-mel Float32Array. */
  computeMel() {
    const nAdvance = Math.min(N_FRAMES, Math.floor(this.pendingSamples / HOP));
    this.pendingSamples -= nAdvance * HOP;

    if (!this.initialized || nAdvance >= N_FRAMES - 3) {
      // First fill, or so much new audio that nearly every frame changed.
      for (let i = 0; i < N_FRAMES; i++) this._computeFrame(i);
      this.initialized = true;
    } else if (nAdvance > 0) {
      this._shiftAndRecompute(nAdvance);
    }
    return this._cmnAndEmit();
  }

  /** ingest + computeMel in one call (used by the verify harness). */
  step(audioChunk) {
    this.ingest(audioChunk);
    return this.computeMel();
  }

  /** Slide the mel ring buffer left by nAdvance and recompute only the frames
   *  that actually changed: the new/right-edge frames (new audio + the 2
   *  reflect-padding frames at the right) and the 2 reflect-padding frames at
   *  the left. Interior frames are unchanged (causal filtering ⇒ immutable
   *  filtered samples ⇒ immutable frames) and just shift position. */
  _shiftAndRecompute(nAdvance) {
    const mb = this.melBuffer;
    for (let mi = 0; mi < N_MELS; mi++) {
      const base = mi * N_FRAMES;
      // new column j ← old column (j + nAdvance)
      mb.copyWithin(base, base + nAdvance, base + N_FRAMES);
    }
    // Left edge always uses reflect padding → recompute.
    this._computeFrame(0);
    this._computeFrame(1);
    // Right region: frames touching new audio + the 2 right reflect-pad frames.
    let rStart = N_FRAMES - nAdvance - 2;
    if (rStart < 2) rStart = 2;
    for (let i = rStart; i < N_FRAMES; i++) this._computeFrame(i);
  }

  /** Compute log-mel for frame index i, write into melBuffer column i.
   *  Frame i spans filteredBuffer[i*HOP - N_FFT/2, i*HOP + N_FFT/2) with
   *  reflect padding at the buffer edges (matches torch.stft center=True,
   *  pad_mode='reflect'). */
  _computeFrame(i) {
    const lo = i * HOP - (N_FFT >> 1);
    const fb = this.filteredBuffer, re = this._re, im = this._im, win = WINDOW;
    for (let n = 0; n < N_FFT; n++) {
      let s = lo + n;
      if (s < 0) s = -s; // reflect around 0 (no edge repeat)
      else if (s >= WINDOW_SAMPLES) s = 2 * (WINDOW_SAMPLES - 1) - s;
      if (s < 0) s = 0; else if (s >= WINDOW_SAMPLES) s = WINDOW_SAMPLES - 1;
      re[n] = fb[s] * win[n];
      im[n] = 0;
    }
    fftRadix2(re, im);

    const spec = this._spec;
    for (let k = 0; k < N_BINS; k++) {
      const ar = re[k], ai = im[k];
      spec[k] = ar * ar + ai * ai; // power
    }

    const mb = this.melBuffer;
    for (let mi = 0; mi < N_MELS; mi++) {
      let acc = 0;
      const sparse = MEL_FB_SPARSE[mi];
      for (let p = 0; p < sparse.length; p++) acc += sparse[p][1] * spec[sparse[p][0]];
      // torch: log(max(power, 1e-9))
      mb[mi * N_FRAMES + i] = Math.log(acc > 1e-9 ? acc : 1e-9);
    }
  }

  /** CMN per mel bin across the 101 frames, emit channels-first. */
  _cmnAndEmit() {
    const mb = this.melBuffer, out = this._output;
    for (let mi = 0; mi < N_MELS; mi++) {
      const base = mi * N_FRAMES;
      let sum = 0;
      for (let f = 0; f < N_FRAMES; f++) sum += mb[base + f];
      const mean = sum / N_FRAMES;
      for (let f = 0; f < N_FRAMES; f++) out[base + f] = mb[base + f] - mean;
    }
    return out;
  }
}
