"""Audio I/O, mel features, normalization, trimming.

Pure torch + scipy + soundfile. No torchaudio C-extension required - this
was a portability issue on Windows where torchaudio's binary often mismatches
the installed torch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf
import torch

from . import HOP_SAMPLES, N_FFT, N_MELS, SAMPLE_RATE, WIN_SAMPLES, WINDOW_FRAMES


# ----- pure-torch mel feature extractor (matches torchaudio at <1e-3 max-abs) ---


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _build_mel_filterbank(
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    sample_rate: int = SAMPLE_RATE,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> torch.Tensor:
    if fmax is None:
        fmax = sample_rate / 2
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = torch.tensor([_mel_to_hz(m.item()) for m in mel_pts])
    freqs = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(n_mels):
        left, center, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        rising = (freqs - left) / (center - left + 1e-9)
        falling = (right - freqs) / (right - center + 1e-9)
        fb[m] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return fb


_MEL_FB: Optional[torch.Tensor] = None
_HANN: Optional[torch.Tensor] = None


def _mel_fb() -> torch.Tensor:
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _build_mel_filterbank()
    return _MEL_FB


def _hann() -> torch.Tensor:
    global _HANN
    if _HANN is None:
        _HANN = torch.hann_window(WIN_SAMPLES)
    return _HANN


def log_mel(audio: torch.Tensor, apply_cmn: bool = True) -> torch.Tensor:
    """Compute log-mel spectrogram. Input (T,) or (B, T); output (B, n_mels, F).

    When `apply_cmn` is True (default), subtracts the per-clip mean across
    time from each mel bin. This is **cepstral mean normalization** - a
    standard ASR trick that makes the representation invariant to mic
    frequency response and channel gain (these become additive constants
    in log-mel, eliminated by mean subtraction). Critical for cross-mic
    robustness; the model trained without it is locked to the trainer's
    mic spectrum.

    Must be applied **consistently** at train and inference: a model trained
    with CMN won't work without it and vice versa. Hence default-on
    everywhere in this codebase.
    """
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    spec = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_SAMPLES,
        win_length=WIN_SAMPLES,
        window=_hann().to(audio.device),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    power = spec.real.pow(2) + spec.imag.pow(2)
    fb = _mel_fb().to(audio.device)
    mel = fb @ power  # (B, n_mels, T)
    out = torch.log(mel.clamp(min=1e-9))
    if apply_cmn:
        # Per-clip mean across time, per mel bin. Subtraction removes the
        # multiplicative mic-spectrum contribution that became additive
        # after log.
        out = out - out.mean(dim=-1, keepdim=True)
    return out


def expected_frames(window_samples: int = SAMPLE_RATE) -> int:
    return window_samples // HOP_SAMPLES + 1


# ----- WAV I/O with scipy-based resample ------------------------------------


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """scipy polyphase resample. Good quality, no C-ext dependency."""
    if src_sr == dst_sr:
        return audio
    g = math.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    return scipy.signal.resample_poly(audio, up, down).astype(np.float32)


_LOW_SR_WARNED: set[str] = set()


def load_wav(path: str | Path) -> torch.Tensor:
    """Load any WAV file → mono float32 tensor at SAMPLE_RATE.

    Auto-resamples non-16 kHz inputs via scipy polyphase, auto-downmixes
    stereo to mono, auto-converts integer PCM to float32 (handled by
    soundfile). The only sample-rate footgun is *very low* rates: telephony
    (8 kHz) drops content above 4 kHz, which removes most fricatives
    (s, sh, f). We warn once per file in that case.
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr < 14000 and str(path) not in _LOW_SR_WARNED:
        _LOW_SR_WARNED.add(str(path))
        print(
            f"[warn] {path} is only {sr} Hz - high-frequency consonants "
            f"(s/sh/f) will be missing. Re-record at 16 kHz or higher if "
            f"your wake word contains them."
        )
    if sr != SAMPLE_RATE:
        audio = _resample(audio, sr, SAMPLE_RATE)
    return torch.from_numpy(np.ascontiguousarray(audio))


def save_wav(path: str | Path, audio: torch.Tensor | np.ndarray) -> None:
    """Save mono audio to WAV at SAMPLE_RATE."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


# ----- normalization + windowing -------------------------------------------


def peak_normalize(audio: torch.Tensor, target_dbfs: float = -3.0) -> torch.Tensor:
    """Scale audio so peak is at target_dbfs."""
    peak = audio.abs().max().clamp(min=1e-9)
    target = 10 ** (target_dbfs / 20.0)
    if peak < 1e-6:
        return audio
    return audio * (target / peak)


# Cached filter coefficients so we don't recompute butter() on every clip.
_HPF_CACHE: dict[tuple[float, int, int], np.ndarray] = {}
_NOTCH_CACHE: dict[tuple[float, float, int], np.ndarray] = {}


def _hpf_coefficients(cutoff_hz: float, sample_rate: int, order: int) -> np.ndarray:
    key = (float(cutoff_hz), int(sample_rate), int(order))
    if key not in _HPF_CACHE:
        sos = scipy.signal.butter(order, cutoff_hz, btype="highpass",
                                  fs=sample_rate, output="sos")
        _HPF_CACHE[key] = sos
    return _HPF_CACHE[key]


def _notch_coefficients(freq_hz: float, q: float, sample_rate: int) -> np.ndarray:
    """IIR notch (biquad) coefficients. q higher = narrower notch."""
    key = (float(freq_hz), float(q), int(sample_rate))
    if key not in _NOTCH_CACHE:
        b, a = scipy.signal.iirnotch(freq_hz, q, fs=sample_rate)
        # convert to SOS so the causal sosfilt cascade is applied uniformly
        sos = scipy.signal.tf2sos(b, a)
        _NOTCH_CACHE[key] = sos
    return _NOTCH_CACHE[key]


def highpass_filter(
    audio: torch.Tensor,
    cutoff_hz: float = 100.0,
    sample_rate: int = SAMPLE_RATE,
    order: int = 8,
    apply_mains_notch: bool = True,
) -> torch.Tensor:
    """Aggressively remove sub-cutoff_hz content + mains-hum notches.

    Default 8th-order Butterworth at 100 Hz: -48 dB/octave rolloff, so 60 Hz
    is at ~-32 dB and 30 Hz at ~-56 dB. Loses the very bottom of male voice
    fundamentals (typical adult male 85-180 Hz; we sacrifice 85-100 Hz),
    but speech intelligibility lives in the formants (300-3500 Hz), so the
    model still has plenty to work with - and the gain in mic-noise
    rejection is large.

    Plus optional notch filters at 50 Hz and 60 Hz, killing mains hum
    that bleeds into low-end harmonics. Notches are narrow (Q=30) so they
    don't audibly affect anything but the offending frequencies.

    CAUSAL single-pass (``scipy.signal.sosfilt``), initialized with the
    steady-state ``sosfilt_zi * x[0]`` so there is no startup DC step. This
    is the key property that makes real-time streaming possible: a causal
    filter's output for a given sample never changes once later samples
    arrive, so a streaming deployment can filter each new chunk once (with
    retained state via :class:`StreamingHighpass`) and never recompute old
    audio. Filtering a stream chunk-by-chunk with retained state is
    bit-identical (it is an LTI system) to one call of this function over
    the concatenated signal - so a model trained on clips filtered here
    sees consistent features when deployed with the streaming filter.

    (The previous implementation used zero-phase ``sosfiltfilt``, which runs
    the filter forwards *and* backwards; the backward pass makes every output
    sample depend on all future samples, which is impossible to stream and
    forced the JS preprocessor to recompute the entire 1-s window every hop.)
    """
    if audio.numel() < 1:
        return audio
    flat = audio.flatten().detach().cpu().numpy().astype(np.float64)
    try:
        sos_hpf = _hpf_coefficients(cutoff_hz, sample_rate, order)
        zi = scipy.signal.sosfilt_zi(sos_hpf)
        flat, _ = scipy.signal.sosfilt(sos_hpf, flat, zi=zi * flat[0])
        if apply_mains_notch:
            # Both 50 Hz (Europe) and 60 Hz (Americas) - we don't know which.
            # Applying both is cheap and harmless: each is a narrow notch.
            for f in (50.0, 60.0):
                sos_notch = _notch_coefficients(f, q=30.0, sample_rate=sample_rate)
                zin = scipy.signal.sosfilt_zi(sos_notch)
                flat, _ = scipy.signal.sosfilt(sos_notch, flat, zi=zin * flat[0])
    except ValueError:
        return audio  # falls through on any unexpected scipy edge case
    flat = np.ascontiguousarray(flat.astype(np.float32))
    out = torch.from_numpy(flat)
    return out if audio.ndim == 1 else out.view_as(audio)


class StreamingHighpass:
    """Causal, stateful counterpart to :func:`highpass_filter` for real-time
    streaming inference.

    Holds the per-section biquad state across calls, so each incoming audio
    chunk is filtered exactly once and old audio is never re-touched. The
    output is bit-identical (an LTI cascade) to calling ``highpass_filter``
    on the concatenation of all chunks seen so far - so it stays consistent
    with the offline filtering used to prepare training clips.

    This is the exact algorithm mirrored in the JS demos
    (``examples/*/preprocessing.js``); keeping the two in lock-step is what
    lets the browser / React-Native runtimes match Python bit-for-bit.
    """

    def __init__(
        self,
        cutoff_hz: float = 100.0,
        sample_rate: int = SAMPLE_RATE,
        order: int = 8,
        apply_mains_notch: bool = True,
    ) -> None:
        self._sos = [_hpf_coefficients(cutoff_hz, sample_rate, order)]
        if apply_mains_notch:
            self._sos += [
                _notch_coefficients(f, q=30.0, sample_rate=sample_rate)
                for f in (50.0, 60.0)
            ]
        # Unit steady-state initial conditions per stage; scaled by each
        # stage's first input sample on the first chunk (matches scipy's
        # sosfilt_zi * x[0] convention used in highpass_filter above).
        self._zi_unit = [scipy.signal.sosfilt_zi(sos) for sos in self._sos]
        self._zi: list[Optional[np.ndarray]] = [None] * len(self._sos)

    def reset(self) -> None:
        self._zi = [None] * len(self._sos)

    def __call__(self, chunk: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        is_tensor = isinstance(chunk, torch.Tensor)
        x = (chunk.detach().cpu().numpy() if is_tensor else np.asarray(chunk))
        x = x.astype(np.float64).flatten()
        if x.size == 0:
            return chunk
        for i, sos in enumerate(self._sos):
            if self._zi[i] is None:
                self._zi[i] = self._zi_unit[i] * x[0]
            x, self._zi[i] = scipy.signal.sosfilt(sos, x, zi=self._zi[i])
        x = np.ascontiguousarray(x.astype(np.float32))
        return torch.from_numpy(x) if is_tensor else x


def trim_silence(
    audio: torch.Tensor,
    frame_ms: float = 20.0,
    threshold_db: float = -40.0,
) -> torch.Tensor:
    """Trim leading/trailing silence below threshold (relative to peak)."""
    if audio.numel() == 0:
        return audio
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)
    if frame_len <= 0 or audio.numel() <= frame_len:
        return audio
    pad = (-audio.numel()) % frame_len
    padded = torch.cat([audio, audio.new_zeros(pad)])
    frames = padded.unfold(0, frame_len, frame_len)
    rms = frames.pow(2).mean(dim=-1).sqrt()
    peak_rms = rms.max().clamp(min=1e-9)
    threshold = peak_rms * (10 ** (threshold_db / 20.0))
    voiced = rms > threshold
    if not voiced.any():
        return audio
    idx = voiced.nonzero(as_tuple=False).squeeze(-1)
    start = idx.min().item() * frame_len
    end = (idx.max().item() + 1) * frame_len
    return audio[start:end]


def center_in_window(audio: torch.Tensor, window_samples: int = SAMPLE_RATE) -> torch.Tensor:
    """Place audio in a fixed-length window, centered. Pad with zeros or truncate."""
    n = audio.numel()
    if n == window_samples:
        return audio
    if n < window_samples:
        pad_total = window_samples - n
        left = pad_total // 2
        right = pad_total - left
        return torch.cat([audio.new_zeros(left), audio, audio.new_zeros(right)])
    excess = n - window_samples
    start = excess // 2
    return audio[start : start + window_samples]


def prepare_clip(audio: torch.Tensor, window_samples: int = SAMPLE_RATE) -> torch.Tensor:
    """Normalize, high-pass filter, trim silence, and center in a fixed window."""
    audio = highpass_filter(audio)  # remove sub-80Hz rumble before anything else
    audio = peak_normalize(audio, target_dbfs=-3.0)
    audio = trim_silence(audio)
    if audio.numel() == 0:
        return torch.zeros(window_samples)
    audio = peak_normalize(audio, target_dbfs=-3.0)
    return center_in_window(audio, window_samples)


def end_align_in_window(
    audio: torch.Tensor,
    window_samples: int = SAMPLE_RATE,
    trailing_silence_ms: float = 50.0,
) -> torch.Tensor:
    """Place audio at the right edge of a fixed window with optional trailing
    silence padding to the right (so we don't end exactly at the buffer's last
    sample, which is unnatural). Used to create end-aligned positive variants
    that train the model to fire on phrase-completion at the buffer's right
    edge - matching how the sliding inference buffer captures fresh audio.
    """
    n = audio.numel()
    if n >= window_samples:
        return audio[-window_samples:]
    trailing = min(int(SAMPLE_RATE * trailing_silence_ms / 1000),
                   window_samples - n)
    leading = window_samples - n - trailing
    return torch.cat([audio.new_zeros(leading), audio, audio.new_zeros(trailing)])


def prepare_clip_end_aligned(
    audio: torch.Tensor,
    window_samples: int = SAMPLE_RATE,
    trailing_silence_ms: float = 50.0,
) -> torch.Tensor:
    """prepare_clip variant: HPF + normalize + trim silence + end-align (right edge)."""
    audio = highpass_filter(audio)
    audio = peak_normalize(audio, target_dbfs=-3.0)
    audio = trim_silence(audio)
    if audio.numel() == 0:
        return torch.zeros(window_samples)
    audio = peak_normalize(audio, target_dbfs=-3.0)
    return end_align_in_window(audio, window_samples, trailing_silence_ms)


def random_align_in_window(
    audio: torch.Tensor,
    window_samples: int = SAMPLE_RATE,
    rng: "random.Random | None" = None,
) -> torch.Tensor:
    """Place audio at a RANDOM position within a fixed-length window.

    Streaming inference sees the wake phrase at every alignment as the
    rolling buffer scrolls. Centered + end-aligned training cover the two
    extremes; random-aligned variants fill the in-between space so the
    model isn't pinned to specific positions.
    """
    import random as _r
    rng = rng or _r
    n = audio.numel()
    if n >= window_samples:
        # Random crop instead of forced alignment
        start = rng.randint(0, n - window_samples)
        return audio[start : start + window_samples]
    # Otherwise: pick a random left-pad amount and place the audio
    leading = rng.randint(0, window_samples - n)
    trailing = window_samples - n - leading
    return torch.cat([audio.new_zeros(leading), audio, audio.new_zeros(trailing)])


def prepare_clip_random_aligned(
    audio: torch.Tensor,
    window_samples: int = SAMPLE_RATE,
    rng: "random.Random | None" = None,
) -> torch.Tensor:
    """prepare_clip variant: HPF + normalize + trim silence + random-align.

    Adds variance to where speech appears within the 1-s window. Combined
    with the existing centered + end-aligned variants, the model learns to
    detect the wake phrase regardless of its alignment in the buffer -
    closing the train/infer mismatch from streaming buffers scrolling
    audio through every possible position.
    """
    audio = highpass_filter(audio)
    audio = peak_normalize(audio, target_dbfs=-3.0)
    audio = trim_silence(audio)
    if audio.numel() == 0:
        return torch.zeros(window_samples)
    audio = peak_normalize(audio, target_dbfs=-3.0)
    return random_align_in_window(audio, window_samples, rng=rng)


def load_dir_clips(directory: str | Path, window_samples: int = SAMPLE_RATE) -> list[torch.Tensor]:
    """Load every .wav in a directory, return prepared fixed-length clips."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    clips: list[torch.Tensor] = []
    for p in sorted(directory.glob("*.wav")):
        try:
            audio = load_wav(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skipping {p}: {exc}")
            continue
        clips.append(prepare_clip(audio, window_samples))
    return clips
