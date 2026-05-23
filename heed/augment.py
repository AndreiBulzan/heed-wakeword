"""Signal-processing audio augmentation. No external models.

The point of these augmentations is to simulate the variability across
speakers and environments using only the user's own recordings:
  - pitch shift  : different vocal pitch (higher/lower voice)
  - speed change : different speaking rate
  - VTLP-ish freq warp on mel : different vocal-tract length (formant shift)
  - additive noise : environmental robustness
  - simple synthetic reverb : room acoustics
  - volume jitter : mic-distance / loudness variation

These are not as expressive as a multi-speaker TTS but they substantially
broaden a single speaker's recordings without any external dependencies.
"""

from __future__ import annotations

import math
import random

import numpy as np
import scipy.signal
import torch

from . import SAMPLE_RATE


def speaker_warp(audio: torch.Tensor, factor: float) -> torch.Tensor:
    """VTLP-style simultaneous pitch + formant + speed shift via linear-interp.

    factor > 1 → faster, higher pitch, formants up (shorter vocal tract).
    factor < 1 → slower, lower pitch, formants down (longer vocal tract).

    Uses linear interpolation directly - orders of magnitude faster than
    torchaudio.functional.resample (which recomputes a filter kernel for every
    unique sample-rate pair). The slight aliasing is irrelevant here because
    the downstream mel filter bank low-passes the signal anyway.
    """
    if abs(factor - 1.0) < 1e-3:
        return audio
    if audio.ndim != 1:
        # apply along last dim
        return torch.stack([speaker_warp(a, factor) for a in audio])
    n = audio.numel()
    new_len = max(8, int(n / factor))
    idx = torch.linspace(0, n - 1, new_len, device=audio.device, dtype=audio.dtype)
    lo = idx.long()
    hi = (lo + 1).clamp(max=n - 1)
    frac = idx - lo.float()
    return audio[lo] * (1 - frac) + audio[hi] * frac


def add_noise(audio: torch.Tensor, snr_db: float, noise: torch.Tensor | None = None) -> torch.Tensor:
    """Mix Gaussian (or supplied) noise at the requested SNR (dB)."""
    if audio.numel() == 0:
        return audio
    if noise is None:
        noise = torch.randn_like(audio)
    else:
        # tile / trim noise to match audio length
        n = audio.numel()
        if noise.numel() < n:
            reps = math.ceil(n / noise.numel())
            noise = noise.repeat(reps)[:n]
        else:
            start = random.randint(0, noise.numel() - n)
            noise = noise[start : start + n]
    sig_power = audio.pow(2).mean().clamp(min=1e-12)
    noise_power = noise.pow(2).mean().clamp(min=1e-12)
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = noise * (target_noise_power / noise_power).sqrt()
    return audio + noise


def synthetic_reverb(audio: torch.Tensor, decay: float = 0.3, taps: int = 8) -> torch.Tensor:
    """Lightweight comb-filter reverb. Cheaper than full RIR convolution."""
    if decay <= 0 or taps <= 1:
        return audio
    out = audio.clone()
    for i in range(1, taps):
        delay_samples = int(0.02 * i * SAMPLE_RATE * (1.0 + 0.1 * random.random()))
        gain = decay ** i
        if delay_samples >= out.numel():
            continue
        delayed = torch.zeros_like(out)
        delayed[delay_samples:] = audio[: out.numel() - delay_samples] * gain
        out = out + delayed
    # renormalize to prevent clipping
    peak = out.abs().max().clamp(min=1e-9)
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out


def gain_jitter(audio: torch.Tensor, db_range: float = 6.0) -> torch.Tensor:
    """Random gain in ± db_range dB."""
    db = random.uniform(-db_range, db_range)
    return audio * (10 ** (db / 20.0))


def mel_freq_warp(mel: torch.Tensor, alpha: float) -> torch.Tensor:
    """VTLP-like frequency warp on a log-mel spectrogram.

    Resamples along the frequency axis with factor `alpha`.
    alpha > 1: shift formants down (longer vocal tract)
    alpha < 1: shift formants up (shorter vocal tract / higher voice)
    Approximates VTLP without modifying the audio.
    """
    if abs(alpha - 1.0) < 1e-3:
        return mel
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)
    B, F, T = mel.shape
    # Create a sampling grid along the frequency axis
    src_idx = torch.linspace(0, F - 1, F, device=mel.device)
    sample_pos = src_idx / alpha
    sample_pos = sample_pos.clamp(0, F - 1)
    low = sample_pos.floor().long()
    high = (low + 1).clamp(max=F - 1)
    frac = (sample_pos - low.float()).view(1, F, 1)
    warped = mel[:, low, :] * (1 - frac) + mel[:, high, :] * frac
    if warped.shape[0] == 1:
        warped = warped.squeeze(0)
    return warped


# ----- combined "speaker variation" sampler -----

def augment_clip(
    audio: torch.Tensor,
    *,
    warp_range: tuple[float, float] = (0.85, 1.18),
    noise_snr_range: tuple[float, float] = (5.0, 30.0),
    reverb_prob: float = 0.55,
    gain_db: float = 6.0,
    noise_pool: list[torch.Tensor] | None = None,
    rir_pool: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply a randomized fast augmentation chain.

    Order: warp → RIR/reverb (over warped signal) → noise → gain.
    The warp covers pitch/speed/formant in one resample call.
    If `rir_pool` is provided, use real-RIR convolution (better than the
    comb-filter fallback). If `noise_pool` is provided, sample from it
    instead of using Gaussian noise.
    """
    out = audio.clone()
    if random.random() < 0.95:
        out = speaker_warp(out, random.uniform(*warp_range))
    if random.random() < reverb_prob:
        if rir_pool:
            out = convolve_rir(out, random.choice(rir_pool))
        else:
            out = synthetic_reverb(out, decay=random.uniform(0.1, 0.3),
                                   taps=random.randint(3, 6))
    if random.random() < 0.9:
        snr = random.uniform(*noise_snr_range)
        noise = random.choice(noise_pool) if noise_pool else None
        out = add_noise(out, snr_db=snr, noise=noise)
    if random.random() < 0.7:
        out = gain_jitter(out, db_range=gain_db)
    return out


def random_freq_warp(mel: torch.Tensor, alpha_range: tuple[float, float] = (0.85, 1.15)) -> torch.Tensor:
    """Random VTLP-like warp on mel."""
    alpha = random.uniform(*alpha_range)
    return mel_freq_warp(mel, alpha)


# ===== SpecAugment ==========================================================

def specaugment(
    mel: torch.Tensor,
    n_freq_masks: int = 2,
    freq_mask_width: int = 8,
    n_time_masks: int = 2,
    time_mask_width: int = 12,
    apply_prob: float = 0.7,
) -> torch.Tensor:
    """SpecAugment (Park et al. 2019): frequency and time masking on log-mel.

    Standard trick from ASR; transfers well to KWS. Costs essentially nothing
    and improves robustness consistently because the model is forced not to
    rely on any single time/frequency band.

    Operates on (n_mels, T) or (B, n_mels, T) tensors.
    """
    squeeze_back = False
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)
        squeeze_back = True
    B, F, T = mel.shape
    out = mel.clone()
    fill = float(out.mean())
    for b in range(B):
        for _ in range(n_freq_masks):
            if random.random() < apply_prob:
                w = random.randint(0, freq_mask_width)
                if w > 0 and F > w:
                    s = random.randint(0, F - w)
                    out[b, s : s + w, :] = fill
        for _ in range(n_time_masks):
            if random.random() < apply_prob:
                w = random.randint(0, time_mask_width)
                if w > 0 and T > w:
                    s = random.randint(0, T - w)
                    out[b, :, s : s + w] = fill
    return out.squeeze(0) if squeeze_back else out


# ===== Parametric RIR convolution ==========================================

def generate_parametric_rir(
    rt60: float = 0.4,
    distance_m: float = 1.5,
    sample_rate: int = SAMPLE_RATE,
    seed: int | None = None,
) -> torch.Tensor:
    """Parametric room-impulse-response generator (image-source approximation).

    Returns a 1-D RIR tensor. Convolving a clean signal with this gives a
    realistic-sounding reverb - orders of magnitude more useful than the
    comb-filter we used before, and 20% relative accuracy gain in published
    KWS benchmarks.

    Args:
        rt60: reverberation time (s) - time for the energy to decay 60 dB.
              Small rooms ≈ 0.2-0.4, big rooms ≈ 0.5-1.0, large halls > 1.0.
        distance_m: distance from source to microphone (m).
    """
    rng = np.random.default_rng(seed)
    n = int(rt60 * 1.5 * sample_rate)
    if n < 32:
        n = 32
    rir = np.zeros(n, dtype=np.float32)

    # Direct sound: delta at distance/c
    direct = int(distance_m / 343.0 * sample_rate)
    if direct < n:
        rir[direct] = 1.0

    # Early reflections - 5-15 sparse taps within ~70 ms of direct
    n_early = int(rng.integers(5, 16))
    early_window_end = direct + int(0.07 * sample_rate)
    for _ in range(n_early):
        lo = direct + int(0.005 * sample_rate)
        hi = min(early_window_end, n - 1)
        if hi <= lo:
            break
        t = int(rng.integers(lo, hi))
        rir[t] += float(rng.uniform(0.1, 0.45) * (1.0 if rng.random() < 0.5 else -1.0))

    # Late reverb: exponentially decaying noise tail
    if early_window_end < n:
        tail_len = n - early_window_end
        decay = np.exp(-3.0 * np.log(10.0) * np.arange(tail_len) / max(1.0, rt60 * sample_rate))
        noise = rng.standard_normal(tail_len).astype(np.float32) * 0.25
        rir[early_window_end:] += (noise * decay).astype(np.float32)

    peak = float(np.abs(rir).max())
    if peak > 1e-9:
        rir = rir / peak
    return torch.from_numpy(rir.astype(np.float32))


def convolve_rir(audio: torch.Tensor, rir: torch.Tensor,
                 preserve_length: bool = True) -> torch.Tensor:
    """Fast FFT convolution of audio with an RIR. Preserves original length.

    Renormalizes the output to the input's original peak so the loudness
    stays stable - this matters because downstream `peak_normalize` runs
    before the model, so unrescaled reverb-amplified clips would be normalized
    differently from clean ones.
    """
    if rir.numel() < 2:
        return audio
    n = audio.numel()
    audio_np = audio.detach().cpu().numpy().astype(np.float32)
    rir_np = rir.detach().cpu().numpy().astype(np.float32)
    full = scipy.signal.fftconvolve(audio_np, rir_np, mode="full")
    if preserve_length:
        full = full[:n]
    in_peak = float(np.abs(audio_np).max())
    out_peak = float(np.abs(full).max())
    if in_peak > 0 and out_peak > 0:
        full = full * (in_peak * 0.95 / out_peak)
    return torch.from_numpy(full.astype(np.float32))


def build_rir_pool(n_rirs: int = 40, seed: int = 0) -> list[torch.Tensor]:
    """Pre-generate a diverse pool of RIRs spanning room sizes and distances."""
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(n_rirs):
        rt60 = float(rng.uniform(0.15, 0.85))           # small room → medium
        distance = float(rng.uniform(0.3, 4.0))         # close-up → far field
        pool.append(generate_parametric_rir(rt60=rt60, distance_m=distance,
                                            seed=int(rng.integers(0, 2**31 - 1))))
    return pool


# ===== Parametric noise pool ==============================================

def _gen_noise(kind: str, n: int, rng: np.random.Generator) -> torch.Tensor:
    """Generate one noise sample of the given kind, length n."""
    if kind == "white":
        x = rng.standard_normal(n).astype(np.float32) * 0.30
    elif kind == "pink":
        # Voss-McCartney-ish: filter white through a 4-pole approx of 1/f
        white = rng.standard_normal(n).astype(np.float32)
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float32)
        a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400], dtype=np.float32)
        x = scipy.signal.lfilter(b, a, white).astype(np.float32) * 0.30
    elif kind == "brown":
        white = rng.standard_normal(n).astype(np.float32)
        x = np.cumsum(white).astype(np.float32) * 0.008
        # detrend (subtract mean) to avoid DC drift
        x = (x - x.mean()).astype(np.float32)
        # normalize to reasonable amplitude
        peak = float(np.abs(x).max())
        if peak > 0:
            x = x * (0.25 / peak)
    elif kind == "hum":
        # mains hum at 50 or 60 Hz + harmonics
        t = np.arange(n).astype(np.float32) / SAMPLE_RATE
        fund = float(rng.choice([50.0, 60.0]))
        x = (np.sin(2 * np.pi * fund * t)
             + 0.30 * np.sin(2 * np.pi * fund * 2 * t)
             + 0.10 * np.sin(2 * np.pi * fund * 3 * t)).astype(np.float32)
        x += rng.standard_normal(n).astype(np.float32) * 0.05
        x = (x * 0.15).astype(np.float32)
    elif kind == "fan":
        # broadband filtered noise (HVAC / fan)
        white = rng.standard_normal(n).astype(np.float32)
        sos = scipy.signal.butter(4, [80.0, 1500.0], btype="bandpass",
                                  fs=SAMPLE_RATE, output="sos")
        x = scipy.signal.sosfilt(sos, white).astype(np.float32) * 0.35
    elif kind == "babble":
        # sum of a few low-frequency sinusoids + noise (background chatter)
        t = np.arange(n).astype(np.float32) / SAMPLE_RATE
        x = np.zeros(n, dtype=np.float32)
        for _ in range(int(rng.integers(3, 7))):
            freq = float(rng.uniform(110.0, 750.0))
            x += np.sin(2 * np.pi * freq * t + float(rng.uniform(0, 2 * np.pi))).astype(np.float32) \
                 * float(rng.uniform(0.04, 0.13))
        x += rng.standard_normal(n).astype(np.float32) * 0.08
    else:  # fallback to white
        x = rng.standard_normal(n).astype(np.float32) * 0.30
    return torch.from_numpy(x)


# ===== Spectral envelope matching (light-weight "voice conversion") =========
#
# Full voice conversion (KNN-VC, FreeVC, OpenVoice) needs a 400 MB WavLM model,
# torchaudio, and meaningful GPU time. Most of the win for our problem comes
# from matching the *spectral envelope* of TTS samples to the user's mic +
# room profile - i.e. apply an EQ filter so synthetic clips have the same
# per-frequency power distribution as the real user recordings. That closes
# most of the synth-real gap structurally, in <1 ms per clip, with no deps.

def compute_spectral_envelope(audios: list[torch.Tensor], n_fft: int = 512
                              ) -> torch.Tensor | None:
    """Mean-of-magnitude power spectrum across clips. Returns (n_fft//2+1,) or None."""
    if not audios:
        return None
    window = torch.hann_window(n_fft)
    spectra = []
    for audio in audios:
        if audio.numel() < n_fft:
            continue
        a = audio if audio.ndim == 1 else audio.flatten()
        stft = torch.stft(a.unsqueeze(0), n_fft=n_fft, hop_length=n_fft // 4,
                          window=window, return_complex=True, center=True,
                          pad_mode="reflect")
        mag = stft.abs().squeeze(0)  # (F, T)
        spectra.append(mag.mean(dim=1))
    if not spectra:
        return None
    return torch.stack(spectra).mean(dim=0)  # (F,)


def compute_transfer_eq(source_env: torch.Tensor, target_env: torch.Tensor,
                        strength: float = 0.6,
                        max_gain: float = 2.0,
                        min_gain: float = 0.5) -> torch.Tensor:
    """EQ curve that maps audio with source_env's spectrum toward target_env.

    strength=0 → identity (no change). strength=1 → full equalization.
    Clamped to [min_gain, max_gain] (default [0.5, 2.0], i.e. ±6 dB) so a
    pathological user-mic profile (very narrow voice band, lots of low-
    frequency rumble, etc.) can't pull TTS into an extreme acoustic
    distribution that the user's *deployment-time* mic won't match.
    """
    ratio = target_env / source_env.clamp(min=1e-9)
    eq = ratio.pow(strength)
    return eq.clamp(min_gain, max_gain)


def apply_eq(audio: torch.Tensor, eq_gain: torch.Tensor,
             n_fft: int = 512) -> torch.Tensor:
    """Apply per-frequency-bin EQ to audio via STFT/iSTFT round-trip."""
    if eq_gain.numel() != n_fft // 2 + 1:
        return audio
    a = audio if audio.ndim == 1 else audio.flatten()
    window = torch.hann_window(n_fft)
    stft = torch.stft(a.unsqueeze(0), n_fft=n_fft, hop_length=n_fft // 4,
                      window=window, return_complex=True, center=True,
                      pad_mode="reflect")
    stft = stft * eq_gain.view(1, -1, 1)
    out = torch.istft(stft, n_fft=n_fft, hop_length=n_fft // 4,
                      window=window, length=a.numel())
    return out.squeeze(0)


def build_noise_pool(n_samples: int = 30, duration_s: float = 1.5,
                     seed: int = 0) -> list[torch.Tensor]:
    """Pre-generate a diverse parametric noise pool. Categories cover the
    main real-world failure-mode noises: white hiss, pink/brown rumble,
    mains hum (electrical), filtered broadband (HVAC/fan), and babble."""
    rng = np.random.default_rng(seed)
    kinds = ["white", "pink", "brown", "hum", "fan", "babble"]
    pool: list[torch.Tensor] = []
    n = int(duration_s * SAMPLE_RATE)
    for i in range(n_samples):
        kind = kinds[i % len(kinds)]
        pool.append(_gen_noise(kind, n, rng))
    return pool
