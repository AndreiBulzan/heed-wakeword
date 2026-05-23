"""Cheap pre-filter: skip the model when the audio is obviously not voice."""

from __future__ import annotations

import math

import torch

from . import SAMPLE_RATE


class EnergyGate:
    """Two-stage rule-based gate. Decides whether to invoke the model on a frame.

    Stage 1: RMS energy threshold. Silence and very quiet audio is skipped.
    Stage 2: spectral-band fraction. The fraction of energy inside the human
             voice band (300-3400 Hz) must exceed `voice_band_min`. This
             rejects pure tones, music, fan noise, and most non-speech.

    Both stages run on a single audio frame (~1 sec). Cost is one FFT and
    a couple of reductions - far cheaper than running the CNN.
    """

    def __init__(
        self,
        energy_dbfs: float = -55.0,
        voice_band_min: float = 0.15,      # was 0.35 → 0.20 → 0.15
        sample_rate: int = SAMPLE_RATE,
        band_lo_hz: float = 100.0,         # was 300; real speech goes lower
        band_hi_hz: float = 7000.0,        # was 3400; real speech goes higher
    ) -> None:
        self.energy_threshold = 10 ** (energy_dbfs / 20.0)
        self.voice_band_min = voice_band_min
        self.sample_rate = sample_rate
        self.band_lo_hz = band_lo_hz
        self.band_hi_hz = band_hi_hz

    def __call__(self, audio: torch.Tensor) -> tuple[bool, dict]:
        """Return (pass_through, diagnostics). diagnostics always includes
        rms, rms_dbfs, band_frac, plus a 3-band breakdown (rumble/voice/hiss)
        so the UI can show exactly where the energy is concentrated."""
        audio = audio.detach()
        if audio.ndim > 1:
            audio = audio.flatten()
        n = audio.numel()
        if n < 64:
            return False, {"reason": "too short", "rms": 0.0, "rms_dbfs": -120.0,
                           "band_frac": 0.0, "rumble_frac": 0.0, "hiss_frac": 0.0}

        rms = float(audio.pow(2).mean().sqrt())
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))

        # Always compute the band fraction so the UI shows it (cheap one FFT)
        window = torch.hann_window(n, dtype=audio.dtype, device=audio.device)
        spec = torch.fft.rfft(audio * window)
        mag = spec.abs().pow(2)
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        total = mag.sum().clamp(min=1e-12)
        # Three-band breakdown: rumble (< band_lo), voice (band_lo .. band_hi),
        # hiss (> band_hi). Sum to 1.0. Useful for spotting "voice band low
        # because rumble high" vs "voice band low because mic hiss" - they
        # have different fixes.
        rumble_mask = freqs < self.band_lo_hz
        voice_mask = (freqs >= self.band_lo_hz) & (freqs <= self.band_hi_hz)
        hiss_mask = freqs > self.band_hi_hz
        rumble_frac = float(mag[rumble_mask].sum() / total)
        band_frac = float(mag[voice_mask].sum() / total)
        hiss_frac = float(mag[hiss_mask].sum() / total)

        diag = {"rms": rms, "rms_dbfs": rms_dbfs, "band_frac": band_frac,
                "rumble_frac": rumble_frac, "hiss_frac": hiss_frac}

        if rms < self.energy_threshold:
            diag["reason"] = "below energy"; return False, diag
        if band_frac < self.voice_band_min:
            diag["reason"] = "out of voice band"; return False, diag
        diag["reason"] = "pass"; return True, diag
