"""Preprocessing invariants. These are the guarantees the whole deploy story
rests on, so they double as executable documentation."""

import numpy as np
import torch

from heed import HOP_SAMPLES, N_FFT, N_MELS, SAMPLE_RATE
from heed.audio import StreamingHighpass, highpass_filter, log_mel


def _tone(freq, n=SAMPLE_RATE):
    t = np.arange(n) / SAMPLE_RATE
    return torch.from_numpy((0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32))


def test_nfft_is_power_of_two():
    # A power-of-two FFT is what keeps the transform a fast radix-2 in every
    # deployment language. Guard against a regression to 400.
    assert N_FFT == 512
    assert N_FFT & (N_FFT - 1) == 0


def test_log_mel_shape():
    mel = log_mel(_tone(440))
    assert mel.shape == (1, N_MELS, SAMPLE_RATE // HOP_SAMPLES + 1)  # (1, 40, 101)
    assert torch.isfinite(mel).all()


def test_cmn_is_zero_mean_per_bin():
    mel = log_mel(_tone(440), apply_cmn=True)
    per_bin_time_mean = mel.mean(dim=-1)
    assert per_bin_time_mean.abs().max().item() < 1e-4


def test_hpf_attenuates_sub_100hz():
    low = highpass_filter(_tone(30))     # below the 100 Hz cutoff
    passband = highpass_filter(_tone(1000))  # well inside the passband
    low_rms = low.pow(2).mean().sqrt().item()
    pass_rms = passband.pow(2).mean().sqrt().item()
    assert pass_rms > 3.0 * low_rms


def test_streaming_filter_equals_oneshot():
    # The streaming high-pass must be an LTI system: filtering chunk by chunk
    # with retained state equals one-shot filtering of the whole signal. This
    # is what lets on-device inference filter each chunk once.
    torch.manual_seed(0)
    audio = _tone(300) + 0.1 * torch.randn(SAMPLE_RATE)
    oneshot = highpass_filter(audio).numpy()
    hp = StreamingHighpass()
    parts = [hp(audio[o:o + 1600]).numpy() for o in range(0, len(audio), 1600)]
    streamed = np.concatenate(parts)
    assert np.abs(streamed - oneshot).max() < 1e-5
