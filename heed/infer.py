"""Streaming inference with sliding window + cheap gate."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from . import HOP_SAMPLES, SAMPLE_RATE, WINDOW_FRAMES
from .audio import StreamingHighpass, log_mel, peak_normalize
from .gate import EnergyGate
from .trainer import load_model


class WakeWordDetector:
    """Sliding-window detector with energy/spectral gate and smoothing.

    Feed audio frame-by-frame via `step()`. Returns the smoothed wake-word
    probability for the current sliding window, and a `triggered` flag.

    Internal flow per step:
      1. Append new audio to ring buffer.
      2. Check the energy + spectral gate on the latest 1-sec window.
         If it fails, return without invoking the model.
      3. Compute log-mel for the window.
      4. Forward through the CNN → raw probability.
      5. Smooth with EMA. Apply two-frames-above-threshold + refractory rule.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        gate: EnergyGate | None = None,
        ema_alpha: float = 0.5,
        consecutive_frames: int = 2,
        refractory_seconds: float = 0.7,
        threshold_override: float | None = None,
        backend: str = "pytorch",
        onnx_path: str | Path | None = None,
    ) -> None:
        """Streaming wake-word detector.

        backend: "pytorch" (default), "onnx_fp32", or "onnx_int8". For ONNX
                 backends, `onnx_path` must point to the exported .onnx file.
                 Threshold + phrase are still read from the PyTorch model.pt
                 sidecar so the two backends can be compared apples-to-apples.
        """
        # Always load the PyTorch sidecar for threshold + phrase + arch info,
        # even when running ONNX (the ONNX file is just weights - the
        # metadata lives in the .pt).
        self.model, payload = load_model(model_path)
        self.threshold = float(
            payload["threshold"] if threshold_override is None else threshold_override
        )
        self.phrase = payload.get("phrase", "")

        self.backend = backend
        self._onnx_session = None
        if backend != "pytorch":
            if onnx_path is None or not Path(onnx_path).exists():
                raise RuntimeError(
                    f"backend={backend!r} but onnx_path is missing or not found "
                    f"({onnx_path}). Run `heed export <project>` first."
                )
            try:
                import onnxruntime as ort
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"backend={backend!r} requires onnxruntime: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            self._onnx_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )

        self.window_samples = SAMPLE_RATE
        # `buffer` holds the FILTERED 1-s window. New audio is high-passed once
        # on arrival by the stateful streaming filter and appended; old audio
        # is never re-filtered. See StreamingHighpass for why this matches the
        # offline highpass_filter used to prepare training clips.
        self.buffer = torch.zeros(self.window_samples)
        self._hpf = StreamingHighpass()
        self.gate = gate or EnergyGate()
        self.ema = 0.0
        self.ema_alpha = ema_alpha
        self.above = 0
        self.consecutive_frames = consecutive_frames
        self.refractory_seconds = refractory_seconds
        self.last_trigger_time = -1e9

    def _forward(self, mel: torch.Tensor) -> float:
        """Run a single forward pass through the chosen backend, return prob."""
        if self.backend == "pytorch":
            with torch.no_grad():
                return float(torch.sigmoid(self.model(mel)).item())
        # ONNX path
        logit = self._onnx_session.run(None, {"mel": mel.numpy()})[0]
        return float(1.0 / (1.0 + np.exp(-float(np.asarray(logit).flatten()[0]))))

    def step(self, audio_chunk: torch.Tensor, now: float | None = None) -> dict:
        """Run the model on the rolling 1-s window. ALWAYS runs the model
        (returns the score even when the gate fails) so the UI can show
        the actual probability for diagnostics. Triggers are still
        gated."""
        if now is None:
            now = time.monotonic()
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk.astype("float32"))
        audio_chunk = audio_chunk.flatten()
        # Causal streaming high-pass: filter ONLY the new samples (retaining
        # biquad state), then append to the rolling 1-s window. Removes sub-
        # 100Hz rumble + 50/60Hz mains hum exactly as prepare_clip does for
        # training data, so train/infer see consistent spectra - but in O(new
        # samples) per step instead of re-filtering the whole window, and
        # without ever changing already-filtered audio.
        filt_new = self._hpf(audio_chunk)
        n = filt_new.numel()
        if n >= self.window_samples:
            self.buffer = filt_new[-self.window_samples :].clone()
        else:
            self.buffer = torch.cat([self.buffer[n:], filt_new])

        # Gate - checked on the FILTERED buffer so voice-band % reflects
        # post-HPF audio (the model's actual input), not raw mic.
        pass_through, diag = self.gate(self.buffer)

        # Short-circuit on gate fail: don't run the model on essentially-
        # silent audio. peak_normalize amplifies the mic-noise floor by
        # 40-60 dB when the buffer is mostly silence, and the model's
        # output on amplified noise is meaningless and confusingly high
        # (0.4-0.8). The gate already determined this isn't real audio;
        # we save the compute AND avoid the misleading prob display.
        # EMA decays gracefully so brief gate flickers during real speech
        # don't reset our state hard.
        if not pass_through:
            self.ema = self.ema * 0.5  # fast decay during silence
            self.above = 0
            return {
                "prob": 0.0,
                "ema": self.ema,
                "triggered": False,
                "gated": True,
                "diag": diag,
            }

        # Gate passed - run the model
        normalized = peak_normalize(self.buffer)
        mel = log_mel(normalized)  # (1, n_mels, T)
        prob = self._forward(mel)

        # EMA smoothing
        self.ema = self.ema_alpha * prob + (1 - self.ema_alpha) * self.ema

        # Trigger on consecutive RAW-prob crossings. The smoothed EMA lags by
        # design, so requiring ema > threshold too suppressed short, confident
        # detections (the word would clear threshold for 2-3 frames while ema
        # was still climbing). The consecutive-frame count is the debounce.
        if prob > self.threshold:
            self.above += 1
        else:
            self.above = 0

        triggered = False
        if (
            self.above >= self.consecutive_frames
            and now - self.last_trigger_time > self.refractory_seconds
        ):
            triggered = True
            self.last_trigger_time = now
            self.above = 0

        return {
            "prob": prob,
            "ema": self.ema,
            "triggered": triggered,
            "gated": False,
            "diag": diag,
        }


def scan_file(
    model_path: str | Path,
    audio: torch.Tensor,
    hop_seconds: float = 0.1,
    threshold_override: float | None = None,
) -> list[dict]:
    """Run the detector over an offline audio tensor. Returns list of frame results."""
    detector = WakeWordDetector(model_path, threshold_override=threshold_override)
    hop = int(hop_seconds * SAMPLE_RATE)
    results = []
    t = 0.0
    for start in range(0, max(1, audio.numel() - hop + 1), hop):
        chunk = audio[start : start + hop]
        res = detector.step(chunk, now=t)
        res["time"] = round(t, 3)
        results.append(res)
        t += hop_seconds
    return results
