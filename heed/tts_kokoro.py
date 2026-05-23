"""Second TTS engine: Kokoro (MIT, ~330 MB ONNX, 67+ voices).

Mirrors the public surface of `tts.py` so trainer.py + web.py can use
either engine interchangeably. The whole point of having a *second*
generator is **acoustic diversity** - Piper-LibriTTS-R is one neural family
with shared vocoder artifacts; a model can pass cross-speaker held-out test
on Piper voices while still failing on real humans, because the held-out
voices share the same TTS artifacts as the training voices.

Kokoro voices are listed at:
    https://github.com/thewh1teagle/kokoro-onnx
We use voice **names** (e.g. "af_bella", "bm_george") rather than integer
IDs, matching the Kokoro convention.

Naming convention (kokoro): `<region><gender>_<name>` -
  af_ : American female,  am_ : American male
  bf_ : British female,   bm_ : British male

Held-out picks span gender × region so the cross-TTS test can't be passed
by accidentally generalizing along a single demographic axis.
"""

from __future__ import annotations

import json
import math
import os
import random
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import torch

from . import SAMPLE_RATE


DEFAULT_MODEL_NAME = "kokoro-v1.0"
DEFAULT_VOICE_DIR = Path.home() / ".heed" / "voices" / "kokoro"

# Three voices held out of any future Kokoro training pool, chosen for
# demographic spread (gender × region). The cross-TTS evaluation runs the
# trained model against these so we can see whether the model generalizes
# beyond the Piper-LibriTTS-R acoustic family.
HELDOUT_VOICE_IDS: tuple[str, ...] = ("af_bella", "am_michael", "bm_george")

# A broader pool used when sampling for training augmentation (if/when we
# decide to add Kokoro voices to training). Pruned to voices that exist
# across recent Kokoro releases (v0.19 + v1.0).
KOKORO_VOICES: tuple[str, ...] = (
    # American female
    "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    # American male
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    # British female
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    # British male
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)

# Public download URLs. Pinned to a specific release so we get reproducible
# voice IDs and a known model file size. Update DEFAULT_MODEL_NAME above if
# you bump the version here.
_MODEL_BASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
MODEL_URLS = {
    "kokoro-v1.0": (
        f"{_MODEL_BASE}/kokoro-v1.0.onnx",
        f"{_MODEL_BASE}/voices-v1.0.bin",
    ),
}


# ----- model management ----------------------------------------------------


def voice_paths(name: str = DEFAULT_MODEL_NAME,
                voice_dir: Path | None = None) -> tuple[Path, Path]:
    voice_dir = voice_dir or DEFAULT_VOICE_DIR
    onnx = voice_dir / f"{name}.onnx"
    voices = voice_dir / "voices-v1.0.bin"
    return onnx, voices


def is_voice_available(name: str = DEFAULT_MODEL_NAME,
                       voice_dir: Path | None = None) -> bool:
    onnx, voices = voice_paths(name, voice_dir)
    return onnx.exists() and voices.exists()


def download_voice(
    name: str = DEFAULT_MODEL_NAME,
    voice_dir: Path | None = None,
    progress_fn=print,
) -> tuple[Path, Path]:
    """Download the Kokoro ONNX model + voices file. ~340 MB total."""
    voice_dir = voice_dir or DEFAULT_VOICE_DIR
    voice_dir.mkdir(parents=True, exist_ok=True)
    if name not in MODEL_URLS:
        raise ValueError(
            f"unknown kokoro model {name!r}. Known: {list(MODEL_URLS)}"
        )
    onnx_url, voices_url = MODEL_URLS[name]
    onnx_path, voices_path = voice_paths(name, voice_dir)

    if not voices_path.exists():
        progress_fn(f"downloading voices  → {voices_path}  (~28 MB)")
        _download(voices_url, voices_path, progress_fn)
    else:
        progress_fn(f"voices already at  {voices_path}")
    if not onnx_path.exists():
        progress_fn(f"downloading model   → {onnx_path}  (~310 MB)")
        _download(onnx_url, onnx_path, progress_fn)
    else:
        progress_fn(f"model already at   {onnx_path}")
    return onnx_path, voices_path


def _download(url: str, dest: Path, progress_fn) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _hook(blocks: int, blocksize: int, total: int) -> None:
        if total <= 0:
            return
        done = min(blocks * blocksize, total)
        pct = done * 100 // total
        if blocks % 200 == 0 or done == total:
            progress_fn(f"  {pct:3d}%  {done/1e6:6.1f} / {total/1e6:.1f} MB")

    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    os.replace(tmp, dest)


# ----- import / capability checks ------------------------------------------


def _import_kokoro():
    """Import kokoro-onnx. Raises RuntimeError with diagnostic info on failure."""
    try:
        from kokoro_onnx import Kokoro
        return Kokoro
    except Exception as exc:  # noqa: BLE001 - really catch anything
        import sys
        underlying = f"{type(exc).__name__}: {exc}"
        msg = (
            f"kokoro-onnx is not usable from this Python.\n"
            f"  underlying error: {underlying}\n"
            f"  python:           {sys.executable}\n"
            f"  sys.prefix:       {sys.prefix}\n"
            f"\n"
            f"  → install with this exact interpreter:\n"
            f"      {sys.executable} -m pip install kokoro-onnx\n"
            f"\n"
            f"  Kokoro shares its runtime (onnxruntime) with Piper, so if "
            f"`heed doctor` shows onnxruntime as broken, fixing that fixes "
            f"both engines."
        )
        raise RuntimeError(msg) from exc


def is_kokoro_importable() -> bool:
    """Return True iff kokoro-onnx can be imported in the current process."""
    try:
        _import_kokoro()
        return True
    except Exception:
        return False


_KOKORO_CACHE: dict[str, object] = {}


def _load_kokoro(name: str = DEFAULT_MODEL_NAME,
                 voice_dir: Path | None = None):
    onnx_path, voices_path = voice_paths(name, voice_dir)
    if not is_voice_available(name, voice_dir):
        raise RuntimeError(
            f"kokoro model {name!r} not found at {onnx_path.parent}. "
            f"Run `heed download-kokoro` first."
        )
    cache_key = str(onnx_path)
    if cache_key in _KOKORO_CACHE:
        return _KOKORO_CACHE[cache_key]
    Kokoro = _import_kokoro()
    kokoro = Kokoro(str(onnx_path), str(voices_path))

    # Surface which onnxruntime providers Kokoro actually attached. Same
    # diagnostic as Piper - without this, a CPU-only onnxruntime install
    # silently makes Kokoro 6-10x slower on machines that should be using
    # CUDAExecutionProvider.
    try:
        sess = getattr(kokoro, "sess", None) or getattr(kokoro, "session", None)
        if sess is not None and hasattr(sess, "get_providers"):
            providers = sess.get_providers()
            device = "cuda" if any("CUDA" in p for p in providers) else "cpu"
            print(f"[tts] kokoro loaded on {device} "
                  f"(onnxruntime providers: {providers})")
        else:
            print(f"[tts] kokoro loaded (provider detection unavailable)")
    except Exception:
        pass

    _KOKORO_CACHE[cache_key] = kokoro
    return kokoro


def n_voices() -> int:
    """Return the number of distinct voices available for sampling."""
    return len(KOKORO_VOICES)


# ----- synthesis ----------------------------------------------------------


def _resample_np(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g = math.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    return scipy.signal.resample_poly(audio, up, down).astype(np.float32)


def _kokoro_create(kokoro, text: str, voice: str, speed: float, lang: str
                   ) -> tuple[np.ndarray, int]:
    """Wrap kokoro.create() with version-tolerant arg handling.

    Kokoro-onnx's public API: `create(text, voice, speed=1.0, lang="en-us")`
    returns (samples, sample_rate). Some pinned versions expose only a
    subset - we degrade gracefully so the augmentation pipeline doesn't
    blow up if the user has a slightly older release.
    """
    try:
        samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
    except TypeError:
        # Older API: no `lang` kwarg (defaulted internally to en-us)
        samples, sr = kokoro.create(text, voice=voice, speed=speed)
    return np.asarray(samples, dtype=np.float32), int(sr)


def synthesize_phrase(
    text: str,
    n_samples: int,
    *,
    name: str = DEFAULT_MODEL_NAME,
    voice_dir: Path | None = None,
    speed_range: tuple[float, float] = (0.85, 1.18),
    lang: str = "en-us",
    seed: int = 0,
    progress_fn=None,
    exclude_voices: tuple[str, ...] = HELDOUT_VOICE_IDS,
) -> list[torch.Tensor]:
    """Synthesize `n_samples` clips of `text` across Kokoro voices, balanced.

    Held-out voice IDs (default: HELDOUT_VOICE_IDS) are excluded so the
    cross-TTS holdout test always uses voices the model has never seen
    during training - analogous to the Piper held-out speaker design.

    Voices are assigned **round-robin** rather than uniform-random. With
    only ~24 eligible Kokoro voices and a typical 200-sample request,
    uniform random gives some voices ~5 samples and others ~12 by chance,
    which is too few for the model to disentangle voice from content (we
    saw bm_george score 0.4+ on every distractor for this reason).
    Round-robin guarantees each voice gets floor(n/k) or ceil(n/k) samples,
    minimizing per-voice imbalance.
    """
    if n_samples <= 0:
        return []
    kokoro = _load_kokoro(name, voice_dir)
    rng = random.Random(seed)
    eligible = [v for v in KOKORO_VOICES if v not in set(exclude_voices)]
    if not eligible:
        raise RuntimeError("no eligible Kokoro voices after exclusions")

    # Round-robin assignment: each voice gets n_samples // k or +1.
    # Voice order is randomized once so the +1-sample voices vary by seed.
    voice_order = list(eligible)
    rng.shuffle(voice_order)
    assignments: list[str] = []
    for i in range(n_samples):
        assignments.append(voice_order[i % len(voice_order)])

    out: list[torch.Tensor] = []
    for i, voice in enumerate(assignments):
        speed = rng.uniform(*speed_range)
        try:
            samples, src_sr = _kokoro_create(kokoro, text, voice, speed, lang)
        except Exception as exc:
            if progress_fn:
                progress_fn(f"  [warn] kokoro synth failed for voice={voice!r} "
                            f"text={text!r}: {exc}")
            continue
        if src_sr != SAMPLE_RATE:
            samples = _resample_np(samples, src_sr, SAMPLE_RATE)
        out.append(torch.from_numpy(samples.astype(np.float32)))
        if progress_fn and (i + 1) % 50 == 0:
            progress_fn(f"  synthesized {i + 1} / {n_samples}")
    return out


_CACHE_MANIFEST_NAME = "tts_cache.json"


def _build_kokoro_cache_key(
    text: str, n_samples: int, name: str,
    speed_range: tuple[float, float], lang: str, seed: int,
    exclude_voices: tuple[str, ...],
) -> dict:
    return {
        "engine": "kokoro",
        "text": text,
        "voice_name": name,
        "n_samples": n_samples,
        "speed_range": list(speed_range),
        "lang": lang,
        "seed": seed,
        "exclude_voices": list(exclude_voices),
    }


def _diff_keys(a: dict, b: dict) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append(f"{k}: {a.get(k)!r} → {b.get(k)!r}")
    return out


def synthesize_phrase_with_cache(
    text: str,
    n_samples: int,
    cache_dir: Path,
    *,
    name: str = DEFAULT_MODEL_NAME,
    voice_dir: Path | None = None,
    speed_range: tuple[float, float] = (0.85, 1.18),
    lang: str = "en-us",
    seed: int = 0,
    progress_fn=None,
    exclude_voices: tuple[str, ...] = HELDOUT_VOICE_IDS,
    force_regenerate: bool = False,
    clip_prefix: str = "kokoro_pos",
) -> list[torch.Tensor]:
    """Cache-aware variant of synthesize_phrase for Kokoro.

    See tts.synthesize_phrase_with_cache for the cache contract - same
    layout, same invalidation rules, just keyed on Kokoro params instead
    of Piper params.
    """
    from .audio import load_wav, save_wav

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / _CACHE_MANIFEST_NAME
    expected_manifest = _build_kokoro_cache_key(
        text=text, n_samples=n_samples, name=name,
        speed_range=speed_range, lang=lang, seed=seed,
        exclude_voices=tuple(exclude_voices),
    )

    log = progress_fn or (lambda _msg: None)

    if not force_regenerate and cache_dir.exists() and manifest_path.exists():
        try:
            cached_manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            cached_manifest = None
        if cached_manifest == expected_manifest:
            clip_paths = sorted(cache_dir.glob("*.wav"))
            if len(clip_paths) == n_samples and n_samples > 0:
                log(f"  TTS cache HIT: reusing {len(clip_paths)} cached "
                    f"Kokoro clips ({cache_dir.name}) - skipping synthesis")
                return [load_wav(p) for p in clip_paths]
            log(f"  TTS cache STALE: manifest matched but file count "
                f"{len(clip_paths)} ≠ {n_samples} - regenerating")
        elif cached_manifest is not None:
            diffs = _diff_keys(cached_manifest, expected_manifest)
            preview = ", ".join(diffs[:3]) + ("…" if len(diffs) > 3 else "")
            log(f"  TTS cache STALE ({preview or 'manifest differs'}) - "
                f"regenerating")

    if force_regenerate:
        log("  TTS cache forced regenerate (force_regenerate_tts=True)")

    clips = synthesize_phrase(
        text, n_samples, name=name, voice_dir=voice_dir,
        speed_range=speed_range, lang=lang, seed=seed,
        progress_fn=progress_fn, exclude_voices=exclude_voices,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("*.wav"):
        old.unlink()
    for i, clip in enumerate(clips):
        save_wav(cache_dir / f"{clip_prefix}_{i:04d}.wav", clip)
    manifest_path.write_text(json.dumps(expected_manifest, indent=2))
    log(f"  TTS cache WROTE: {len(clips)} clips → {cache_dir.name}")
    return clips


def synthesize_from_voices(
    text: str,
    voice_ids: tuple[str, ...],
    *,
    name: str = DEFAULT_MODEL_NAME,
    voice_dir: Path | None = None,
    speed: float = 1.0,
    lang: str = "en-us",
) -> list[torch.Tensor]:
    """Synthesize one clip per requested voice_id. For held-out evaluation.

    Returns clips in the same order as `voice_ids`. If a voice fails to
    synthesize, returns a 1-second silent clip in its place so the caller
    can still zip them with the IDs without an alignment bug.
    """
    kokoro = _load_kokoro(name, voice_dir)
    out: list[torch.Tensor] = []
    for vid in voice_ids:
        try:
            samples, src_sr = _kokoro_create(kokoro, text, vid, speed, lang)
            if src_sr != SAMPLE_RATE:
                samples = _resample_np(samples, src_sr, SAMPLE_RATE)
            out.append(torch.from_numpy(samples.astype(np.float32)))
        except Exception:
            out.append(torch.zeros(SAMPLE_RATE))
    return out
