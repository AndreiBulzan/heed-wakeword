"""Multi-speaker TTS augmentation via Piper-TTS (MIT licensed).

The LibriTTS-R medium voice model contains 904 speakers. We sample voices
uniformly at random, perturb length_scale (speech rate) and the two
noise scales, and feed the resulting clips into the training set as
additional positives - or, if the user supplies distractor phrases, as
hard negatives.

This is the single most impactful addition for cross-speaker generalization
when you only have one trainer's recordings. With TTS on, the model sees
~500-2000 voices saying the wake phrase before deployment.
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


DEFAULT_VOICE_NAME = "en_US-libritts_r-medium"
DEFAULT_VOICE_DIR = Path.home() / ".heed" / "voices"

# Three speakers that are NEVER used at training time. They become the
# user's automatic "did my model only learn me, or does it generalize?"
# checkpoint - after training, the UI synthesizes the wake phrase + a few
# false-trigger phrases from these voices and runs the model on them.
# IDs picked to span the LibriTTS-R speaker range (0-903).
HELDOUT_SPEAKER_IDS: tuple[int, ...] = (47, 451, 832)


# Common short companions to wake-word prefixes ("hey X", "hi X", "ok X"
# usually have one of these as X in everyday speech). When the user's wake
# phrase is "hey jasper", these become "hey siri", "hey john", etc., which
# are exactly the false-trigger phrases that broke the model in practice.
# Curated for PHONETIC DIVERSITY - covers a wide range of ending phonemes
# so for any reasonable wake phrase, many of these end up being near-rhymes
# (e.g. for "hey doc": dog, john, rock, hot, fox, jog, top, mom all become
# critical hard negatives). Without this diversity, the model trained only
# on "hey siri / google / alexa" knew about marketing wake words but had
# no signal against "hey [any short English word]".
_PREFIX_COMPANION_WORDS = [
    # NAMES FIRST. Short common names cover the same vowel structures as
    # most wake-phrase tails ("don/ron/john" share /ɒ/ with "doc") but
    # differ on the final consonant - these are vowel-rhymes, useful for
    # teaching the model that the final phoneme matters. Names are also
    # the most common real-world false-trigger phrase ("hey John!" said
    # to a friend).
    "john", "mike", "sam", "max", "tom", "dave", "paul", "ben",
    "don", "ron", "dawn", "ann", "dan", "stan", "sean", "kim",
    "joe", "jake", "kate", "lisa", "jane", "anna", "mark", "alex",
    # General "hey X" pronouns and short words (different consonant
    # structure from typical wake-word tails - safe distractors).
    "gone", "on", "off", "out", "in", "up",
    "there", "now", "you", "buddy", "friend", "guys", "everybody",
    "everyone", "man", "girl", "boy", "kid", "dude", "honey",
    "watch", "listen", "look", "hold on", "wait", "again", "today",
    "yesterday", "soon", "later", "really", "sure",
    # Animals / objects / verbs / adjectives - broad phonetic coverage,
    # mostly far from any specific wake-word's tail phoneme.
    "cat", "fox", "bird", "fish", "cow", "pig",
    "bear", "mouse", "horse", "owl", "rat",
    "door", "chair", "book", "lamp", "phone", "key", "pen", "ball",
    "cup", "box", "bag", "light", "screen",
    "come", "run", "jump", "sit", "sleep", "eat", "drink", "talk",
    "walk", "think", "work", "play", "drive", "read", "ride",
    "cold", "big", "small", "quick", "slow", "dark", "soft",
    "hard", "warm", "cool", "loud", "quiet", "tall", "short",
    "oh", "ah", "hm", "eh", "ow", "oops", "wow", "whoa", "yeah",
    "no", "ok", "well", "yo", "hi",
    # Smart-speaker names (legacy - distinct from most wake words)
    "siri", "google", "alexa", "cortana",
    # NB: explicit same-final-consonant rhyme clusters (dock/lock/rock for
    # /-ok/, dog/fog/log for /-og/, hot/dot/lot for /-ot/, etc.) were
    # REMOVED here on purpose. Per hard-negative-mining research, training
    # against near-homophones collapses the decision boundary and reduces
    # positive confidence on a 10K-param model. Vowel-rhymes (don/gone/
    # john/ron) provide useful gradient; full-phoneme-rhymes do not.
]

# Common alternate prefixes (these replace "hey" in "hey jasper" → "hi jasper",
# "say jasper", etc.) - forces the model to also discriminate the prefix word.
_ALTERNATE_PREFIXES = [
    "hi", "hello", "say", "call", "tell", "ask", "find", "yo", "way",
    "may", "they", "see", "let", "give", "show",
]


def phonetic_neighbor_distractors(phrase: str, *, max_neighbors: int = 30) -> list[str]:
    """Generate phonetic neighbors of a wake phrase that are likely to cause
    false triggers if not present as hard negatives.

    For "hey jasper" this returns ~30 phrases like:
        "hey siri", "hey google", "hey there", ... (same prefix, different X)
        "hi jasper", "say jasper", "yo jasper", ... (different prefix, same X)
    Plus a couple of single-word distractors that are subwords of the phrase.
    """
    phrase = phrase.strip().lower()
    parts = phrase.split()
    if not parts:
        return []
    out: list[str] = []

    # Case 1: multi-word wake phrase like "hey jasper"
    if len(parts) >= 2:
        # SUBWORD NEGATIVES FIRST. Single words from the phrase, isolated,
        # are by far the highest-value phonetic neighbors - "hey" alone and
        # "doc" alone are the most likely sources of false triggers
        # ("hey" looks like the prefix of "hey doc" with nothing after).
        # Putting them first guarantees they survive the max_neighbors cap.
        for w in parts:
            if len(w) >= 2 and w not in out:
                out.append(w)
        first = parts[0]
        rest = " ".join(parts[1:])
        # "first companion" - keep prefix, vary suffix
        for w in _PREFIX_COMPANION_WORDS:
            cand = f"{first} {w}"
            if cand != phrase and cand not in out:
                out.append(cand)
        # "alt-prefix + rest" - vary prefix, keep suffix
        for p in _ALTERNATE_PREFIXES:
            cand = f"{p} {rest}"
            if cand != phrase and cand not in out:
                out.append(cand)

    # Case 2: single-word wake phrase like "computer" - vary by common context
    else:
        word = parts[0]
        for ctx in ["the", "my", "a", "this", "that", "his", "her"]:
            out.append(f"{ctx} {word}")
        for tail in ["please", "now", "okay", "actually"]:
            out.append(f"{word} {tail}")

    return out[:max_neighbors]


# Pool of short, common phrases used as automatic hard negatives. These have
# the typical "hey/short-word + short-word" rhythm and 1-2-syllable structure
# that wake words tend to share, so they're the phrases most likely to cause
# false triggers in practice. Synthesizing them across many voices teaches
# the model to discriminate by content, not by rhythm.
DEFAULT_NEAR_DISTRACTOR_PHRASES = [
    "hey there",
    "say hello",
    "okay then",
    "alright now",
    "see you later",
    "tell me more",
    "good morning",
    "good evening",
    "no thank you",
    "what time is it",
    "be right back",
    "let me check",
    "play some music",
    "thanks again",
    "I don't think so",
    "are you sure",
    "open the door",
    "close the window",
    "turn it on",
    "turn it off",
]

# Hugging Face URLs (rhasspy/piper-voices, mirror of the canonical Piper voices)
_VOICE_URL_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"
)
VOICE_URLS = {
    "en_US-libritts_r-medium": (
        f"{_VOICE_URL_BASE}/libritts_r/medium/en_US-libritts_r-medium.onnx",
        f"{_VOICE_URL_BASE}/libritts_r/medium/en_US-libritts_r-medium.onnx.json",
    ),
}


# ----- model management ----------------------------------------------------


def voice_paths(name: str = DEFAULT_VOICE_NAME, voice_dir: Path | None = None
                ) -> tuple[Path, Path]:
    voice_dir = voice_dir or DEFAULT_VOICE_DIR
    onnx = voice_dir / f"{name}.onnx"
    cfg = voice_dir / f"{name}.onnx.json"
    return onnx, cfg


def is_voice_available(name: str = DEFAULT_VOICE_NAME,
                       voice_dir: Path | None = None) -> bool:
    onnx, cfg = voice_paths(name, voice_dir)
    return onnx.exists() and cfg.exists()


def download_voice(
    name: str = DEFAULT_VOICE_NAME,
    voice_dir: Path | None = None,
    progress_fn=print,
) -> tuple[Path, Path]:
    """Download a Piper voice model + config into voice_dir."""
    voice_dir = voice_dir or DEFAULT_VOICE_DIR
    voice_dir.mkdir(parents=True, exist_ok=True)
    if name not in VOICE_URLS:
        raise ValueError(
            f"unknown voice {name!r}. Known: {list(VOICE_URLS)}"
        )
    onnx_url, cfg_url = VOICE_URLS[name]
    onnx_path, cfg_path = voice_paths(name, voice_dir)

    if not cfg_path.exists():
        progress_fn(f"downloading config → {cfg_path}")
        _download(cfg_url, cfg_path, progress_fn)
    else:
        progress_fn(f"config already at {cfg_path}")
    if not onnx_path.exists():
        progress_fn(f"downloading model  → {onnx_path}  (~75 MB)")
        _download(onnx_url, onnx_path, progress_fn)
    else:
        progress_fn(f"model already at  {onnx_path}")
    return onnx_path, cfg_path


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


# ----- synthesis ----------------------------------------------------------


def _import_piper():
    """Import piper-tts. Raises RuntimeError with diagnostic info on failure."""
    try:
        from piper.voice import PiperVoice
        from piper.config import SynthesisConfig
        return PiperVoice, SynthesisConfig
    except Exception as exc:  # noqa: BLE001 - really catch anything
        import sys
        underlying = f"{type(exc).__name__}: {exc}"
        msg = (
            f"piper-tts is not usable from this Python.\n"
            f"  underlying error: {underlying}\n"
            f"  python:           {sys.executable}\n"
            f"  sys.prefix:       {sys.prefix}\n"
            f"\n"
            f"  → install / re-install with this exact interpreter:\n"
            f"      {sys.executable} -m pip install piper-tts\n"
            f"\n"
            f"  If it's already installed but failing to import (DLL load "
            f"error, onnxruntime mismatch, etc.), run `heed doctor` for "
            f"a full diagnostic."
        )
        raise RuntimeError(msg) from exc


def is_piper_importable() -> bool:
    """Return True iff piper-tts can be imported in the current process."""
    try:
        _import_piper()
        return True
    except Exception:
        return False


_VOICE_CACHE: dict[str, object] = {}


def _load_voice(name: str = DEFAULT_VOICE_NAME,
                voice_dir: Path | None = None,
                use_cuda: bool | None = None):
    onnx_path, _ = voice_paths(name, voice_dir)
    if not is_voice_available(name, voice_dir):
        raise RuntimeError(
            f"voice {name!r} not found at {onnx_path.parent}. "
            f"Run `heed download-tts` first."
        )
    if use_cuda is None:
        # auto-detect
        try:
            import torch
            use_cuda = bool(torch.cuda.is_available())
        except Exception:
            use_cuda = False
    cache_key = f"{onnx_path}|cuda={use_cuda}"
    if cache_key in _VOICE_CACHE:
        return _VOICE_CACHE[cache_key]
    PiperVoice, _ = _import_piper()
    actual_device = "cuda" if use_cuda else "cpu"
    try:
        voice = PiperVoice.load(str(onnx_path), use_cuda=use_cuda)
    except Exception as exc:
        if use_cuda:
            # graceful fallback to CPU if CUDA load fails (missing onnxruntime-gpu, etc.)
            print(f"[tts] piper CUDA load failed ({exc}); falling back to CPU")
            voice = PiperVoice.load(str(onnx_path), use_cuda=False)
            cache_key = f"{onnx_path}|cuda=False"
            actual_device = "cpu"
        else:
            raise
    # Confirm which providers the underlying ONNX session is actually using -
    # ort.get_available_providers() lists what's installed but only the first
    # provider that gets attached is used at runtime. With onnxruntime (CPU)
    # installed alongside torch+CUDA, use_cuda=True silently no-ops to CPU,
    # which is the failure mode behind 6-10x slower synthesis on RTX 4090s.
    try:
        sess = getattr(voice, "session", None)
        if sess is not None and hasattr(sess, "get_providers"):
            providers = sess.get_providers()
            print(f"[tts] piper loaded on {actual_device} "
                  f"(onnxruntime providers: {providers})")
        else:
            print(f"[tts] piper loaded on {actual_device}")
    except Exception:
        print(f"[tts] piper loaded on {actual_device}")
    _VOICE_CACHE[cache_key] = voice
    return voice


def n_speakers(name: str = DEFAULT_VOICE_NAME) -> int:
    voice = _load_voice(name)
    cfg = getattr(voice, "config", None)
    if cfg is None:
        return 1
    return int(getattr(cfg, "num_speakers", 1) or 1)


def _resample_np(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g = math.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    return scipy.signal.resample_poly(audio, up, down).astype(np.float32)


def synthesize_phrase(
    text: str,
    n_samples: int,
    *,
    name: str = DEFAULT_VOICE_NAME,
    voice_dir: Path | None = None,
    length_scale_range: tuple[float, float] = (0.85, 1.18),
    noise_scale_range: tuple[float, float] = (0.5, 0.85),
    noise_w_range: tuple[float, float] = (0.5, 0.9),
    seed: int = 0,
    progress_fn=None,
    use_cuda: bool | None = None,
    exclude_speakers: tuple[int, ...] = HELDOUT_SPEAKER_IDS,
) -> list[torch.Tensor]:
    """Synthesize n_samples clips of `text` across random speakers.

    Held-out speaker IDs (default: HELDOUT_SPEAKER_IDS) are skipped so the
    user can later test the trained model against voices it has *never*
    seen - the cleanest objective signal for "does this generalize?"
    """
    if n_samples <= 0:
        return []
    _, SynthesisConfig = _import_piper()
    voice = _load_voice(name, voice_dir, use_cuda=use_cuda)
    n_spk = n_speakers(name)
    rng = random.Random(seed)
    out: list[torch.Tensor] = []
    eligible = [sid for sid in range(n_spk) if sid not in set(exclude_speakers)]

    for i in range(n_samples):
        sid = rng.choice(eligible)
        syn = SynthesisConfig(
            speaker_id=sid,
            length_scale=rng.uniform(*length_scale_range),
            noise_scale=rng.uniform(*noise_scale_range),
            noise_w_scale=rng.uniform(*noise_w_range),
        )
        chunks = list(voice.synthesize(text, syn))
        if not chunks:
            continue
        audio = np.concatenate([c.audio_float_array for c in chunks])
        src_sr = chunks[0].sample_rate
        if src_sr != SAMPLE_RATE:
            audio = _resample_np(audio, src_sr, SAMPLE_RATE)
        out.append(torch.from_numpy(audio.astype(np.float32)))
        if progress_fn and (i + 1) % 50 == 0:
            progress_fn(f"  synthesized {i + 1} / {n_samples}")
    return out


_CACHE_MANIFEST_NAME = "tts_cache.json"


def _build_piper_cache_key(
    text: str, n_samples: int, name: str,
    length_scale_range: tuple[float, float],
    noise_scale_range: tuple[float, float],
    noise_w_range: tuple[float, float],
    seed: int,
    exclude_speakers: tuple[int, ...],
) -> dict:
    """Stable serializable representation of every input that affects the
    generated audio. Cache invalidates iff this object differs from the one
    saved alongside the cached WAVs.
    """
    return {
        "engine": "piper",
        "text": text,
        "voice_name": name,
        "n_samples": n_samples,
        "length_scale_range": list(length_scale_range),
        "noise_scale_range": list(noise_scale_range),
        "noise_w_range": list(noise_w_range),
        "seed": seed,
        "exclude_speakers": list(exclude_speakers),
    }


def _diff_keys(a: dict, b: dict) -> list[str]:
    """Field-by-field diff for human-readable cache-stale messages."""
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
    name: str = DEFAULT_VOICE_NAME,
    voice_dir: Path | None = None,
    length_scale_range: tuple[float, float] = (0.85, 1.18),
    noise_scale_range: tuple[float, float] = (0.5, 0.85),
    noise_w_range: tuple[float, float] = (0.5, 0.9),
    seed: int = 0,
    progress_fn=None,
    use_cuda: bool | None = None,
    exclude_speakers: tuple[int, ...] = HELDOUT_SPEAKER_IDS,
    force_regenerate: bool = False,
    clip_prefix: str = "tts_pos",
) -> list[torch.Tensor]:
    """Cache-aware variant of synthesize_phrase.

    Cache layout (per cache_dir):
        tts_cache.json   # exact param manifest of the saved samples
        <prefix>_NNNN.wav x n_samples

    On call:
      * If manifest exists, fields match, and on-disk WAV count == n_samples
        → load and return (no synthesis).
      * Otherwise → wipe stale WAVs, synthesize fresh, write manifest+WAVs,
        return.

    NB: samples saved here are PRE-EQ (raw TTS output). Spectral matching
    is applied downstream in the trainer, so changing user positives
    (which changes the user spectral envelope) does NOT invalidate the
    cache - the same raw samples get a freshly-fitted EQ each train run.
    """
    from .audio import load_wav, save_wav

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / _CACHE_MANIFEST_NAME
    expected_manifest = _build_piper_cache_key(
        text=text, n_samples=n_samples, name=name,
        length_scale_range=length_scale_range,
        noise_scale_range=noise_scale_range,
        noise_w_range=noise_w_range,
        seed=seed, exclude_speakers=tuple(exclude_speakers),
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
                    f"Piper clips ({cache_dir.name}) - skipping synthesis")
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
        length_scale_range=length_scale_range,
        noise_scale_range=noise_scale_range,
        noise_w_range=noise_w_range,
        seed=seed, progress_fn=progress_fn, use_cuda=use_cuda,
        exclude_speakers=exclude_speakers,
    )

    # Persist cache (atomic enough: wipe-then-write of small files)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("*.wav"):
        old.unlink()
    for i, clip in enumerate(clips):
        save_wav(cache_dir / f"{clip_prefix}_{i:04d}.wav", clip)
    manifest_path.write_text(json.dumps(expected_manifest, indent=2))
    log(f"  TTS cache WROTE: {len(clips)} clips → {cache_dir.name}")
    return clips


def synthesize_from_speakers(
    text: str,
    speaker_ids: tuple[int, ...],
    *,
    name: str = DEFAULT_VOICE_NAME,
    voice_dir: Path | None = None,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_w_scale: float = 0.8,
    use_cuda: bool | None = None,
) -> list[torch.Tensor]:
    """Synthesize one clip per requested speaker_id. For held-out evaluation."""
    _, SynthesisConfig = _import_piper()
    voice = _load_voice(name, voice_dir, use_cuda=use_cuda)
    out: list[torch.Tensor] = []
    for sid in speaker_ids:
        syn = SynthesisConfig(
            speaker_id=int(sid),
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )
        chunks = list(voice.synthesize(text, syn))
        if not chunks:
            out.append(torch.zeros(SAMPLE_RATE))
            continue
        audio = np.concatenate([c.audio_float_array for c in chunks])
        src_sr = chunks[0].sample_rate
        if src_sr != SAMPLE_RATE:
            audio = _resample_np(audio, src_sr, SAMPLE_RATE)
        out.append(torch.from_numpy(audio.astype(np.float32)))
    return out
