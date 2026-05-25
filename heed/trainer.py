"""Personalization training: one tiny model per wake word, in <60 s on CPU."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import N_MELS, SAMPLE_RATE, WINDOW_FRAMES
from . import SAMPLE_RATE as _SAMPLE_RATE
from .audio import (
    load_dir_clips,
    log_mel,
    prepare_clip,
    prepare_clip_end_aligned,
    prepare_clip_random_aligned,
    save_wav,
    trim_silence,
)
from .augment import (
    apply_eq,
    augment_clip,
    build_noise_pool,
    build_rir_pool,
    compute_spectral_envelope,
    compute_transfer_eq,
    random_freq_warp,
    specaugment,
)
from .model import TinyWakeWordNet, count_parameters


@dataclass
class TrainerConfig:
    phrase: str = ""
    epochs: int = 35
    batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    aug_positives_per_real: int = 40  # for 8 real → 320 augmented
    aug_negatives_per_real: int = 25
    prototype_alpha_range: tuple[float, float] = (0.0, 0.3)
    prototype_apply_prob: float = 0.5
    val_split: float = 0.2
    seed: int = 42
    threshold_target_fpr: float = 0.01
    # TTS augmentation (optional; needs `pip install piper-tts` + voice download).
    tts_positives: int = 0
    tts_negative_phrases: list[str] = field(default_factory=list)
    tts_negatives_per_phrase: int = 20
    tts_voice_name: str = "en_US-libritts_r-medium"
    # Second TTS engine (Kokoro) for cross-TTS-family generalization. The
    # Piper-only model can pass cross-speaker held-out tests against Piper
    # voices it never saw while still failing on Kokoro voices (different
    # acoustic family). Adding Kokoro positives forces the model to learn
    # phrase content independent of TTS-family-specific spectral artifacts.
    # Default 0 = off; recommended ~150-300 when enabled.
    kokoro_positives: int = 0
    kokoro_voice_name: str = "kokoro-v1.0"
    # Device: "auto" (CUDA if available, else CPU), "cuda", "cpu".
    device: str = "auto"
    # Partial-utterance hard negatives: for each positive, split into first
    # half and second half and add both as negatives. Teaches the model to
    # require the FULL phrase (not just "hey" or just "jasper").
    partial_negatives: bool = True
    # SpecAugment (Park et al. 2019): random freq/time masking on log-mel.
    # Cheap, proven KWS win - model can't lean on any one freq/time band.
    specaugment: bool = True
    spec_freq_masks: int = 2
    spec_time_masks: int = 2
    spec_freq_width: int = 8
    spec_time_width: int = 12
    # Real RIR convolution (parametric image-source method) - replaces the
    # comb-filter reverb. ~20% reported KWS accuracy gain.
    use_rir: bool = True
    rir_pool_size: int = 40
    # Parametric noise pool (white/pink/brown/hum/fan/babble). Used when
    # user's ambient is absent or sparse; mixed in alongside it when present.
    use_parametric_noise: bool = True
    parametric_noise_pool_size: int = 30
    # Spectral envelope matching (light-weight voice conversion): apply a
    # per-frequency EQ to TTS samples that pulls their power spectrum toward
    # the user's average mic+room profile. Closes most of the synth-real gap
    # in <1 ms per clip, no extra dependencies. Strength is 0.4: stronger
    # matching can over-correct when the user's mic profile is unusual, pulling
    # TTS into a domain that doesn't match deployment-time mic audio.
    spectral_matching: bool = True
    spectral_matching_strength: float = 0.4  # 0=off, 1=full eq match
    # KNN-VC: full voice conversion via WavLM. Not implemented yet - would
    # need torchaudio + 400 MB checkpoint + GPU. Planned future work.
    voice_conversion: bool = False
    vc_apply_prob: float = 0.6
    # Loss function: "focal" (default - focuses gradient budget on hard
    # examples, helps with class imbalance) or "bce" (plain binary cross-
    # entropy, the previous default, kept available for A/B testing).
    loss_function: str = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # End-aligned positive variants: for each user/TTS positive, add an
    # extra copy where the phrase sits at the right edge of the 1-s window
    # (silence padding on the left). Additive - doesn't replace the centered
    # version. Teaches the model to fire on phrase completion regardless
    # of where it falls in the sliding inference buffer.
    end_aligned_variants: bool = True
    # TTS cache control. When True, ignore any cached samples and resynth
    # everything. Cache normally invalidates automatically when phrase /
    # voice / count / seed / range params change; this flag is for "I want
    # different voices even though all params are the same" - e.g. after
    # bumping the seed mentally but you want fresh randomness anyway, or
    # to verify caching isn't corrupting results.
    force_regenerate_tts: bool = False
    # Model size preset. "small" = current default (channels=32, n_blocks=3,
    # ~10K params, ~41 KB float32). "medium" (channels=64, n_blocks=3,
    # ~27K params) gives more discriminative capacity. "large"
    # (channels=96, n_blocks=4, ~60K) is the upper end for tiny KWS - still
    # sub-250 KB, still sub-ms inference. Larger models help most when the
    # remaining failure mode is "model can't tell phonetically-close
    # phrases apart" rather than "training data is missing".
    model_size: str = "small"


@dataclass
class TrainingArtifact:
    """Everything you need to deploy. Saved next to the .pt."""
    phrase: str
    threshold: float
    train_loss_final: float
    val_accuracy: float
    val_tpr_at_target_fpr: float
    n_positives_real: int
    n_negatives_real: int
    n_positives_aug: int
    n_negatives_aug: int
    n_params: int
    seconds: float
    tts_positives_used: int = 0
    tts_distractor_phrases: int = 0
    kokoro_positives_used: int = 0
    # User-voice-only metric: model evaluated on the user's own positive_*.wav
    # and negative_*.wav (no augmentation, no synthesis). This is the metric
    # that matches the user's subjective "does this trigger on me?" - the
    # validation accuracy is dominated by TTS samples and can be high while
    # the model fails on the trainer's voice.
    user_voice_tpr: float = 0.0
    user_voice_fpr: float = 0.0
    n_user_positives_eval: int = 0
    n_user_negatives_eval: int = 0
    history: list[dict] = field(default_factory=list)


def _make_partial_negatives(clips: list[torch.Tensor]) -> list[torch.Tensor]:
    """For each clip, return its first-half and second-half as separate
    untrimmed audio tensors. After `prepare_clip` these become 1-second
    windows containing only the first or second part of the phrase, which
    act as hard negatives that force the model to require the full phrase.
    """
    out: list[torch.Tensor] = []
    min_voice_samples = int(_SAMPLE_RATE * 0.20)  # 200 ms of speech minimum
    for clip in clips:
        trimmed = trim_silence(clip)
        n = trimmed.numel()
        if n < min_voice_samples:
            continue
        mid = n // 2
        # Take a small overlap so neither half ends mid-phoneme abruptly
        overlap = int(_SAMPLE_RATE * 0.04)  # ~40 ms
        first = trimmed[:max(1, mid - overlap)].clone()
        second = trimmed[min(n, mid + overlap):].clone()
        if first.numel() >= 800:
            out.append(first)
        if second.numel() >= 800:
            out.append(second)
    return out


def _safe_filename(s: str) -> str:
    """Normalize a phrase to a filesystem-friendly directory name."""
    keep = "".join(c if c.isalnum() else "_" for c in s.strip().lower())
    return keep[:60] or "phrase"


def _save_clips(clips: list[torch.Tensor], target_dir: Path, prefix: str) -> None:
    """Wipe + repopulate a directory with the given clips, named <prefix>_NNNN.wav."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for old in target_dir.glob("*.wav"):
        old.unlink()
    for i, clip in enumerate(clips):
        save_wav(target_dir / f"{prefix}_{i:04d}.wav", clip)


def _build_dataset(
    positives: list[torch.Tensor],
    negatives: list[torch.Tensor],
    cfg: TrainerConfig,
    noise_pool: list[torch.Tensor] | None = None,
    log_fn=print,
    tts_cache_dir: Path | None = None,
    rir_pool: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand user clips with audio augmentation, return mel features + labels.

    Returns:
        mels:   (N, n_mels, T) log-mel features
        labels: (N,)
        speaker_prototype: (n_mels,) mean log-mel pooled over negative recordings
                          (the trainer's "voice prototype" - used by the
                          speaker-prototype regularizer).
    """
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    augmented_pos: list[torch.Tensor] = []
    augmented_neg: list[torch.Tensor] = []

    for clip in positives:
        augmented_pos.append(clip)  # always keep original (centered)
        for _ in range(cfg.aug_positives_per_real - 1):
            aug = augment_clip(clip, noise_pool=noise_pool, rir_pool=rir_pool)
            augmented_pos.append(prepare_clip(aug))
        # End-aligned variant: same phrase, just shifted to the right edge of
        # the 1-s window. Teaches the model the trigger position used by the
        # sliding inference buffer at the moment a phrase finishes.
        if cfg.end_aligned_variants:
            augmented_pos.append(prepare_clip_end_aligned(clip))
        # Random-aligned variants. Streaming inference sees the phrase at
        # every possible buffer alignment as audio scrolls in. Centered +
        # end-aligned cover two extremes; 3 random-aligned per positive
        # covers the rest. Critical for closing the train/infer mismatch
        # we measured in user listen logs (probs swinging wildly as the
        # phrase rolls through the buffer).
        for _ in range(3):
            augmented_pos.append(prepare_clip_random_aligned(clip, rng=rng))

    for clip in negatives:
        augmented_neg.append(clip)
        for _ in range(cfg.aug_negatives_per_real - 1):
            aug = augment_clip(clip, noise_pool=noise_pool, rir_pool=rir_pool)
            augmented_neg.append(prepare_clip(aug))

    # Partial-utterance hard negatives: split each positive into its first
    # and second half and add both as negatives, with audio augmentation.
    # Forces the model to discriminate the *whole* phrase, not "hey" or
    # "jasper" alone.
    if cfg.partial_negatives and positives:
        partials = _make_partial_negatives(positives)
        if partials:
            log_fn(f"partial-utterance negatives: {len(partials)} halves "
                   f"(forces full-phrase match)")
            aug_per_partial = max(2, cfg.aug_negatives_per_real // 2)
            for clip in partials:
                prepped = prepare_clip(clip)
                if prepped.abs().max() < 1e-3:
                    continue
                augmented_neg.append(prepped)
                for _ in range(aug_per_partial - 1):
                    aug = augment_clip(prepped, noise_pool=noise_pool, rir_pool=rir_pool)
                    augmented_neg.append(prepare_clip(aug))

    # TTS augmentation: many synthetic speakers saying the same wake phrase.
    # This is the big lever for cross-speaker generalization.
    if cfg.tts_positives > 0 and cfg.phrase:
        from .tts import synthesize_phrase_with_cache  # local import keeps base install slim
        log_fn(f"requesting {cfg.tts_positives} Piper positives of "
               f"{cfg.phrase!r} across multiple speakers…")
        pos_cache_dir = (tts_cache_dir / "positive") if tts_cache_dir else None
        if pos_cache_dir is None:
            # Fallback: cache off if we have no output path. Should not happen
            # in normal flow because trainer always passes a tts_cache_dir.
            from .tts import synthesize_phrase
            tts_clips = synthesize_phrase(
                cfg.phrase, cfg.tts_positives, name=cfg.tts_voice_name,
                seed=cfg.seed, progress_fn=log_fn,
            )
        else:
            tts_clips = synthesize_phrase_with_cache(
                cfg.phrase,
                cfg.tts_positives,
                cache_dir=pos_cache_dir,
                name=cfg.tts_voice_name,
                seed=cfg.seed,
                progress_fn=log_fn,
                force_regenerate=cfg.force_regenerate_tts,
                clip_prefix="tts_pos",
            )
        # Spectral envelope matching: pull TTS audio toward the user's
        # mic+room spectrum. One EQ filter, applied once to every TTS
        # sample BEFORE augmentation. The augmented copies then carry both
        # phonetic content from TTS and spectral profile from the user's
        # actual recording environment.
        if cfg.spectral_matching and positives:
            user_env = compute_spectral_envelope(positives)
            tts_env = compute_spectral_envelope(tts_clips[: min(64, len(tts_clips))])
            if user_env is not None and tts_env is not None:
                # Compute pre-clamp ratio to detect extreme mic profiles
                raw_ratio = (user_env / tts_env.clamp(min=1e-9)).pow(
                    cfg.spectral_matching_strength)
                raw_max, raw_min = float(raw_ratio.max()), float(raw_ratio.min())
                eq = compute_transfer_eq(tts_env, user_env,
                                         strength=cfg.spectral_matching_strength)
                msg = (f"  spectral matching: applying user→TTS EQ "
                       f"(strength={cfg.spectral_matching_strength:.2f}, "
                       f"clamped gain {eq.min():.2f}-{eq.max():.2f})")
                if raw_max > 3.0 or raw_min < 0.33:
                    msg += (f"\n  WARNING: pre-clamp gain range was "
                            f"{raw_min:.2f}-{raw_max:.2f} - your mic profile "
                            f"is unusual. Consider re-recording positives "
                            f"at deployment conditions if test_once fails.")
                log_fn(msg)
                tts_clips = [apply_eq(c, eq) for c in tts_clips]

        # NB: cache already wrote the raw (pre-EQ) clips to tts_cache/positive/
        # via synthesize_phrase_with_cache. We do not re-save the EQ-applied
        # versions, otherwise next run would load EQ-applied as "raw" and
        # apply EQ a second time. Users can inspect tts_cache/positive/ for
        # raw TTS audio; the EQ effect is bounded to ±6 dB by design.

        # Heavier augmentation on TTS samples: they're clean studio voices,
        # but the user's mic-recorded audio has room reverb, mic EQ, and a
        # higher noise floor. Without aggressive augmentation here, the
        # model over-fits to TTS-specific spectral characteristics and
        # fails on real mic input (the synth-real gap).
        for clip in tts_clips:
            prepped = prepare_clip(clip)
            augmented_pos.append(prepped)
            # Add TWO heavily-augmented copies of each TTS sample, to
            # break the TTS-specific feature distribution and push the
            # model toward mic-realistic features.
            for _ in range(2):
                aug = augment_clip(
                    prepped,
                    warp_range=(0.92, 1.10),
                    noise_snr_range=(5.0, 25.0),
                    reverb_prob=0.85,
                    gain_db=8.0,
                    noise_pool=noise_pool,
                    rir_pool=rir_pool,
                )
                augmented_pos.append(prepare_clip(aug))
            # End-aligned variant of each TTS positive.
            if cfg.end_aligned_variants:
                augmented_pos.append(prepare_clip_end_aligned(clip))

        # Partial halves of TTS positives → cross-speaker partial negatives,
        # capped so they don't dominate the negative set. With 400 TTS
        # positives we'd otherwise add 800 partials, which makes the
        # negative class 50% TTS-half - model then learns "TTS-half = no"
        # rather than "phrase content = match".
        if cfg.partial_negatives:
            tts_partials = _make_partial_negatives(tts_clips)
            # Cap to keep TTS partials at most ~25% of negatives. With
            # typical 200-400 normal negatives, that's ~100-150 partials.
            max_tts_partials = max(60, len(augmented_neg) // 2)
            if len(tts_partials) > max_tts_partials:
                tts_partials = rng.sample(tts_partials, max_tts_partials)
            log_fn(f"  → {len(tts_partials)} partial-utterance negatives "
                   f"from TTS (cross-speaker, capped)")
            for clip in tts_partials:
                prepped = prepare_clip(clip)
                if prepped.abs().max() > 1e-3:
                    augmented_neg.append(prepped)

    # Kokoro positives - second TTS family. Cross-TTS generalization test
    # exposed that a Piper-only model has highly Piper-specific positive
    # features (Kokoro held-out scores: 0.05 / 0.24 / 0.63, hugely variable).
    # Adding Kokoro forces phrase-content features that survive the family
    # boundary. Held-out Kokoro voices (HELDOUT_VOICE_IDS) remain excluded
    # so the cross-TTS eval still measures what it measured before.
    if cfg.kokoro_positives > 0 and cfg.phrase:
        from .tts_kokoro import synthesize_phrase_with_cache as kokoro_synthesize
        log_fn(f"requesting {cfg.kokoro_positives} Kokoro positives of "
               f"{cfg.phrase!r} (second TTS family)…")
        kokoro_cache_dir = (tts_cache_dir / "kokoro_positive") if tts_cache_dir else None
        if kokoro_cache_dir is None:
            from .tts_kokoro import synthesize_phrase as kokoro_synth_uncached
            kokoro_clips = kokoro_synth_uncached(
                cfg.phrase, cfg.kokoro_positives, name=cfg.kokoro_voice_name,
                seed=cfg.seed + 1, progress_fn=log_fn,
            )
        else:
            kokoro_clips = kokoro_synthesize(
                cfg.phrase,
                cfg.kokoro_positives,
                cache_dir=kokoro_cache_dir,
                name=cfg.kokoro_voice_name,
                seed=cfg.seed + 1,  # different seed → different voice sequence
                progress_fn=log_fn,
                force_regenerate=cfg.force_regenerate_tts,
                clip_prefix="kokoro_pos",
            )
        # Per-family spectral matching: Kokoro's spectral envelope differs
        # from Piper's, so we need a SEPARATE EQ curve fitted to Kokoro
        # samples specifically (not the Piper one reused).
        if cfg.spectral_matching and positives and kokoro_clips:
            user_env = compute_spectral_envelope(positives)
            kokoro_env = compute_spectral_envelope(
                kokoro_clips[: min(64, len(kokoro_clips))])
            if user_env is not None and kokoro_env is not None:
                eq = compute_transfer_eq(kokoro_env, user_env,
                                         strength=cfg.spectral_matching_strength)
                log_fn(f"  Kokoro spectral matching: gain "
                       f"{eq.min():.2f}-{eq.max():.2f} "
                       f"(strength={cfg.spectral_matching_strength:.2f})")
                kokoro_clips = [apply_eq(c, eq) for c in kokoro_clips]

        # Cache is already populated with raw (pre-EQ) Kokoro clips above -
        # no second save here, otherwise next run would treat EQ-applied
        # samples as raw and re-apply EQ.

        # Same heavy augmentation as Piper TTS positives - clean studio
        # voice → mic-domain reality bridge.
        for clip in kokoro_clips:
            prepped = prepare_clip(clip)
            augmented_pos.append(prepped)
            for _ in range(2):
                aug = augment_clip(
                    prepped,
                    warp_range=(0.92, 1.10),
                    noise_snr_range=(5.0, 25.0),
                    reverb_prob=0.85,
                    gain_db=8.0,
                    noise_pool=noise_pool,
                    rir_pool=rir_pool,
                )
                augmented_pos.append(prepare_clip(aug))
            if cfg.end_aligned_variants:
                augmented_pos.append(prepare_clip_end_aligned(clip))

        # Kokoro partial-utterance negatives, capped together with any Piper
        # partials so the combined TTS-partial share of negatives stays
        # bounded (otherwise 400 Piper + 200 Kokoro = 1200 partials would
        # dominate the negative class).
        if cfg.partial_negatives:
            kokoro_partials = _make_partial_negatives(kokoro_clips)
            max_kokoro_partials = max(40, len(augmented_neg) // 4)
            if len(kokoro_partials) > max_kokoro_partials:
                kokoro_partials = rng.sample(kokoro_partials, max_kokoro_partials)
            log_fn(f"  → {len(kokoro_partials)} partial-utterance negatives "
                   f"from Kokoro (capped)")
            for clip in kokoro_partials:
                prepped = prepare_clip(clip)
                if prepped.abs().max() > 1e-3:
                    augmented_neg.append(prepped)

    # TTS-generated distractor phrases - hard negatives in many voices.
    if cfg.tts_negative_phrases:
        from .tts import synthesize_phrase_with_cache
        for distractor in cfg.tts_negative_phrases:
            log_fn(f"requesting {cfg.tts_negatives_per_phrase} Piper negatives "
                   f"of {distractor!r}…")
            if tts_cache_dir is None:
                from .tts import synthesize_phrase
                tts_neg = synthesize_phrase(
                    distractor,
                    cfg.tts_negatives_per_phrase,
                    name=cfg.tts_voice_name,
                    seed=cfg.seed + hash(distractor) % 1000,
                )
            else:
                neg_cache = tts_cache_dir / "negative" / _safe_filename(distractor)
                tts_neg = synthesize_phrase_with_cache(
                    distractor,
                    cfg.tts_negatives_per_phrase,
                    cache_dir=neg_cache,
                    name=cfg.tts_voice_name,
                    seed=cfg.seed + hash(distractor) % 1000,
                    progress_fn=log_fn,
                    force_regenerate=cfg.force_regenerate_tts,
                    clip_prefix="tts_neg",
                )
            # Apply same spectral matching to distractors so they're in the
            # same acoustic domain as the user's mic recordings.
            # (Clamped to ±6 dB inside compute_transfer_eq to prevent extreme
            # transformations on unusual mic profiles.)
            if cfg.spectral_matching and positives and tts_neg:
                user_env = compute_spectral_envelope(positives)
                neg_env = compute_spectral_envelope(tts_neg)
                if user_env is not None and neg_env is not None:
                    eq = compute_transfer_eq(neg_env, user_env,
                                             strength=cfg.spectral_matching_strength)
                    tts_neg = [apply_eq(c, eq) for c in tts_neg]
            for clip in tts_neg:
                augmented_neg.append(prepare_clip(clip))

    # Batch all clips into one tensor for vectorized mel extraction
    pos_batch = torch.stack(augmented_pos)  # (Np, samples)
    neg_batch = torch.stack(augmented_neg)
    pos_mels = log_mel(pos_batch)  # (Np, n_mels, T)
    neg_mels = log_mel(neg_batch)

    # Mel-domain VTLP warp adds further speaker variation cheaply
    pos_mels_aug = torch.stack([random_freq_warp(m) for m in pos_mels])
    neg_mels_aug = torch.stack([random_freq_warp(m) for m in neg_mels])

    mels = torch.cat([pos_mels_aug, neg_mels_aug], dim=0)
    labels = torch.cat(
        [
            torch.ones(pos_mels_aug.shape[0]),
            torch.zeros(neg_mels_aug.shape[0]),
        ]
    )

    # Speaker prototype: mean across original (un-augmented) negative recordings
    if negatives:
        orig_neg = torch.stack(negatives)
        orig_neg_mels = log_mel(orig_neg)
        prototype = orig_neg_mels.mean(dim=(0, 2))  # (n_mels,)
    else:
        prototype = torch.zeros(N_MELS)

    # Truncate / pad mel time-axis to a consistent WINDOW_FRAMES (defensive)
    target_T = mels.shape[-1]
    if target_T != WINDOW_FRAMES + 1 and target_T >= 8:
        # MelSpectrogram with center=True returns WINDOW_FRAMES+1 - keep what we got
        pass

    return mels, labels, prototype


def _split(mels: torch.Tensor, labels: torch.Tensor, val_frac: float, seed: int):
    n = mels.shape[0]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)
    n_val = max(1, int(val_frac * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return (
        mels[train_idx], labels[train_idx],
        mels[val_idx], labels[val_idx],
    )


class _FocalLoss(nn.Module):
    """Focal loss for binary classification with logits.

    Down-weights "easy" examples (those the model already classifies
    confidently) so the gradient budget concentrates on hard ones. With
    our typical class imbalance (negatives ≈ positives × 1.5) and easy-vs-
    hard mix (real speech is mostly trivially classified), this is a
    strict improvement over BCE in expectation. Reference: Lin et al. 2017.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()


def _resolve_device(spec: str, log_fn=print) -> torch.device:
    """Resolve "auto" / "cuda" / "cpu" → torch.device, with friendly fallback."""
    if spec == "cuda" or (spec == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            log_fn(f"device: {dev} ({torch.cuda.get_device_name(0)})")
            return dev
        if spec == "cuda":
            log_fn("device: cuda requested but unavailable - falling back to cpu")
    return torch.device("cpu")


def _calibrate_threshold(
    scores: torch.Tensor, labels: torch.Tensor, target_fpr: float
) -> tuple[float, float]:
    """Pick a sensible decision threshold.

    Validation negatives are often *easier* than what the model meets in
    deployment (especially before the user has added phonetically similar
    distractors), so the naive (1 - target_fpr) quantile sits very close to 0
    and produces a useless threshold. We use the max of two estimates:

    1. The (1 - target_fpr) quantile of validation negative scores
       - tight when val negatives are representative.
    2. A midpoint between mean(pos) and mean(neg) scores
       - robust when val negatives are too easy.

    Final threshold is clamped to [0.30, 0.95] so a real deployment never
    starts in the wildly-too-loose regime.
    """
    neg_scores = scores[labels == 0]
    pos_scores = scores[labels == 1]
    if neg_scores.numel() == 0 or pos_scores.numel() == 0:
        return 0.5, float((pos_scores > 0.5).float().mean()) if pos_scores.numel() else 0.0

    q = max(0.0, min(1.0, 1.0 - target_fpr))
    tight = float(torch.quantile(neg_scores, q))

    pos_mean = float(pos_scores.mean())
    neg_mean = float(neg_scores.mean())
    midpoint = (pos_mean + neg_mean) / 2.0

    threshold = max(tight, midpoint * 0.85)
    threshold = float(min(max(threshold, 0.30), 0.95))
    tpr = float((pos_scores > threshold).float().mean())
    return threshold, tpr


def train_wake_word(
    positive_dir: str | Path,
    negative_dir: str | Path,
    output_path: str | Path,
    phrase: str = "",
    cfg: TrainerConfig | None = None,
    log_fn=print,
) -> TrainingArtifact:
    """Train a wake-word model end-to-end.

    Positives go in positive_dir/*.wav; negatives in negative_dir/*.wav.
    Each WAV is loaded, peak-normalized, silence-trimmed, centered in a 1-sec window.
    """
    t0 = time.time()
    cfg = cfg or TrainerConfig(phrase=phrase)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = _resolve_device(cfg.device, log_fn)

    positives = load_dir_clips(positive_dir)
    negatives = load_dir_clips(negative_dir)
    if not positives:
        raise RuntimeError(f"no positive .wav files found in {positive_dir}")
    if not negatives:
        raise RuntimeError(f"no negative .wav files found in {negative_dir}")
    log_fn(f"loaded {len(positives)} positives, {len(negatives)} negatives")

    if cfg.tts_positives > 0:
        msg = (f"TTS: ON - {cfg.tts_positives} Piper positives + "
               f"{len(cfg.tts_negative_phrases)} distractor phrases")
        if cfg.kokoro_positives > 0:
            msg += f" + {cfg.kokoro_positives} Kokoro positives (cross-TTS-family)"
        log_fn(msg)
    elif cfg.kokoro_positives > 0:
        log_fn(f"TTS: ON - {cfg.kokoro_positives} Kokoro positives "
               f"(Piper disabled)")
    else:
        log_fn("TTS: OFF - model will only learn from this speaker's voice "
               "(may false-trigger on similar tones)")
    # Noise pool - combine user's ambient (real domain) + parametric noise
    # (broader category coverage). User's ambient is precious if present:
    # it's the exact mic+room signature of where the model will be deployed.
    noise_pool: list[torch.Tensor] = []
    silence_negative_clips: list[torch.Tensor] = []  # for direct "no speech" negatives
    project_dir = Path(positive_dir).parent
    ambient_path = project_dir / "ambient.wav"
    if ambient_path.exists():
        from .audio import load_wav
        ambient = load_wav(ambient_path)
        chunk_n = SAMPLE_RATE
        ambient_chunks: list[torch.Tensor] = []
        for s in range(0, ambient.numel() - chunk_n + 1, chunk_n // 2):
            ambient_chunks.append(ambient[s : s + chunk_n])
        noise_pool.extend(ambient_chunks)
        silence_negative_clips.extend(ambient_chunks)
        if ambient_chunks:
            log_fn(f"loaded room ambient: {ambient.numel()/SAMPLE_RATE:.1f}s "
                   f"→ {len(ambient_chunks)} chunks "
                   f"(user's mic/room domain; also added as silence negatives)")
    if cfg.use_parametric_noise:
        param_noise = build_noise_pool(cfg.parametric_noise_pool_size, seed=cfg.seed)
        noise_pool.extend(param_noise)
        log_fn(f"+ {len(param_noise)} parametric noise samples "
               f"(white/pink/brown/hum/fan/babble)")
        # Also add the non-babble parametric noises as direct silence-class
        # negatives. Babble is intentionally excluded because it has speech-
        # like structure and would teach the model "any voice = no" - too
        # aggressive. The build_noise_pool cycles kinds in this order:
        # white, pink, brown, hum, fan, babble. We keep 5 of each non-babble.
        kinds = ("white", "pink", "brown", "hum", "fan", "babble")
        for i, sample in enumerate(param_noise):
            if kinds[i % len(kinds)] != "babble":
                silence_negative_clips.append(sample)
        # A few true-zero "absolutely nothing" anchors. Without these, the
        # model never sees the "deafening silence" case - gives it a solid
        # negative anchor at zero-energy input.
        for _ in range(4):
            silence_negative_clips.append(torch.zeros(SAMPLE_RATE))
    if not noise_pool:
        noise_pool = None
        log_fn("noise pool: empty - falling back to Gaussian during augment")

    # Merge silence negatives into the user-negatives list so they get the
    # standard 25× augmentation pipeline. Critical fix: training previously
    # used ambient ONLY as noise-mixing material on top of speech clips,
    # never as labeled "no speech" negatives. Result: at inference, a 1-sec
    # buffer of pure mic ambient was out-of-distribution and the model fell
    # to ~0.5 (false-fires on fan / breathing / typing). Adding these as
    # labeled negatives fixes that.
    if silence_negative_clips:
        # Must apply prepare_clip - parametric noise samples are 1.5s long
        # by default and the user-side negatives list expects 1s windows
        # (load_dir_clips already prepares user .wav files). Without this
        # the downstream torch.stack(augmented_neg) blows up with mixed
        # tensor sizes. Pure-zero clips intentionally remain as zeros after
        # prepare_clip; they're a valid "absolutely nothing" training anchor.
        prepared = [prepare_clip(c) for c in silence_negative_clips]
        negatives = list(negatives) + prepared
        log_fn(f"silence-class negatives added: {len(prepared)} "
               f"(ambient + non-babble parametric + zeros) - teaches "
               f"'no speech in buffer = score 0'")

    # RIR pool - pre-generate a diverse parametric pool spanning room sizes
    # and distances. ~5 ms each, 40 of them ≈ negligible memory.
    rir_pool: list[torch.Tensor] | None = None
    if cfg.use_rir:
        rir_pool = build_rir_pool(cfg.rir_pool_size, seed=cfg.seed)
        log_fn(f"built RIR pool: {len(rir_pool)} parametric impulse responses "
               f"(replacing comb-filter reverb)")

    log_fn("building augmented dataset…")
    tts_cache_dir = Path(output_path).parent / "tts_cache"
    any_tts = (cfg.tts_positives > 0 or cfg.tts_negative_phrases
               or cfg.kokoro_positives > 0)
    mels, labels, prototype = _build_dataset(
        positives, negatives, cfg, log_fn=log_fn,
        tts_cache_dir=tts_cache_dir if any_tts else None,
        noise_pool=noise_pool,
        rir_pool=rir_pool,
    )
    log_fn(f"  → {int((labels==1).sum())} positive frames, "
           f"{int((labels==0).sum())} negative frames, mel shape {tuple(mels.shape[1:])}")

    train_x, train_y, val_x, val_y = _split(mels, labels, cfg.val_split, cfg.seed)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    size_presets = {
        "small":  {"channels": 32, "n_blocks": 3},
        "medium": {"channels": 64, "n_blocks": 3},
        "large":  {"channels": 96, "n_blocks": 4},
    }
    preset = size_presets.get(cfg.model_size, size_presets["small"])
    model = TinyWakeWordNet(n_mels=N_MELS, **preset).to(device)
    n_params = count_parameters(model)
    log_fn(f"model: TinyWakeWordNet ({cfg.model_size}, channels={preset['channels']}, "
           f"n_blocks={preset['n_blocks']}) - {n_params} parameters "
           f"({n_params * 4 / 1024:.1f} KB float32)")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.loss_function == "focal":
        loss_fn = _FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
        log_fn(f"loss: focal (alpha={cfg.focal_alpha}, gamma={cfg.focal_gamma})")
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        log_fn("loss: BCEWithLogits")

    proto = prototype.view(N_MELS, 1).to(device)  # broadcastable to (n_mels, T)

    history: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        n = train_x.shape[0]
        idx = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, n, cfg.batch_size):
            batch_idx = idx[s : s + cfg.batch_size]
            x = train_x[batch_idx]
            y = train_y[batch_idx]

            # Speaker-prototype regularizer:
            # with prob p, push or pull the input toward the speaker prototype.
            # Equally for positives and negatives - the model must classify
            # correctly regardless of how strongly the speaker's voice profile
            # is present.
            if random.random() < cfg.prototype_apply_prob:
                lo, hi = cfg.prototype_alpha_range
                alpha = random.uniform(lo, hi) * random.choice([-1.0, 1.0])
                x = x + alpha * proto

            # SpecAugment: random freq/time masking on the mel feature.
            # Forces the model to not rely on any single band as a shortcut.
            if cfg.specaugment:
                x = specaugment(
                    x,
                    n_freq_masks=cfg.spec_freq_masks,
                    freq_mask_width=cfg.spec_freq_width,
                    n_time_masks=cfg.spec_time_masks,
                    time_mask_width=cfg.spec_time_width,
                )

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= max(1, n_batches)

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_probs = torch.sigmoid(val_logits)
            val_acc = float(((val_probs > 0.5).float() == val_y).float().mean())
            val_loss = float(loss_fn(val_logits, val_y))

        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            log_fn(f"  epoch {epoch:02d}: train_loss={epoch_loss:.4f} "
                   f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    # Final calibration on full validation set
    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(val_x)).cpu()
    val_y_cpu = val_y.cpu()
    threshold, tpr_at_fpr = _calibrate_threshold(val_scores, val_y_cpu, cfg.threshold_target_fpr)
    log_fn(f"calibrated threshold = {threshold:.3f} "
           f"(TPR={tpr_at_fpr:.2f} at FPR={cfg.threshold_target_fpr:.2f})")

    # User-voice-only evaluation + threshold recalibration.
    # The val-set calibration above is dominated by TTS samples (hundreds of
    # synthesized voices) and produces a threshold tuned to those. The user
    # only cares about whether the model triggers on THEIR voice. So: score
    # the user's raw positive_*.wav and negative_*.wav, then sweep candidate
    # thresholds and pick the one that maximizes user-F1 with TPR ≥ 0.9.
    # Falls back to the val-calibrated threshold if too few user clips.
    user_tpr, user_fpr = 0.0, 0.0
    n_user_pos_eval, n_user_neg_eval = 0, 0
    try:
        from .audio import load_wav as _load_wav, prepare_clip as _prep, log_mel as _log_mel
        model_cpu_for_eval = model.cpu()
        model_cpu_for_eval.eval()
        pos_scores: list[float] = []
        neg_scores: list[float] = []
        for wp in sorted(Path(positive_dir).glob("*.wav")):
            try:
                clip = _prep(_load_wav(wp))
                with torch.no_grad():
                    pos_scores.append(float(torch.sigmoid(
                        model_cpu_for_eval(_log_mel(clip))).item()))
            except Exception:
                continue
        for wn in sorted(Path(negative_dir).glob("*.wav")):
            try:
                clip = _prep(_load_wav(wn))
                with torch.no_grad():
                    neg_scores.append(float(torch.sigmoid(
                        model_cpu_for_eval(_log_mel(clip))).item()))
            except Exception:
                continue
        n_user_pos_eval = len(pos_scores)
        n_user_neg_eval = len(neg_scores)

        # Threshold calibration via MAX-MARGIN in the gap. The previous logic
        # picked the lowest threshold satisfying TPR≥0.9 + F1=optimal - but
        # with a clean gap between user negatives and positives, MANY
        # thresholds tie on F1 and we'd pick the lowest. That's exactly wrong
        # for streaming inference, where probs are noisier and frequently
        # ride into the lower part of the gap on non-speech audio.
        #
        # Better: pick a threshold that sits 70% of the way from max_neg
        # toward min_passing_pos. That maximizes the false-trigger margin
        # while still leaving ~30% of the gap on the positive side as
        # safety for streaming-induced positive-score variation.
        if n_user_pos_eval >= 3 and n_user_neg_eval >= 3:
            val_threshold = threshold
            sorted_pos = sorted(pos_scores)
            # The lowest TPR-passing positive (we accept losing the bottom 10%
            # so we don't get pinned by a single weak recording)
            n_pos_to_keep = max(1, int(n_user_pos_eval * 0.9))
            min_passing_pos = sorted_pos[-n_pos_to_keep]
            max_neg = max(neg_scores) if neg_scores else 0.0

            if min_passing_pos > max_neg:
                # Clean gap - push threshold 70% toward positives
                new_threshold = max_neg + 0.7 * (min_passing_pos - max_neg)
                # Snap to a reasonable resolution and clamp to safe range
                new_threshold = float(min(max(round(new_threshold, 3), 0.30), 0.90))
                # Sanity check: at this threshold, compute actual TPR/FPR
                tp = sum(1 for s in pos_scores if s > new_threshold)
                fp = sum(1 for s in neg_scores if s > new_threshold)
                tpr_check = tp / n_user_pos_eval
                fpr_check = fp / n_user_neg_eval
                if tpr_check >= 0.85:  # safety: don't pick something that breaks recall
                    threshold = new_threshold
                    log_fn(f"user-voice threshold: val-calibrated={val_threshold:.3f} "
                           f"→ max-margin={threshold:.3f}  "
                           f"(max_neg={max_neg:.3f}, min_pos@TPR≥0.9={min_passing_pos:.3f}, "
                           f"TPR={tpr_check:.2f} FPR={fpr_check:.2f})")
                else:
                    log_fn(f"  ⚠ max-margin threshold {new_threshold:.3f} would "
                           f"drop TPR to {tpr_check:.2f}; keeping val threshold")
            else:
                # Overlap - no clean gap exists. The model isn't reliably
                # discriminating; warn loudly. Keep val threshold.
                log_fn(f"  ⚠ user-voice positive/negative overlap "
                       f"(max_neg={max_neg:.3f} ≥ min_pos@TPR≥0.9={min_passing_pos:.3f}) "
                       f"- model can't separate your voice from your negatives. "
                       f"Record more positives with variation, or more hard negatives.")

        # Final metrics at the chosen threshold
        pos_hits = sum(1 for s in pos_scores if s > threshold)
        neg_hits = sum(1 for s in neg_scores if s > threshold)
        user_tpr = pos_hits / max(1, n_user_pos_eval)
        user_fpr = neg_hits / max(1, n_user_neg_eval)
        log_fn(f"user-voice eval @ threshold={threshold:.3f}: "
               f"TPR={user_tpr:.2f} ({pos_hits}/{n_user_pos_eval}) "
               f"FPR={user_fpr:.2f} ({neg_hits}/{n_user_neg_eval})  "
               f"← directly measures 'does it trigger on YOU?'")
        model = model_cpu_for_eval
    except Exception as exc:
        log_fn(f"  [warn] user-voice eval failed: {type(exc).__name__}: {exc}")

    # Save model + metadata (always to CPU for portability)
    model = model.cpu()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "phrase": phrase,
        "threshold": threshold,
        "speaker_prototype": prototype,
        "n_mels": N_MELS,
        "sample_rate": SAMPLE_RATE,
        "window_frames": WINDOW_FRAMES,
        # Persist architecture so load_model rebuilds the right shape.
        # Previously load_model hardcoded channels=32/n_blocks=3 which
        # would silently mis-shape larger models loaded later.
        "channels": preset["channels"],
        "n_blocks": preset["n_blocks"],
        "model_size": cfg.model_size,
    }
    torch.save(payload, output_path)

    artifact = TrainingArtifact(
        phrase=phrase,
        threshold=threshold,
        train_loss_final=history[-1]["train_loss"],
        val_accuracy=history[-1]["val_acc"],
        val_tpr_at_target_fpr=tpr_at_fpr,
        n_positives_real=len(positives),
        n_negatives_real=len(negatives),
        n_positives_aug=int((labels == 1).sum()),
        n_negatives_aug=int((labels == 0).sum()),
        n_params=n_params,
        seconds=time.time() - t0,
        tts_positives_used=cfg.tts_positives,
        tts_distractor_phrases=len(cfg.tts_negative_phrases),
        kokoro_positives_used=cfg.kokoro_positives,
        user_voice_tpr=user_tpr,
        user_voice_fpr=user_fpr,
        n_user_positives_eval=n_user_pos_eval,
        n_user_negatives_eval=n_user_neg_eval,
        history=history,
    )

    # Sidecar JSON for human inspection
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(json.dumps(asdict(artifact), indent=2), encoding="utf-8")

    log_fn(f"saved → {output_path}  ({output_path.stat().st_size/1024:.1f} KB)")

    # Also save a slot copy under project/models/<size>.pt so subsequent
    # trainings at different sizes don't clobber this one. The "active" model
    # remains the project-root model.pt (kept for backward compat with all
    # existing code paths); the UI activates a slot by copying it back to
    # the project root. Slot name = model size by default; trainings at the
    # same size overwrite the same slot.
    import shutil
    models_dir = output_path.parent / "models"
    models_dir.mkdir(exist_ok=True)
    slot_name = cfg.model_size or "small"
    slot_pt = models_dir / f"{slot_name}.pt"
    slot_json = models_dir / f"{slot_name}.json"
    shutil.copy2(output_path, slot_pt)
    shutil.copy2(sidecar, slot_json)
    log_fn(f"slot   → {slot_pt}  (activatable from UI; "
           f"existing slots preserved across trainings)")

    log_fn(f"training took {artifact.seconds:.1f}s")
    return artifact


def load_model(path: str | Path) -> tuple[TinyWakeWordNet, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Read architecture from payload (saved by recent trainer); fall back to
    # the legacy small-model defaults so models from before the size-preset
    # change still load.
    channels = int(payload.get("channels", 32))
    n_blocks = int(payload.get("n_blocks", 3))
    model = TinyWakeWordNet(n_mels=payload["n_mels"],
                            channels=channels, n_blocks=n_blocks)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
