"""Training must bound audio/model batches without losing clips or labels."""

import random
import weakref
from dataclasses import replace

import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from heed import N_MELS, SAMPLE_RATE, WINDOW_FRAMES
from heed import trainer
from heed.audio import log_mel, prepare_clip, save_wav
from heed.model import TinyWakeWordNet


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _clips(count):
    generator = torch.Generator().manual_seed(123)
    return [prepare_clip(torch.randn(SAMPLE_RATE, generator=generator))
            for _ in range(count)]


@pytest.mark.parametrize("batch_size", [1, 4])
def test_features_and_prototype_use_bounded_batches(monkeypatch, batch_size):
    positives, negatives = _clips(2), _clips(7)
    expected_pos = log_mel(torch.stack(positives))
    expected_neg = log_mel(torch.stack(negatives))
    stft = torch.stft
    batch_sizes = []

    def bounded_stft(audio, *args, **kwargs):
        batch_sizes.append(audio.shape[0])
        assert audio.shape[0] <= batch_size
        return stft(audio, *args, **kwargs)

    monkeypatch.setattr(torch, "stft", bounded_stft)
    monkeypatch.setattr(trainer, "random_freq_warp", lambda mel: mel)
    cfg = trainer.TrainerConfig(
        batch_size=batch_size, aug_positives_per_real=1,
        aug_negatives_per_real=1, partial_negatives=False,
        end_aligned_variants=False,
    )
    pos, neg, prototype = trainer._build_dataset(positives, negatives, cfg)

    assert len(pos) == len(positives) * 4  # original + three random alignments
    assert len(neg) == len(negatives)
    for i in range(len(positives)):
        mel, label = pos[i * 4]
        torch.testing.assert_close(mel, expected_pos[i])
        assert label == 1
    for i in range(len(negatives)):
        mel, label = neg[i]
        torch.testing.assert_close(mel, expected_neg[i])
        assert label == 0
    torch.testing.assert_close(prototype, expected_neg.mean(dim=(0, 2)),
                               atol=1e-6, rtol=1e-5)
    assert max(batch_sizes) == batch_size


def test_generated_waveforms_are_released_after_extraction():
    dataset = trainer._MelDataset(label=0.0, batch_size=4)
    references = []
    for _ in range(11):
        clip = torch.randn(SAMPLE_RATE)
        references.append(weakref.ref(clip))
        dataset.append(clip)
        del clip
        assert sum(ref() is not None for ref in references) < 4
    dataset.finish()
    assert all(ref() is None for ref in references)
    assert len(dataset) == 11
    assert dataset[10][0].shape == (N_MELS, WINDOW_FRAMES + 1)


@pytest.mark.parametrize("with_tts", [False, True])
def test_batch_size_preserves_augmented_examples(monkeypatch, tmp_path, with_tts):
    # Include both TTS families and distractors without optional engines or
    # downloads. Their clips go through the real augmentation/feature pipeline.
    if with_tts:
        from heed import tts, tts_kokoro

        def synthesize(phrase, count, **kwargs):
            return _clips(count)

        monkeypatch.setattr(tts, "synthesize_phrase_with_cache", synthesize)
        monkeypatch.setattr(tts_kokoro, "synthesize_phrase_with_cache", synthesize)
    cfg = trainer.TrainerConfig(
        phrase="test phrase", batch_size=128,
        aug_positives_per_real=2, aug_negatives_per_real=2,
        tts_positives=2 if with_tts else 0,
        kokoro_positives=1 if with_tts else 0,
        tts_negative_phrases=["another phrase"] if with_tts else [],
        tts_negatives_per_phrase=2,
    )
    positives, negatives = _clips(2), _clips(3)
    random.seed(cfg.seed)
    expected_pos, expected_neg, expected_proto = trainer._build_dataset(
        positives, negatives, cfg, tts_cache_dir=tmp_path,
    )
    random.seed(cfg.seed)
    pos, neg, prototype = trainer._build_dataset(
        positives, negatives, replace(cfg, batch_size=3), tts_cache_dir=tmp_path,
    )
    assert len(pos) == (24 if with_tts else 12)
    assert len(neg) == (22 if with_tts else 14)
    for actual, expected in ((pos, expected_pos), (neg, expected_neg)):
        assert len(actual) == len(expected)
        for i in range(len(actual)):
            torch.testing.assert_close(actual[i], expected[i])
    torch.testing.assert_close(prototype, expected_proto, atol=1e-6, rtol=1e-5)


def test_split_shares_features_and_batches_do_not_mutate_cache():
    dataset = trainer._MelDataset(label=1.0, batch_size=3)
    for clip in _clips(11):
        dataset.append(clip)
    dataset.finish()
    combined = ConcatDataset([dataset])
    train, val = trainer._split(combined, val_frac=0.3, seed=42)
    assert train.dataset is val.dataset is combined
    assert len(train) == 8
    assert len(val) == 3
    assert set(train.indices).isdisjoint(val.indices)
    assert sorted(train.indices + val.indices) == list(range(len(combined)))
    train_again, val_again = trainer._split(combined, val_frac=0.3, seed=42)
    assert train.indices == train_again.indices
    assert val.indices == val_again.indices

    original = train[0][0].clone()
    x, _ = next(iter(DataLoader(train, batch_size=3)))
    x.add_(100)
    torch.testing.assert_close(train[0][0], original)


def test_batched_validation_matches_full_validation():
    torch.manual_seed(42)
    model = TinyWakeWordNet(channels=64).eval()
    x = torch.randn(11, N_MELS, WINDOW_FRAMES + 1)
    y = torch.arange(11).remainder(2).float()
    with torch.no_grad():
        expected = model(x)
    batch_sizes = []

    def check_batch(module, inputs):
        batch_sizes.append(inputs[0].shape[0])
        assert inputs[0].shape[0] <= 4
        assert not torch.is_grad_enabled()

    model.register_forward_pre_hook(check_batch)
    logits, labels = trainer._validation_logits(
        model, DataLoader(TensorDataset(x, y), batch_size=4), torch.device("cpu"),
    )
    assert batch_sizes == [4, 4, 3]
    assert not logits.requires_grad
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(labels, y)
    for loss_fn in (trainer._FocalLoss(), torch.nn.BCEWithLogitsLoss()):
        torch.testing.assert_close(loss_fn(logits, labels), loss_fn(expected, y))
    assert trainer._calibrate_threshold(logits.sigmoid(), labels, 0.01) == pytest.approx(
        trainer._calibrate_threshold(expected.sigmoid(), y, 0.01), abs=1e-6,
    )


def test_training_batches_every_model_call_and_saves_artifacts(monkeypatch, tmp_path):
    pos_dir, neg_dir = tmp_path / "positive", tmp_path / "negative"
    pos_dir.mkdir()
    neg_dir.mkdir()
    for directory, count in ((pos_dir, 2), (neg_dir, 3)):
        for i, clip in enumerate(_clips(count)):
            save_wav(directory / f"{i}.wav", clip)

    calls = []

    def check_batch(module, inputs):
        calls.append((module.training, len(inputs[0])))
        assert len(inputs[0]) <= 3

    def make_model(**kwargs):
        model = TinyWakeWordNet(**kwargs)
        model.register_forward_pre_hook(check_batch)
        return model

    monkeypatch.setattr(trainer, "TinyWakeWordNet", make_model)
    cfg = trainer.TrainerConfig(
        epochs=1, batch_size=3, aug_positives_per_real=2,
        aug_negatives_per_real=2, use_rir=False, use_parametric_noise=False,
        val_split=0.4, model_size="medium", device="cpu",
    )
    output = tmp_path / "model.pt"
    artifact = trainer.train_wake_word(pos_dir, neg_dir, output, cfg=cfg)
    assert artifact.n_positives_aug == 12
    assert artifact.n_negatives_aug == 14
    assert any(training for training, _ in calls)
    assert sum(not training for training, _ in calls) > 1
    assert output.with_suffix(".json").is_file()
    assert (tmp_path / "models" / "medium.pt").is_file()
    model, payload = trainer.load_model(output)
    assert payload["model_size"] == "medium"
    with torch.no_grad():
        assert torch.isfinite(model(log_mel(_clips(1)[0]))).all()
