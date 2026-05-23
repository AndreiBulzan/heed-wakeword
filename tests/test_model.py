"""Model shape and size guarantees."""

import torch

from heed import N_MELS
from heed.model import TinyWakeWordNet, count_parameters


def test_forward_shape():
    model = TinyWakeWordNet()
    out = model(torch.randn(2, N_MELS, 101))
    assert out.shape == (2,)  # one logit per item in the batch


def test_model_stays_tiny():
    # The default ("small") model is the headline "tiny" claim. Keep it tiny.
    assert count_parameters(TinyWakeWordNet()) < 50_000
