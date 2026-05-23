"""ONNX export must be numerically equivalent to PyTorch. Skipped when
onnxruntime is not installed (it is an opt-in `[export]` extra)."""

import numpy as np
import torch

from heed import N_MELS
from heed.model import TinyWakeWordNet


def test_onnx_matches_torch(tmp_path):
    import pytest
    ort = pytest.importorskip("onnxruntime")

    model = TinyWakeWordNet().eval()
    dummy = torch.randn(1, N_MELS, 101)
    onnx_path = tmp_path / "m.onnx"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["mel"], output_names=["logit"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )

    with torch.no_grad():
        pt_out = model(dummy).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"mel": dummy.numpy()})[0]

    assert np.abs(pt_out - onnx_out).max() < 1e-4
