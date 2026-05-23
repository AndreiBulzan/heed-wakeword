"""Tiny streaming-friendly CNN for wake-word detection.

Depthwise-separable 1D conv over time, treating mel bins as channels.
Target footprint: ~10 K params, <50 KB float32. INT8 ≈ 12 KB.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import N_MELS


class DSConvBlock(nn.Module):
    """Depthwise + pointwise conv with BN + ReLU. ~1.5 K params at C=32, k=7."""

    def __init__(self, channels: int, kernel_size: int = 7, stride: int = 1) -> None:
        super().__init__()
        self.depth = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.point = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.point(self.depth(x))), inplace=True)


class TinyWakeWordNet(nn.Module):
    """Compact wake-word classifier.

    Input:  (B, n_mels, T) log-mel spectrogram. T defaults to ~100 frames (1 sec).
    Output: (B,) raw logit. Apply sigmoid for probability.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        channels: int = 32,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(n_mels)
        # Stem: stride-2 conv reduces time resolution by 2 immediately
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        # Blocks: alternate plain and stride-2 to compress time further
        blocks: list[nn.Module] = []
        for i in range(n_blocks):
            stride = 2 if i == n_blocks // 2 else 1
            blocks.append(DSConvBlock(channels, kernel_size=7, stride=stride))
        self.blocks = nn.Sequential(*blocks)
        # Global average pool over time -> linear classifier
        self.classifier = nn.Linear(channels, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, n_mels, T)
        x = self.input_norm(mel)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)  # global avg pool over time
        return self.classifier(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
