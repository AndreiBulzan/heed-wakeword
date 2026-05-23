"""Evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from .audio import load_dir_clips, log_mel
from .trainer import load_model


@dataclass
class EvalReport:
    n_positives: int
    n_negatives: int
    threshold: float
    accuracy: float
    tpr: float            # true positive rate (sensitivity / recall on positives)
    fpr: float            # false positive rate
    auc_approx: float     # approximate ROC AUC
    mean_pos_score: float
    mean_neg_score: float


def _approx_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """Approximate ROC AUC by counting concordant pairs (Wilcoxon)."""
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.0
    p = pos_scores.unsqueeze(1)  # (P, 1)
    n = neg_scores.unsqueeze(0)  # (1, N)
    greater = (p > n).float().mean()
    equal = (p == n).float().mean() * 0.5
    return float(greater + equal)


@torch.no_grad()
def evaluate_dirs(
    model_path: str | Path,
    positive_dir: str | Path | None = None,
    negative_dir: str | Path | None = None,
    threshold_override: float | None = None,
) -> EvalReport:
    """Run model on labelled directories. Reports per-clip classification."""
    model, payload = load_model(model_path)
    threshold = float(
        payload["threshold"] if threshold_override is None else threshold_override
    )

    pos_clips = load_dir_clips(positive_dir) if positive_dir else []
    neg_clips = load_dir_clips(negative_dir) if negative_dir else []

    pos_scores = torch.zeros(0)
    neg_scores = torch.zeros(0)
    if pos_clips:
        pos_mels = torch.cat([log_mel(c) for c in pos_clips], dim=0)
        pos_scores = torch.sigmoid(model(pos_mels))
    if neg_clips:
        neg_mels = torch.cat([log_mel(c) for c in neg_clips], dim=0)
        neg_scores = torch.sigmoid(model(neg_mels))

    n_pos = pos_scores.numel()
    n_neg = neg_scores.numel()
    tp = int((pos_scores > threshold).sum())
    fn = n_pos - tp
    fp = int((neg_scores > threshold).sum())
    tn = n_neg - fp

    tpr = tp / max(1, n_pos)
    fpr = fp / max(1, n_neg)
    acc = (tp + tn) / max(1, n_pos + n_neg)
    auc = _approx_auc(pos_scores, neg_scores)
    mean_pos = float(pos_scores.mean()) if n_pos else 0.0
    mean_neg = float(neg_scores.mean()) if n_neg else 0.0

    return EvalReport(
        n_positives=n_pos,
        n_negatives=n_neg,
        threshold=threshold,
        accuracy=acc,
        tpr=tpr,
        fpr=fpr,
        auc_approx=auc,
        mean_pos_score=mean_pos,
        mean_neg_score=mean_neg,
    )


def fmt_report(report: EvalReport) -> str:
    d = asdict(report)
    return (
        f"positives:        {d['n_positives']}\n"
        f"negatives:        {d['n_negatives']}\n"
        f"threshold:        {d['threshold']:.3f}\n"
        f"accuracy:         {d['accuracy']:.3f}\n"
        f"TPR (recall):     {d['tpr']:.3f}\n"
        f"FPR:              {d['fpr']:.3f}\n"
        f"approx AUC:       {d['auc_approx']:.3f}\n"
        f"mean pos score:   {d['mean_pos_score']:.3f}\n"
        f"mean neg score:   {d['mean_neg_score']:.3f}\n"
    )
