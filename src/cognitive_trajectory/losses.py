"""Loss functions for trajectory quantiles."""

import torch

from .schema import QUANTILES


def pinball_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean pinball across people and five quantiles (median = half MAE).

    Definition: https://pytorch-forecasting.readthedocs.io/en/stable/api/
    pytorch_forecasting.metrics.quantile.QuantileLoss.html
    """
    if predictions.ndim != 2 or predictions.shape[1] != len(QUANTILES):
        raise ValueError("predictions must have shape [B, 5]")
    if targets.ndim != 1 or targets.shape[0] != predictions.shape[0] or not targets.numel():
        raise ValueError("targets must have nonempty shape [B] matching predictions")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise TypeError("predictions and targets must be floating point")
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must be on the same device")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("predictions and targets must be finite")
    quantiles = predictions.new_tensor(QUANTILES)
    error = targets[:, None] - predictions
    return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()
