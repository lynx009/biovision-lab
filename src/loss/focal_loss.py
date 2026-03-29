"""Focal loss for detection and segmentation with class imbalance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for handling class imbalance.

    Parameters
    ----------
    alpha:
        Weighting factor for the rare class in ``[0, 1]``.
    gamma:
        Focusing parameter (``gamma=0`` reduces to cross-entropy).
    reduction:
        Specifies the reduction to apply to the output: ``"mean"`` | ``"sum"``
        | ``"none"``.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none'; got '{reduction}'.")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        predictions:
            Raw logits of shape ``(N, C)`` for classification or
            ``(N, C, H, W)`` for dense prediction.
        targets:
            Integer class indices of shape ``(N,)`` or ``(N, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar (or per-element if ``reduction="none"``) loss value.
        """
        ce_loss = F.cross_entropy(predictions, targets.long(), reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
