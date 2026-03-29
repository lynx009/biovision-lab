"""Dice loss for medical image segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary and multi-class segmentation.

    Parameters
    ----------
    smooth:
        Laplace smoothing term to prevent division by zero.
    include_background:
        When ``False`` (default for multi-class), channel 0 (background) is
        excluded from the loss computation.
    softmax:
        Apply ``softmax`` to *predictions* before computing the loss.  Set to
        ``False`` when your model already outputs probabilities.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        include_background: bool = True,
        softmax: bool = False,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.softmax = softmax

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice loss.

        Parameters
        ----------
        predictions:
            Logits or probabilities of shape ``(N, C, *spatial)``.
        targets:
            One-hot encoded ground truth of shape ``(N, C, *spatial)`` or
            integer class map of shape ``(N, *spatial)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if self.softmax:
            predictions = F.softmax(predictions, dim=1)

        num_classes = predictions.shape[1]

        # Convert integer targets to one-hot if needed.
        if targets.dim() == predictions.dim() - 1:
            targets = F.one_hot(targets.long(), num_classes).permute(0, -1, *range(1, targets.dim())).float()

        start_channel = 0 if self.include_background else 1
        predictions = predictions[:, start_channel:]
        targets = targets[:, start_channel:]

        # Flatten spatial dimensions.
        p = predictions.reshape(predictions.shape[0], predictions.shape[1], -1)
        t = targets.reshape(targets.shape[0], targets.shape[1], -1)

        intersection = (p * t).sum(dim=-1)
        union = p.sum(dim=-1) + t.sum(dim=-1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
