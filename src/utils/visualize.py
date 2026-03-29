"""Visualization utilities for medical imaging predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def visualize_prediction(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    boxes: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
    scores: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay segmentation masks and/or bounding boxes on *image*.

    Parameters
    ----------
    image:
        2-D greyscale ``(H, W)`` or 3-channel RGB ``(H, W, 3)`` array with
        values in ``[0, 1]`` or ``[0, 255]``.
    mask:
        Binary or integer label mask of shape ``(H, W)``.
    boxes:
        Bounding boxes of shape ``(N, 4)`` in ``[x1, y1, x2, y2]`` format.
    labels:
        Class name for each box (length ``N``).
    scores:
        Confidence score for each box (length ``N``).
    save_path:
        When given, the annotated image is saved to this path.
    alpha:
        Opacity of the segmentation mask overlay.
    colormap:
        Matplotlib colourmap name used to colour the segmentation mask.

    Returns
    -------
    np.ndarray
        Annotated RGB image (``uint8``, shape ``(H, W, 3)``).
    """
    import cv2  # type: ignore
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # Prepare base RGB image.
    # ------------------------------------------------------------------
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    elif img_u8.shape[2] == 3:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    canvas = img_u8.copy()

    # ------------------------------------------------------------------
    # Overlay segmentation mask.
    # ------------------------------------------------------------------
    if mask is not None:
        cmap = plt.get_cmap(colormap)
        mask_norm = mask.astype(np.float32)
        if mask_norm.max() > 0:
            mask_norm /= mask_norm.max()
        coloured = (cmap(mask_norm)[..., :3] * 255).astype(np.uint8)
        coloured_bgr = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)
        nonzero = mask > 0
        canvas[nonzero] = cv2.addWeighted(canvas, 1 - alpha, coloured_bgr, alpha, 0)[nonzero]

    # ------------------------------------------------------------------
    # Draw bounding boxes.
    # ------------------------------------------------------------------
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_parts = []
            if labels is not None:
                text_parts.append(str(labels[i]))
            if scores is not None:
                text_parts.append(f"{scores[i]:.2f}")
            if text_parts:
                cv2.putText(
                    canvas,
                    " ".join(text_parts),
                    (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

    result_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), canvas)

    return result_rgb
