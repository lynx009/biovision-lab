"""Evaluation metrics for detection (mAP) and segmentation (Dice Score)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def compute_dice_score(
    predictions: np.ndarray,
    targets: np.ndarray,
    smooth: float = 1.0,
    threshold: float = 0.5,
) -> float:
    """Compute the Dice Similarity Coefficient (DSC).

    Parameters
    ----------
    predictions:
        Predicted probability map of shape ``(H, W)`` or ``(D, H, W)``, or
        a binary mask of the same shape.
    targets:
        Binary ground-truth mask of the same shape as *predictions*.
    smooth:
        Laplace smoothing factor.
    threshold:
        Binarisation threshold applied to *predictions*.

    Returns
    -------
    float
        DSC in ``[0, 1]``.
    """
    pred_binary = (predictions >= threshold).astype(np.float32)
    target_binary = targets.astype(np.float32)

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    return float((2.0 * intersection + smooth) / (union + smooth))


def compute_map(
    detections: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: Optional[int] = None,
) -> float:
    """Compute mean Average Precision (mAP) at a fixed IoU threshold.

    Each detection / ground-truth dictionary must contain:

    * ``"boxes"`` – array of shape ``(N, 4)`` in ``[x1, y1, x2, y2]`` format.
    * ``"scores"`` – array of shape ``(N,)`` (detections only).
    * ``"labels"`` – integer class index array of shape ``(N,)``.

    Parameters
    ----------
    detections:
        List of per-image detection dicts.
    ground_truths:
        List of per-image ground-truth dicts (same order as *detections*).
    iou_threshold:
        Minimum IoU to count a detection as a true positive.
    num_classes:
        Total number of foreground classes.  Inferred from the data when not
        provided.

    Returns
    -------
    float
        mAP value in ``[0, 1]``.
    """
    if len(detections) != len(ground_truths):
        raise ValueError("detections and ground_truths must have the same length.")

    all_labels = [
        lbl
        for entry in ground_truths
        for lbl in np.asarray(entry["labels"]).tolist()
    ]
    classes = sorted(set(all_labels))
    if num_classes is not None:
        classes = list(range(num_classes))

    ap_per_class: List[float] = []
    for cls in classes:
        tp_list: List[int] = []
        score_list: List[float] = []
        n_gt = 0

        for dets, gts in zip(detections, ground_truths):
            gt_boxes = np.asarray(gts["boxes"])
            gt_labels = np.asarray(gts["labels"])
            cls_gt = gt_boxes[gt_labels == cls]
            n_gt += len(cls_gt)
            matched = np.zeros(len(cls_gt), dtype=bool)

            det_boxes = np.asarray(dets["boxes"])
            det_labels = np.asarray(dets["labels"])
            det_scores = np.asarray(dets["scores"])
            mask = det_labels == cls
            cls_boxes = det_boxes[mask]
            cls_scores = det_scores[mask]
            order = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            for box, score in zip(cls_boxes, cls_scores):
                score_list.append(float(score))
                if len(cls_gt) == 0:
                    tp_list.append(0)
                    continue
                ious = _batch_iou(box[np.newaxis], cls_gt).squeeze(0)
                best_idx = int(np.argmax(ious))
                if ious[best_idx] >= iou_threshold and not matched[best_idx]:
                    matched[best_idx] = True
                    tp_list.append(1)
                else:
                    tp_list.append(0)

        if n_gt == 0:
            continue
        ap_per_class.append(_average_precision(np.array(tp_list), np.array(score_list), n_gt))

    return float(np.mean(ap_per_class)) if ap_per_class else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return IoU matrix of shape ``(len(boxes_a), len(boxes_b))``."""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, np.newaxis] + area_b[np.newaxis] - inter
    return inter / np.maximum(union, 1e-6)


def _average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """Compute AP from a TP indicator array sorted by descending score."""
    order = np.argsort(-scores)
    tp_sorted = tp[order]
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1 - tp_sorted)

    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / (n_gt + 1e-6)

    # Append sentinel values for interpolation.
    precision = np.concatenate([[1.0], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [recall[-1]]])

    # Monotonically decreasing precision envelope.
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))
