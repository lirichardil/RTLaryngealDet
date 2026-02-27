"""Detection and SR evaluation metrics for SRE-YOLO."""
from typing import Dict, List

import torch
import numpy as np


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes (xyxy format)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve (every-point interpolation)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Monotonically decreasing envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def compute_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.001,
    nc: int = 4,
) -> Dict[str, float]:
    """Compute mAP@0.5, per-class AP, macro precision, and macro recall.

    Args:
        predictions: List of per-image dicts:
            {'boxes': Tensor(N,4) xyxy, 'scores': Tensor(N,), 'labels': Tensor(N,)}
        ground_truths: List of per-image dicts:
            {'boxes': Tensor(M,4) xyxy, 'labels': Tensor(M,)}
        iou_thresh: IoU threshold for a true positive (default 0.5).
        conf_thresh: Discard predictions below this confidence.
        nc: Number of classes.

    Returns:
        Dict with keys:
            'mAP50'       – mean AP across all classes
            'precision'   – macro-averaged precision
            'recall'      – macro-averaged recall
            'ap_<cls>'    – per-class AP (keyed by class index string)
    """
    # Per-class accumulators: list of (score, is_tp) tuples
    class_tp: Dict[int, List] = {c: [] for c in range(nc)}
    class_n_gt: Dict[int, int] = {c: 0 for c in range(nc)}

    for preds, gts in zip(predictions, ground_truths):
        pred_boxes  = preds["boxes"]    # (N, 4)
        pred_scores = preds["scores"]   # (N,)
        pred_labels = preds["labels"]   # (N,)

        gt_boxes  = gts["boxes"]        # (M, 4)
        gt_labels = gts["labels"]       # (M,)

        # Count ground-truth instances per class
        for c in gt_labels.tolist():
            class_n_gt[int(c)] += 1

        # Filter by conf_thresh
        keep = pred_scores >= conf_thresh
        pred_boxes  = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        matched_gt = set()

        # Sort predictions by descending confidence
        order = torch.argsort(pred_scores, descending=True)
        for idx in order.tolist():
            cls   = int(pred_labels[idx].item())
            score = float(pred_scores[idx].item())
            pb    = pred_boxes[idx].unsqueeze(0)  # (1, 4)

            # Find GT boxes of the same class
            gt_mask = gt_labels == cls
            if gt_mask.sum() == 0:
                class_tp[cls].append((score, 0))
                continue

            gt_cls_boxes = gt_boxes[gt_mask]
            ious = box_iou(pb, gt_cls_boxes)[0]  # (M_cls,)

            best_iou, best_j = ious.max(0)
            # Map back to global GT index
            global_gt_indices = gt_mask.nonzero(as_tuple=False).squeeze(1)
            global_j = int(global_gt_indices[best_j].item())

            if best_iou >= iou_thresh and global_j not in matched_gt:
                matched_gt.add(global_j)
                class_tp[cls].append((score, 1))
            else:
                class_tp[cls].append((score, 0))

    # Compute per-class AP, precision, recall
    aps, precisions, recalls = [], [], []
    results: Dict[str, float] = {}

    for c in range(nc):
        n_gt = class_n_gt[c]
        detections = sorted(class_tp[c], key=lambda x: -x[0])

        if n_gt == 0 and len(detections) == 0:
            continue

        tp_cumsum = np.cumsum([d[1] for d in detections]).astype(float)
        fp_cumsum = np.cumsum([1 - d[1] for d in detections]).astype(float)

        recall_curve    = tp_cumsum / (n_gt + 1e-7)
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

        ap = compute_ap(recall_curve, precision_curve)
        aps.append(ap)
        results[f"ap_{c}"] = round(ap, 4)

        if len(detections):
            precisions.append(float(precision_curve[-1]))
            recalls.append(float(recall_curve[-1]))

    results["mAP50"]     = round(float(np.mean(aps)) if aps else 0.0, 4)
    results["precision"] = round(float(np.mean(precisions)) if precisions else 0.0, 4)
    results["recall"]    = round(float(np.mean(recalls)) if recalls else 0.0, 4)
    return results
