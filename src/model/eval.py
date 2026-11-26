"""Reusable evaluation passes and object-level instance metrics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from tqdm.auto import tqdm

from .loss import build_affinity_targets


class InstanceSegmentationMetrics:
    """Aggregate object detection and affinity metrics across an evaluation set."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        if not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1].")
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        """Reset the aggregate counters for a new evaluation pass."""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.affinity_tp = np.zeros(4, dtype=np.int64)
        self.affinity_fp = np.zeros(4, dtype=np.int64)
        self.affinity_fn = np.zeros(4, dtype=np.int64)
        self.foreground_intersection = 0
        self.foreground_union = 0

    def update(
        self, outputs: Mapping[str, torch.Tensor], targets: torch.Tensor
    ) -> None:
        """Accumulate object-level and directional affinity confusion counts."""
        prediction = outputs["instance_map"].detach()
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        targets = targets.detach()
        if prediction.shape != targets.shape:
            raise ValueError("Predicted instance maps and targets must have equal shape.")
        for predicted, target in zip(prediction, targets):
            valid = target >= 0
            predicted = predicted[valid]
            target = target[valid]
            predicted_foreground = predicted > 0
            target_foreground = target > 0
            self.foreground_intersection += int(
                (predicted_foreground & target_foreground).sum()
            )
            self.foreground_union += int(
                (predicted_foreground | target_foreground).sum()
            )
            self._update_objects(predicted, target)

        if "affinity_logits" in outputs:
            affinity_targets, affinity_valid = build_affinity_targets(targets)
            prediction = outputs["affinity_logits"].detach().sigmoid() >= 0.5
            truth = affinity_targets.bool()
            for direction in range(4):
                valid = affinity_valid[:, direction]
                self.affinity_tp[direction] += int(
                    (prediction[:, direction] & truth[:, direction] & valid).sum()
                )
                self.affinity_fp[direction] += int(
                    (prediction[:, direction] & ~truth[:, direction] & valid).sum()
                )
                self.affinity_fn[direction] += int(
                    (~prediction[:, direction] & truth[:, direction] & valid).sum()
                )

    def _update_objects(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Match predicted and target objects by IoU for one sample."""
        predicted_ids = torch.unique(prediction)
        predicted_ids = predicted_ids[predicted_ids > 0]
        target_ids = torch.unique(target)
        target_ids = target_ids[target_ids > 0]
        if predicted_ids.numel() == 0:
            self.false_negatives += int(target_ids.numel())
            return
        if target_ids.numel() == 0:
            self.false_positives += int(predicted_ids.numel())
            return
        iou = torch.zeros(
            (predicted_ids.numel(), target_ids.numel()),
            dtype=torch.float64,
            device=prediction.device,
        )
        for pred_index, pred_id in enumerate(predicted_ids):
            predicted_mask = prediction == pred_id
            for target_index, target_id in enumerate(target_ids):
                target_mask = target == target_id
                intersection = (predicted_mask & target_mask).sum()
                union = (predicted_mask | target_mask).sum()
                iou[pred_index, target_index] = intersection / union.clamp_min(1)
        pred_matches, target_matches = linear_sum_assignment(
            -iou.cpu().numpy()
        )
        matched = sum(
            iou[pred_index, target_index].item() >= self.iou_threshold
            for pred_index, target_index in zip(pred_matches, target_matches)
        )
        self.true_positives += matched
        self.false_positives += int(predicted_ids.numel()) - matched
        self.false_negatives += int(target_ids.numel()) - matched

    def compute(self) -> dict[str, float]:
        """Return globally aggregated object and pixel-affinity metrics."""
        true_positive = self.true_positives
        false_positive = self.false_positives
        false_negative = self.false_negatives
        epsilon = 1e-8
        metrics = {
            "precision": true_positive / (true_positive + false_positive + epsilon),
            "recall": true_positive / (true_positive + false_negative + epsilon),
            "accuracy": true_positive
            / (true_positive + false_positive + false_negative + epsilon),
            "f1": 2 * true_positive
            / (2 * true_positive + false_positive + false_negative + epsilon),
            "foreground_dice": 2 * self.foreground_intersection
            / (self.foreground_union + self.foreground_intersection + epsilon),
        }
        affinity_precision: list[float] = []
        affinity_recall: list[float] = []
        for direction in range(4):
            tp = int(self.affinity_tp[direction])
            fp = int(self.affinity_fp[direction])
            fn = int(self.affinity_fn[direction])
            affinity_precision.append(tp / (tp + fp + epsilon))
            affinity_recall.append(tp / (tp + fn + epsilon))
        metrics["affinity_precision"] = float(np.mean(affinity_precision))
        metrics["affinity_recall"] = float(np.mean(affinity_recall))
        return metrics


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        images, labels, _ = batch
    elif len(batch) == 2:
        images, labels = batch
    else:
        raise ValueError("Evaluation batch must contain images, labels, and optional lengths.")
    return images, labels


def eval_epoch(
    model: nn.Module,
    device: torch.device,
    eval_loader: Any,
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    metric_fn: Any,
    *,
    amp: bool = False,
    progress_description: str = "Validation",
) -> dict[str, float]:
    """Evaluate one loader without gradients and return mean loss and metrics."""
    model.eval()
    losses: list[float] = []
    batch_metrics: list[dict[str, float]] = []
    if hasattr(metric_fn, "reset"):
        metric_fn.reset()
    progress = tqdm(eval_loader, desc=progress_description, leave=False)
    with torch.no_grad():
        for batch in progress:
            images, labels = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            losses.append(float(loss.detach()))
            if hasattr(metric_fn, "update"):
                metric_fn.update(outputs, labels)
            else:
                batch_metrics.append(dict(metric_fn(outputs, labels)))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
    result: dict[str, float] = {
        "loss": float(np.mean(losses)) if losses else 0.0
    }
    if hasattr(metric_fn, "compute"):
        result.update(metric_fn.compute())
    elif batch_metrics:
        keys = batch_metrics[0]
        result.update(
            {
                key: float(np.mean([metrics[key] for metrics in batch_metrics]))
                for key in keys
            }
        )
    return result
