"""Reusable evaluation passes and object-level instance metrics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation, binary_erosion
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
        self.aji_intersection = 0
        self.aji_union = 0
        self.pq_iou_sum = 0.0
        self.boundary_matched_prediction = 0
        self.boundary_prediction_count = 0
        self.boundary_matched_target = 0
        self.boundary_target_count = 0
        self.ap_samples: list[tuple[np.ndarray, np.ndarray, int]] = []

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
        assignment_probs = outputs.get("assignment_probs")
        for batch_index, (predicted, target) in enumerate(zip(prediction, targets)):
            self._update_boundaries(predicted, target)
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
            probabilities = (
                assignment_probs[batch_index].reshape(
                    assignment_probs.shape[1], -1
                )
                if assignment_probs is not None
                else None
            )
            self._update_objects(predicted, target, probabilities)

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
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        assignment_probs: torch.Tensor | None,
    ) -> None:
        """Match predicted and target objects by IoU for one sample."""
        predicted_ids = torch.unique(prediction)
        predicted_ids = predicted_ids[predicted_ids > 0]
        target_ids = torch.unique(target)
        target_ids = target_ids[target_ids > 0]
        scores = np.ones(int(predicted_ids.numel()), dtype=np.float64)
        if assignment_probs is not None:
            scores = np.asarray(
                [
                    float(
                        assignment_probs[int(predicted_id)]
                        [prediction == predicted_id]
                        .mean()
                    )
                    for predicted_id in predicted_ids
                ],
                dtype=np.float64,
            )
        if predicted_ids.numel() == 0:
            self.false_negatives += int(target_ids.numel())
            self.ap_samples.append(
                (scores, np.zeros((0, int(target_ids.numel()))), int(target_ids.numel()))
            )
            self.aji_union += int((target > 0).sum())
            return
        if target_ids.numel() == 0:
            self.false_positives += int(predicted_ids.numel())
            self.ap_samples.append(
                (scores, np.zeros((len(scores), 0)), 0)
            )
            self.aji_union += int((prediction > 0).sum())
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
        intersections = torch.zeros_like(iou)
        unions = torch.zeros_like(iou)
        for pred_index, pred_id in enumerate(predicted_ids):
            predicted_mask = prediction == pred_id
            for target_index, target_id in enumerate(target_ids):
                target_mask = target == target_id
                intersections[pred_index, target_index] = (
                    predicted_mask & target_mask
                ).sum()
                unions[pred_index, target_index] = (
                    predicted_mask | target_mask
                ).sum()
        pred_matches, target_matches = linear_sum_assignment(
            -iou.cpu().numpy()
        )
        matched_pairs = [
            (pred_index, target_index)
            for pred_index, target_index in zip(pred_matches, target_matches)
            if intersections[pred_index, target_index] > 0
        ]
        threshold_pairs = [
            (pred_index, target_index)
            for pred_index, target_index in matched_pairs
            if iou[pred_index, target_index].item() >= self.iou_threshold
        ]
        matched = len(threshold_pairs)
        self.true_positives += matched
        self.false_positives += int(predicted_ids.numel()) - matched
        self.false_negatives += int(target_ids.numel()) - matched
        matched_prediction_ids = {pair[0] for pair in matched_pairs}
        matched_target_ids = {pair[1] for pair in matched_pairs}
        self.aji_intersection += sum(
            int(intersections[pred_index, target_index])
            for pred_index, target_index in matched_pairs
        )
        self.aji_union += sum(
            int(unions[pred_index, target_index])
            for pred_index, target_index in matched_pairs
        )
        self.aji_union += sum(
            int((prediction == predicted_ids[index]).sum())
            for index in range(len(predicted_ids))
            if index not in matched_prediction_ids
        )
        self.aji_union += sum(
            int((target == target_ids[index]).sum())
            for index in range(len(target_ids))
            if index not in matched_target_ids
        )
        self.pq_iou_sum += sum(
            float(iou[pred_index, target_index])
            for pred_index, target_index in threshold_pairs
        )
        iou_array = iou.detach().cpu().numpy()
        self.ap_samples.append((scores, iou_array, int(target_ids.numel())))

    def _update_boundaries(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Accumulate one-pixel-tolerance instance boundary F1 counts."""
        predicted_array = prediction.detach().cpu().numpy()
        target_array = target.detach().cpu().numpy()
        if predicted_array.ndim != 2 or target_array.ndim != 2:
            raise ValueError("Boundary metrics require 2D instance maps.")
        predicted_array = predicted_array.copy()
        target_array = target_array.copy()
        predicted_array[target_array < 0] = 0
        target_array[target_array < 0] = 0
        predicted_boundary = np.zeros_like(predicted_array, dtype=bool)
        target_boundary = np.zeros_like(target_array, dtype=bool)
        for instance_id in np.unique(predicted_array):
            if instance_id > 0:
                mask = predicted_array == instance_id
                predicted_boundary |= mask & ~binary_erosion(
                    mask, structure=np.ones((3, 3)), border_value=0
                )
        for instance_id in np.unique(target_array):
            if instance_id > 0:
                mask = target_array == instance_id
                target_boundary |= mask & ~binary_erosion(
                    mask, structure=np.ones((3, 3)), border_value=0
                )
        self.boundary_matched_prediction += int(
            (predicted_boundary & binary_dilation(target_boundary)).sum()
        )
        self.boundary_prediction_count += int(predicted_boundary.sum())
        self.boundary_matched_target += int(
            (target_boundary & binary_dilation(predicted_boundary)).sum()
        )
        self.boundary_target_count += int(target_boundary.sum())

    def _average_precision(self, threshold: float) -> float:
        """Calculate all-points interpolated AP for one IoU threshold."""
        detections: list[tuple[float, int]] = []
        total_targets = 0
        for scores, overlaps, target_count in self.ap_samples:
            total_targets += target_count
            matched_targets: set[int] = set()
            for prediction_index in np.argsort(-scores):
                best_target = -1
                best_iou = threshold
                for target_index, iou in enumerate(overlaps[prediction_index]):
                    if target_index not in matched_targets and iou >= best_iou:
                        best_target = target_index
                        best_iou = float(iou)
                if best_target >= 0:
                    matched_targets.add(best_target)
                    detections.append((float(scores[prediction_index]), 1))
                else:
                    detections.append((float(scores[prediction_index]), 0))
        if total_targets == 0 or not detections:
            return 0.0
        detections.sort(key=lambda item: item[0], reverse=True)
        true_positives = np.cumsum([item[1] for item in detections])
        false_positives = np.cumsum([1 - item[1] for item in detections])
        recalls = true_positives / total_targets
        precisions = true_positives / np.maximum(
            true_positives + false_positives, 1
        )
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        for index in range(len(precisions) - 2, -1, -1):
            precisions[index] = max(precisions[index], precisions[index + 1])
        changed_recall = np.where(recalls[1:] != recalls[:-1])[0]
        return float(
            np.sum(
                (recalls[changed_recall + 1] - recalls[changed_recall])
                * precisions[changed_recall + 1]
            )
        )

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
            "foreground_iou": self.foreground_intersection
            / (self.foreground_union + epsilon),
            "ap50": self._average_precision(0.5),
            "ap75": self._average_precision(0.75),
            "aji": self.aji_intersection / (self.aji_union + epsilon),
            "pq": self.pq_iou_sum
            / (
                self.true_positives
                + 0.5 * self.false_positives
                + 0.5 * self.false_negatives
                + epsilon
            ),
        }
        metrics["average_precision"] = (
            metrics["ap50"] + metrics["ap75"]
        ) / 2
        boundary_precision = self.boundary_matched_prediction / (
            self.boundary_prediction_count + epsilon
        )
        boundary_recall = self.boundary_matched_target / (
            self.boundary_target_count + epsilon
        )
        metrics["boundary_f1"] = (
            2 * boundary_precision * boundary_recall
            / (boundary_precision + boundary_recall + epsilon)
        )
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
