"""U-Net predictions, targets, and watershed reconstruction for learned distances."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from skimage.segmentation import watershed
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from .semantic import SemanticSegmentationUNet


def _as_label_map(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError("Labels must have shape (N, H, W) or (N, 1, H, W).")
    return labels.to(dtype=torch.int64)


def build_learned_distance_targets(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build foreground, normalized center-distance, and boundary-distance maps."""
    label_map = _as_label_map(labels)
    label_array = label_map.detach().cpu().numpy()
    targets = np.zeros(
        (label_array.shape[0], 3, *label_array.shape[-2:]), dtype=np.float32
    )
    valid = label_array >= 0

    for batch_index, image_labels in enumerate(label_array):
        foreground = image_labels > 0
        targets[batch_index, 0] = foreground
        for instance_id in np.unique(image_labels[foreground]):
            instance = image_labels == instance_id
            boundary_distance = ndimage.distance_transform_edt(instance)
            boundary_distance = np.maximum(boundary_distance - 1.0, 0.0)
            maximum_boundary_distance = float(boundary_distance[instance].max())
            center = np.unravel_index(
                int(boundary_distance.argmax()), boundary_distance.shape
            )
            rows, columns = np.nonzero(instance)
            center_distance = np.sqrt(
                (rows - center[0]) ** 2 + (columns - center[1]) ** 2
            )
            maximum_center_distance = float(center_distance.max())
            if maximum_center_distance > 0:
                targets[batch_index, 1, rows, columns] = (
                    center_distance / maximum_center_distance
                )
            if maximum_boundary_distance > 0:
                targets[batch_index, 2, rows, columns] = (
                    boundary_distance[rows, columns] / maximum_boundary_distance
                )

    return (
        torch.from_numpy(targets).to(device=labels.device),
        torch.from_numpy(valid[:, None]).to(device=labels.device),
    )


class LearnedDistanceWatershedLoss(nn.Module):
    """Combine foreground BCE+Dice with foreground-masked distance regression."""

    def __init__(
        self,
        center_weight: float = 1.0,
        boundary_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if (
            not np.isfinite(center_weight)
            or not np.isfinite(boundary_weight)
            or center_weight < 0
            or boundary_weight < 0
        ):
            raise ValueError("Distance loss weights must be non-negative.")
        if center_weight + boundary_weight == 0:
            raise ValueError("At least one distance loss weight must be positive.")
        self.center_weight = center_weight
        self.boundary_weight = boundary_weight
        self.last_components: dict[str, float] = {}
        self.epoch_components: dict[str, float] = {}

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if predictions.ndim != 4 or predictions.shape[1] != 3:
            raise ValueError("Predictions must have shape (N, 3, H, W).")
        targets, valid = build_learned_distance_targets(labels)
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and generated targets must have equal shapes.")

        targets = targets.to(dtype=torch.float32)
        foreground_valid = valid.expand_as(targets[:, :1])
        if not torch.any(foreground_valid):
            return predictions.sum() * 0.0
        with torch.autocast(device_type=predictions.device.type, enabled=False):
            predictions = predictions.float()
            foreground_target = targets[:, :1]
            foreground_prediction = predictions[:, :1]
            bce = F.binary_cross_entropy(
                foreground_prediction[foreground_valid],
                foreground_target[foreground_valid],
            )
            intersection = (
                foreground_prediction * foreground_target * foreground_valid
            ).sum()
            dice_denominator = (
                (foreground_prediction + foreground_target) * foreground_valid
            ).sum()
            foreground_loss = bce + 1.0 - (
                2.0 * intersection + 1e-6
            ) / (dice_denominator + 1e-6)

            distance_valid = (targets[:, :1] > 0) & valid
            if torch.any(distance_valid):
                center_loss = F.smooth_l1_loss(
                    predictions[:, 1:2][distance_valid],
                    targets[:, 1:2][distance_valid],
                )
                boundary_loss = F.smooth_l1_loss(
                    predictions[:, 2:3][distance_valid],
                    targets[:, 2:3][distance_valid],
                )
            else:
                center_loss = predictions[:, 1:2].sum() * 0.0
                boundary_loss = predictions[:, 2:3].sum() * 0.0

        total = (
            foreground_loss
            + self.center_weight * center_loss
            + self.boundary_weight * boundary_loss
        )
        self.last_components = {
            "foreground_loss": float(foreground_loss.detach()),
            "center_distance_loss": float(center_loss.detach()),
            "boundary_distance_loss": float(boundary_loss.detach()),
            "loss": float(total.detach()),
        }
        return total


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be either 4 or 8.")
    return ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)


def reconstruct_learned_distance_watershed(
    predictions: torch.Tensor | np.ndarray,
    *,
    foreground_threshold: float = 0.5,
    center_threshold: float = 0.35,
    boundary_threshold: float = 0.5,
    minimum_marker_area: int = 1,
    minimum_instance_area: int = 0,
    connectivity: int = 8,
) -> np.ndarray:
    """Reconstruct one instance map from three foreground/geometry predictions."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim != 3 or predictions.shape[0] != 3:
        raise ValueError("Predictions must have shape (3, H, W).")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must contain only finite values.")
    for value, name in (
        (foreground_threshold, "foreground_threshold"),
        (center_threshold, "center_threshold"),
        (boundary_threshold, "boundary_threshold"),
    ):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1].")
    if (
        isinstance(minimum_marker_area, bool)
        or not isinstance(minimum_marker_area, (int, np.integer))
        or minimum_marker_area < 1
    ):
        raise ValueError("minimum_marker_area must be an integer >= 1.")
    if (
        isinstance(minimum_instance_area, bool)
        or not isinstance(minimum_instance_area, (int, np.integer))
        or minimum_instance_area < 0
    ):
        raise ValueError("minimum_instance_area must be an integer >= 0.")
    if np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("Predictions must be probabilities in [0, 1].")

    structure = _connectivity_structure(connectivity)
    foreground = predictions[0] > foreground_threshold
    if not foreground.any():
        return np.zeros(foreground.shape, dtype=np.int32)

    candidates = (
        foreground
        & (predictions[1] < center_threshold)
        & (predictions[2] > boundary_threshold)
    )
    candidate_labels, candidate_count = ndimage.label(candidates, structure)
    markers = np.zeros(foreground.shape, dtype=np.int32)
    next_marker = 1
    for candidate_id in range(1, candidate_count + 1):
        candidate = candidate_labels == candidate_id
        if int(candidate.sum()) >= minimum_marker_area:
            markers[candidate] = next_marker
            next_marker += 1

    foreground_labels, foreground_count = ndimage.label(foreground, structure)
    for component_id in range(1, foreground_count + 1):
        component = foreground_labels == component_id
        if np.any(markers[component]):
            continue
        coordinates = np.argwhere(component)
        minimum_index = np.argmin(predictions[1][component])
        row, column = coordinates[minimum_index]
        markers[row, column] = next_marker
        next_marker += 1

    labels = watershed(
        predictions[1],
        markers,
        mask=foreground,
        connectivity=structure,
    ).astype(np.int32, copy=False)
    if minimum_instance_area > 0:
        for instance_id in np.unique(labels):
            if instance_id > 0 and np.count_nonzero(labels == instance_id) < minimum_instance_area:
                labels[labels == instance_id] = 0
    relabeled = np.zeros_like(labels)
    for new_id, old_id in enumerate(np.unique(labels[labels > 0]), start=1):
        relabeled[labels == old_id] = new_id
    return relabeled


class UNet_LearnedDistance_Watershed(SemanticSegmentationUNet):
    """Classical U-Net predicting foreground, center distance, and boundary distance."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        *,
        foreground_threshold: float = 0.5,
        center_threshold: float = 0.35,
        boundary_threshold: float = 0.5,
        minimum_marker_area: int = 1,
        minimum_instance_area: int = 0,
        connectivity: int = 8,
    ) -> None:
        super().__init__(in_channels=in_channels, channels=channels)
        for value, name in (
            (foreground_threshold, "foreground_threshold"),
            (center_threshold, "center_threshold"),
            (boundary_threshold, "boundary_threshold"),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        for value, name, minimum in (
            (minimum_marker_area, "minimum_marker_area", 1),
            (minimum_instance_area, "minimum_instance_area", 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.output = nn.Conv2d(channels[0], 3, kernel_size=1)
        self.foreground_threshold = foreground_threshold
        self.center_threshold = center_threshold
        self.boundary_threshold = boundary_threshold
        self.minimum_marker_area = minimum_marker_area
        self.minimum_instance_area = minimum_instance_area
        self.connectivity = connectivity
        _connectivity_structure(connectivity)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities with shape (N, 3, H, W)."""
        return super().forward(inputs).sigmoid()

    @torch.no_grad()
    def reconstruct_instances(
        self, predictions: torch.Tensor | np.ndarray
    ) -> np.ndarray:
        """Convert one prediction or a batch of predictions into instance maps."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        predictions = np.asarray(predictions)
        if predictions.ndim == 3:
            return reconstruct_learned_distance_watershed(
                predictions,
                foreground_threshold=self.foreground_threshold,
                center_threshold=self.center_threshold,
                boundary_threshold=self.boundary_threshold,
                minimum_marker_area=self.minimum_marker_area,
                minimum_instance_area=self.minimum_instance_area,
                connectivity=self.connectivity,
            )
        if predictions.ndim == 4 and predictions.shape[0] > 0:
            return np.stack(
                [
                    self.reconstruct_instances(sample)
                    for sample in predictions
                ]
            )
        raise ValueError("Predictions must have shape (3, H, W) or (N, 3, H, W).")


LearnedDistanceWatershedUNet = UNet_LearnedDistance_Watershed


def eval_learned_distance_watershed_epoch(
    model: UNet_LearnedDistance_Watershed,
    device: torch.device,
    eval_loader: Any,
    loss_fn: nn.Module,
    *,
    amp: bool = False,
    progress_description: str = "Validation",
) -> dict[str, float]:
    """Evaluate target loss and instance metrics after learned-distance watershed."""
    from .eval import InstanceSegmentationMetrics

    model.eval()
    losses: list[float] = []
    metrics = InstanceSegmentationMetrics()
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=progress_description, leave=False):
            if len(batch) == 3:
                images, labels, _ = batch
            elif len(batch) == 2:
                images, labels = batch
            else:
                raise ValueError(
                    "Evaluation batch must contain images, labels, "
                    "and optional lengths."
                )
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                predictions = model(images)
                loss = loss_fn(predictions, labels)
            losses.append(float(loss.detach()))
            instance_maps = model.reconstruct_instances(predictions)
            if instance_maps.ndim == 2:
                instance_maps = instance_maps[None]
            metrics.update(
                {
                    "instance_map": torch.as_tensor(
                        instance_maps, device=device
                    )
                },
                labels,
            )
    result = {"loss": float(np.mean(losses)) if losses else 0.0}
    result.update(metrics.compute())
    return result
