"""Permutation-invariant training objectives for instance segmentation."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .head import AFFINITY_OFFSETS


def _as_label_map(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError("Labels must have shape (N, H, W) or (N, 1, H, W).")
    return labels.to(dtype=torch.int64)


def build_foreground_targets(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create foreground labels and a mask excluding ignored pixels."""
    label_map = _as_label_map(labels)
    valid = label_map >= 0
    targets = (label_map > 0).to(dtype=torch.float32)
    return targets.unsqueeze(1), valid.unsqueeze(1)


def build_affinity_targets(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create same-instance targets and valid foreground pair masks."""
    label_map = _as_label_map(labels)
    batch_size, height, width = label_map.shape
    targets = torch.zeros(
        (batch_size, len(AFFINITY_OFFSETS), height, width),
        device=label_map.device,
        dtype=torch.float32,
    )
    valid = torch.zeros_like(targets, dtype=torch.bool)
    for channel, (dy, dx) in enumerate(AFFINITY_OFFSETS):
        y_start, y_stop = max(0, -dy), min(height, height - dy)
        x_start, x_stop = max(0, -dx), min(width, width - dx)
        source = label_map[:, y_start:y_stop, x_start:x_stop]
        neighbor = label_map[
            :,
            y_start + dy : y_stop + dy,
            x_start + dx : x_stop + dx,
        ]
        pair_valid = (source > 0) & (neighbor > 0)
        target_view = targets[
            :, channel, y_start:y_stop, x_start:x_stop
        ]
        valid_view = valid[:, channel, y_start:y_stop, x_start:x_stop]
        target_view.copy_((source == neighbor).to(dtype=torch.float32))
        valid_view.copy_(pair_valid)
    return targets, valid


def build_assignment_targets(
    assignment_probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> tuple[torch.Tensor, list[tuple[np.ndarray, np.ndarray]]]:
    """Match predicted foreground slots to object masks with Hungarian assignment."""
    label_map = _as_label_map(labels)
    if assignment_probs.ndim != 4 or assignment_probs.shape[0] != label_map.shape[0]:
        raise ValueError("Assignment probabilities must have shape (N, K+1, H, W).")
    if assignment_probs.shape[-2:] != label_map.shape[-2:]:
        raise ValueError("Assignment probabilities and labels must have equal spatial sizes.")
    if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight == 0:
        raise ValueError("Matching cost weights must be non-negative and non-zero.")

    num_slots = assignment_probs.shape[1] - 1
    targets = torch.zeros_like(label_map, dtype=torch.long)
    targets[label_map < 0] = -1
    matchings: list[tuple[np.ndarray, np.ndarray]] = []
    for batch_index, image_labels in enumerate(label_map):
        instance_ids = torch.unique(image_labels)
        instance_ids = instance_ids[instance_ids > 0]
        if instance_ids.numel() > num_slots:
            raise ValueError(
                f"Image {batch_index} contains {instance_ids.numel()} instances, "
                f"exceeding the configured slot capacity of {num_slots}."
            )
        if instance_ids.numel() == 0:
            matchings.append(
                (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            )
            continue

        valid_pixels = image_labels >= 0
        predicted_masks = assignment_probs[
            batch_index, 1:, valid_pixels
        ].clamp(min=1e-6, max=1.0 - 1e-6)
        object_masks = torch.stack(
            [
                (image_labels[valid_pixels] == instance_id).to(
                    dtype=predicted_masks.dtype
                )
                for instance_id in instance_ids
            ],
            dim=0,
        )
        log_predictions = torch.log(predicted_masks)
        log_background = torch.log1p(-predicted_masks)
        num_valid_pixels = max(1, int(valid_pixels.sum()))
        bce_cost = (
            -log_background.sum(dim=-1, keepdim=True)
            + (log_background - log_predictions) @ object_masks.transpose(0, 1)
        ) / num_valid_pixels
        intersection = predicted_masks @ object_masks.transpose(0, 1)
        denominator = (
            predicted_masks.sum(dim=-1, keepdim=True)
            + object_masks.sum(dim=-1).unsqueeze(0)
        )
        dice_cost = 1.0 - (2.0 * intersection + 1e-6) / (
            denominator + 1e-6
        )
        cost = bce_weight * bce_cost + dice_weight * dice_cost
        predicted_indices, object_indices = linear_sum_assignment(
            cost.detach().cpu().numpy()
        )
        predicted_tensor = torch.as_tensor(
            predicted_indices, device=label_map.device, dtype=torch.long
        )
        object_tensor = torch.as_tensor(
            object_indices, device=label_map.device, dtype=torch.long
        )
        for predicted_slot, object_index in zip(
            predicted_tensor, object_tensor
        ):
            target_pixels = image_labels == instance_ids[object_index]
            targets[batch_index, target_pixels] = predicted_slot + 1
        matchings.append((predicted_indices, object_indices))
    return targets, matchings


class AffinityGuidedLoss(nn.Module):
    """Combine foreground, masked affinity, and Hungarian assignment losses."""

    def __init__(
        self,
        foreground_weight: float = 1.0,
        affinity_weight: float = 1.0,
        assignment_weight: float = 1.0,
        matched_mask_weight: float = 0.0,
    ) -> None:
        super().__init__()
        weights = (
            foreground_weight,
            affinity_weight,
            assignment_weight,
            matched_mask_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Loss weights must be non-negative.")
        if sum(weights[:3]) == 0:
            raise ValueError("At least one primary loss weight must be positive.")
        self.foreground_weight = foreground_weight
        self.affinity_weight = affinity_weight
        self.assignment_weight = assignment_weight
        self.matched_mask_weight = matched_mask_weight
        self.last_components: dict[str, float] = {}
        self.epoch_components: dict[str, float] = {}

    def forward(
        self, outputs: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """Return the weighted foreground, affinity, and assignment objective."""
        label_map = _as_label_map(labels)
        foreground_targets, foreground_valid = build_foreground_targets(label_map)
        foreground_logits = outputs["foreground_logits"]
        foreground_bce = F.binary_cross_entropy_with_logits(
            foreground_logits,
            foreground_targets,
            reduction="none",
        )
        foreground_bce = (
            foreground_bce * foreground_valid
        ).sum() / foreground_valid.sum().clamp_min(1)
        foreground_probs = foreground_logits.sigmoid()
        valid_float = foreground_valid.to(dtype=foreground_probs.dtype)
        intersection = (
            foreground_probs * foreground_targets * valid_float
        ).sum()
        dice_denominator = (
            (foreground_probs + foreground_targets) * valid_float
        ).sum()
        foreground_dice = 1.0 - (2.0 * intersection + 1e-6) / (
            dice_denominator + 1e-6
        )
        foreground_loss = foreground_bce + foreground_dice

        affinity_targets, affinity_valid = build_affinity_targets(label_map)
        affinity_bce = F.binary_cross_entropy_with_logits(
            outputs["affinity_logits"], affinity_targets, reduction="none"
        )
        affinity_loss = (affinity_bce * affinity_valid).sum() / (
            affinity_valid.sum().clamp_min(1)
        )

        assignment_targets, matchings = build_assignment_targets(
            outputs["assignment_probs"], label_map
        )
        assignment_loss = F.cross_entropy(
            outputs["assignment_logits"],
            assignment_targets,
            ignore_index=-1,
        )
        mask_loss = outputs["assignment_logits"].new_zeros(())
        if self.matched_mask_weight and matchings:
            slot_probs = outputs["assignment_probs"][:, 1:]
            losses: list[torch.Tensor] = []
            for batch_index, (slots, objects) in enumerate(matchings):
                image_labels = label_map[batch_index]
                ids = torch.unique(image_labels)
                ids = ids[ids > 0]
                for slot_index, object_index in zip(slots, objects):
                    target = (image_labels == ids[object_index]).to(
                        dtype=slot_probs.dtype
                    )
                    prediction = slot_probs[batch_index, slot_index]
                    bce = F.binary_cross_entropy(prediction, target)
                    dice = 1.0 - (
                        2.0 * (prediction * target).sum() + 1e-6
                    ) / (prediction.sum() + target.sum() + 1e-6)
                    losses.append(bce + dice)
            if losses:
                mask_loss = torch.stack(losses).mean()

        total = (
            self.foreground_weight * foreground_loss
            + self.affinity_weight * affinity_loss
            + self.assignment_weight * assignment_loss
            + self.matched_mask_weight * mask_loss
        )
        self.last_components = {
            "foreground_loss": float(foreground_loss.detach()),
            "affinity_loss": float(affinity_loss.detach()),
            "assignment_loss": float(assignment_loss.detach()),
            "matched_mask_loss": float(mask_loss.detach()),
            "loss": float(total.detach()),
        }
        return total
