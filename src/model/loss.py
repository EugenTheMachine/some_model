"""Permutation-invariant training objectives for instance segmentation."""

from __future__ import annotations

import math
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
    """Create same-instance targets and masks for non-ignored neighbor pairs."""
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
        pair_valid = (source >= 0) & (neighbor >= 0)
        target_view = targets[
            :, channel, y_start:y_stop, x_start:x_stop
        ]
        valid_view = valid[:, channel, y_start:y_stop, x_start:x_stop]
        target_view.copy_(
            ((source == neighbor) & (source > 0)).to(dtype=torch.float32)
        )
        valid_view.copy_(pair_valid)
    return targets, valid


def build_dela_affinity_targets(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise affinity only for in-bounds pairs of foreground pixels."""
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
        target_view = targets[
            :, channel, y_start:y_stop, x_start:x_stop
        ]
        valid_view = valid[:, channel, y_start:y_stop, x_start:x_stop]
        valid_pair = (source > 0) & (neighbor > 0)
        target_view.copy_((source == neighbor).to(dtype=torch.float32))
        valid_view.copy_(valid_pair)
    return targets, valid


def discriminative_embedding_loss(
    pixel_embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    max_pixels_per_instance: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-instance pull and inter-instance push losses."""
    if pixel_embeddings.ndim != 4:
        raise ValueError("Pixel embeddings must have shape (N, D, H, W).")
    label_map = _as_label_map(labels)
    if pixel_embeddings.shape[0] != label_map.shape[0]:
        raise ValueError("Embedding and label batch sizes must match.")
    if pixel_embeddings.shape[-2:] != label_map.shape[-2:]:
        raise ValueError("Embedding and label spatial dimensions must match.")
    if not math.isfinite(delta_v) or delta_v < 0:
        raise ValueError("delta_v must be finite and non-negative.")
    if not math.isfinite(delta_d) or delta_d < 0:
        raise ValueError("delta_d must be finite and non-negative.")
    if (
        isinstance(max_pixels_per_instance, bool)
        or not isinstance(max_pixels_per_instance, int)
        or max_pixels_per_instance <= 0
    ):
        raise ValueError("max_pixels_per_instance must be a positive integer.")

    pull_losses: list[torch.Tensor] = []
    push_losses: list[torch.Tensor] = []
    with torch.autocast(device_type=pixel_embeddings.device.type, enabled=False):
        embeddings = F.normalize(
            pixel_embeddings.float(), p=2, dim=1, eps=1e-8
        ).permute(0, 2, 3, 1)
        flat_embeddings = embeddings.reshape(
            embeddings.shape[0], -1, embeddings.shape[-1]
        )
        for batch_index, image_labels in enumerate(label_map):
            instance_ids = torch.unique(image_labels)
            instance_ids = instance_ids[instance_ids > 0]
            means: list[torch.Tensor] = []
            image_pull_losses: list[torch.Tensor] = []
            for instance_id in instance_ids:
                indices = torch.nonzero(
                    image_labels.reshape(-1) == instance_id,
                    as_tuple=False,
                ).flatten()
                instance_embeddings = flat_embeddings[batch_index].index_select(
                    0, indices
                )
                mean = F.normalize(
                    instance_embeddings.mean(dim=0), p=2, dim=0, eps=1e-8
                )
                means.append(mean)
                sample_count = min(
                    max_pixels_per_instance, int(indices.numel())
                )
                if sample_count < indices.numel():
                    sample_positions = torch.linspace(
                        0,
                        indices.numel() - 1,
                        steps=sample_count,
                        device=indices.device,
                    ).long()
                    instance_embeddings = flat_embeddings[batch_index].index_select(
                        0, indices.index_select(0, sample_positions)
                    )
                distances = torch.linalg.vector_norm(
                    instance_embeddings - mean.unsqueeze(0), dim=1
                )
                image_pull_losses.append(
                    F.relu(distances - delta_v).square().mean()
                )
            if image_pull_losses:
                pull_losses.append(torch.stack(image_pull_losses).mean())
            if len(means) > 1:
                pair_distances = torch.pdist(torch.stack(means), p=2)
                push_losses.append(
                    F.relu(2.0 * delta_d - pair_distances).square().mean()
                )

        zero = pixel_embeddings.float().sum() * 0.0
        pull_loss = torch.stack(pull_losses).mean() if pull_losses else zero
        push_loss = torch.stack(push_losses).mean() if push_losses else zero
    return pull_loss, push_loss


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
        with torch.autocast(
            device_type=assignment_probs.device.type,
            enabled=False,
        ):
            predicted_masks = assignment_probs[
                batch_index, 1:, valid_pixels
            ].detach().float().clamp(min=1e-6, max=1.0 - 1e-6)
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
                + (log_background - log_predictions)
                @ object_masks.transpose(0, 1)
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
        if not torch.isfinite(cost).all():
            raise ValueError(
                f"Non-finite Hungarian matching costs for image {batch_index}; "
                "check assignment probabilities for NaN or infinity."
            )
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
        if torch.any(assignment_targets != -1):
            assignment_loss = F.cross_entropy(
                outputs["assignment_logits"],
                assignment_targets,
                ignore_index=-1,
            )
        else:
            assignment_loss = outputs["assignment_logits"].sum() * 0
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


class DiscriminativeEmbeddingLearnedAssignmentLoss(nn.Module):
    """Train DELA foreground, affinity, embedding, and matched slot outputs."""

    def __init__(
        self,
        foreground_weight: float = 1.0,
        affinity_weight: float = 1.0,
        embedding_weight: float = 1.0,
        assignment_weight: float = 1.0,
        matched_mask_weight: float = 0.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        max_pixels_per_instance: int = 2048,
    ) -> None:
        super().__init__()
        weights = (
            foreground_weight,
            affinity_weight,
            embedding_weight,
            assignment_weight,
            matched_mask_weight,
        )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("DELA loss weights must be finite and non-negative.")
        if sum(weights[:4]) == 0:
            raise ValueError("At least one primary DELA loss weight must be positive.")
        if not math.isfinite(delta_v) or delta_v < 0:
            raise ValueError("delta_v must be finite and non-negative.")
        if not math.isfinite(delta_d) or delta_d < 0:
            raise ValueError("delta_d must be finite and non-negative.")
        if (
            isinstance(max_pixels_per_instance, bool)
            or not isinstance(max_pixels_per_instance, int)
            or max_pixels_per_instance <= 0
        ):
            raise ValueError("max_pixels_per_instance must be a positive integer.")
        self.foreground_weight = foreground_weight
        self.affinity_weight = affinity_weight
        self.embedding_weight = embedding_weight
        self.assignment_weight = assignment_weight
        self.matched_mask_weight = matched_mask_weight
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.max_pixels_per_instance = max_pixels_per_instance
        self.last_components: dict[str, float] = {}
        self.epoch_components: dict[str, float] = {}

    def forward(
        self, outputs: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """Return the weighted DELA objective and retain component diagnostics."""
        label_map = _as_label_map(labels)
        foreground_targets, foreground_valid = build_foreground_targets(label_map)
        foreground_logits = outputs["foreground_logits"]
        if foreground_valid.any():
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
            foreground_loss = foreground_bce + 1.0 - (
                2.0 * intersection + 1e-6
            ) / (dice_denominator + 1e-6)
        else:
            foreground_loss = foreground_logits.sum() * 0.0

        affinity_targets, affinity_valid = build_dela_affinity_targets(label_map)
        affinity_bce = F.binary_cross_entropy_with_logits(
            outputs["affinity_logits"], affinity_targets, reduction="none"
        )
        affinity_loss = (affinity_bce * affinity_valid).sum() / (
            affinity_valid.sum().clamp_min(1)
        )

        pull_loss, push_loss = discriminative_embedding_loss(
            outputs["pixel_embeddings"],
            label_map,
            delta_v=self.delta_v,
            delta_d=self.delta_d,
            max_pixels_per_instance=self.max_pixels_per_instance,
        )
        embedding_loss = pull_loss + push_loss
        assignment_targets, matchings = build_assignment_targets(
            outputs["assignment_probs"], label_map
        )
        if torch.any(assignment_targets != -1):
            assignment_loss = F.cross_entropy(
                outputs["assignment_logits"],
                assignment_targets,
                ignore_index=-1,
            )
        else:
            assignment_loss = outputs["assignment_logits"].sum() * 0

        mask_loss = outputs["assignment_logits"].new_zeros(())
        if self.matched_mask_weight:
            slot_probs = outputs["assignment_probs"][:, 1:]
            mask_losses: list[torch.Tensor] = []
            for batch_index, (slots, objects) in enumerate(matchings):
                image_labels = label_map[batch_index]
                instance_ids = torch.unique(image_labels)
                instance_ids = instance_ids[instance_ids > 0]
                valid_pixels = image_labels >= 0
                for slot_index, object_index in zip(slots, objects):
                    target = (
                        image_labels == instance_ids[object_index]
                    )[valid_pixels].to(dtype=torch.float32)
                    prediction = slot_probs[batch_index, slot_index][
                        valid_pixels
                    ].float().clamp(min=1e-6, max=1.0 - 1e-6)
                    bce = F.binary_cross_entropy(prediction, target)
                    dice = 1.0 - (
                        2.0 * (prediction * target).sum() + 1e-6
                    ) / (prediction.sum() + target.sum() + 1e-6)
                    mask_losses.append(bce + dice)
            if mask_losses:
                mask_loss = torch.stack(mask_losses).mean()

        total = (
            self.foreground_weight * foreground_loss
            + self.affinity_weight * affinity_loss
            + self.embedding_weight * embedding_loss
            + self.assignment_weight * assignment_loss
            + self.matched_mask_weight * mask_loss
        )
        self.last_components = {
            "foreground_loss": float(foreground_loss.detach()),
            "affinity_loss": float(affinity_loss.detach()),
            "embedding_pull_loss": float(pull_loss.detach()),
            "embedding_push_loss": float(push_loss.detach()),
            "assignment_loss": float(assignment_loss.detach()),
            "matched_mask_loss": float(mask_loss.detach()),
            "loss": float(total.detach()),
        }
        return total
