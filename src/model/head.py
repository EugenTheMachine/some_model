"""Foreground, affinity, and instance-slot prediction heads."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


AFFINITY_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)


def _shift(
    tensor: torch.Tensor, dy: int, dx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a zero-padded neighbor shift and its in-bounds validity mask."""
    shifted = torch.roll(tensor, shifts=(-dy, -dx), dims=(-2, -1))
    height, width = tensor.shape[-2:]
    valid = torch.ones(
        (height, width), device=tensor.device, dtype=torch.bool
    )
    if dy > 0:
        valid[-dy:, :] = False
    elif dy < 0:
        valid[:-dy, :] = False
    if dx > 0:
        valid[:, -dx:] = False
    elif dx < 0:
        valid[:, :-dx] = False
    return shifted * valid.to(tensor.dtype), valid


class AffinityGuidedAggregation(nn.Module):
    """Aggregate neighbor features once using predicted directional affinities."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("Aggregation epsilon must be positive.")
        self.epsilon = epsilon

    def forward(
        self, features: torch.Tensor, affinities: torch.Tensor
    ) -> torch.Tensor:
        """Return the single affinity-weighted neighbor feature aggregate."""
        if features.ndim != 4 or affinities.ndim != 4:
            raise ValueError("Features and affinities must be NCHW tensors.")
        if affinities.shape[1] != len(AFFINITY_OFFSETS):
            raise ValueError("Affinity tensor must have exactly four channels.")
        if features.shape[0] != affinities.shape[0]:
            raise ValueError("Feature and affinity batch sizes must match.")
        if features.shape[-2:] != affinities.shape[-2:]:
            raise ValueError("Feature and affinity spatial dimensions must match.")

        weights = affinities.sigmoid()
        numerator = torch.zeros_like(features)
        denominator = torch.zeros_like(affinities[:, :1])
        for channel, (dy, dx) in enumerate(AFFINITY_OFFSETS):
            affinity = weights[:, channel : channel + 1]
            forward_features, forward_valid = _shift(features, dy, dx)
            reverse_features, reverse_valid = _shift(features, -dy, -dx)
            reverse_affinity, _ = _shift(affinity, -dy, -dx)
            forward_mask = forward_valid[None, None].to(features.dtype)
            reverse_mask = reverse_valid[None, None].to(features.dtype)

            forward_weight = affinity * forward_mask
            reverse_weight = reverse_affinity * reverse_mask
            numerator = (
                numerator
                + forward_features * forward_weight
                + reverse_features * reverse_weight
            )
            denominator = denominator + forward_weight + reverse_weight
        return numerator / (denominator + self.epsilon)


class InstanceSegmentationHead(nn.Module):
    """Predict foreground, affinities, pixel features, and slot assignments."""

    def __init__(
        self,
        in_channels: int,
        num_slots: int = 32,
        feature_channels: int = 32,
        hidden_channels: Sequence[int] = (64, 32),
        use_aggregation: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or num_slots <= 0 or feature_channels <= 0:
            raise ValueError("Model head channel counts must be positive.")
        if len(hidden_channels) != 2 or any(width <= 0 for width in hidden_channels):
            raise ValueError("The assignment MLP requires two positive widths.")
        self.num_slots = num_slots
        self.use_aggregation = use_aggregation

        self.foreground_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2 or 1, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // 2 or 1, 1, 1),
        )
        self.affinity_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2 or 1, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // 2 or 1, len(AFFINITY_OFFSETS), 1),
        )
        self.feature_projection = nn.Conv2d(
            in_channels, feature_channels, kernel_size=1
        )
        self.aggregation = AffinityGuidedAggregation()
        assignment_channels = feature_channels * (2 if use_aggregation else 1) + 5
        hidden_first, hidden_second = hidden_channels
        self.assignment_mlp = nn.Sequential(
            nn.Conv2d(assignment_channels, hidden_first, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_first, hidden_second, kernel_size=1),
            nn.ReLU(inplace=False),
        )
        self.background_classifier = nn.Conv2d(hidden_second, 1, 1)
        self.slot_classifier = nn.Conv2d(hidden_second, num_slots, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict foreground, affinities, and per-pixel slot assignments."""
        foreground_logits = self.foreground_head(features)
        affinity_logits = self.affinity_head(features)
        foreground_probs = foreground_logits.sigmoid()
        affinity_probs = affinity_logits.sigmoid()
        projected = self.feature_projection(features)
        if self.use_aggregation:
            aggregated = self.aggregation(projected, affinity_logits)
            assignment_input = torch.cat(
                (projected, aggregated, affinity_probs, foreground_probs), dim=1
            )
        else:
            aggregated = torch.zeros_like(projected)
            assignment_input = torch.cat(
                (projected, affinity_probs, foreground_probs), dim=1
            )
        hidden = self.assignment_mlp(assignment_input)
        assignment_logits = torch.cat(
            (
                self.background_classifier(hidden),
                self.slot_classifier(hidden),
            ),
            dim=1,
        )
        return {
            "foreground_logits": foreground_logits,
            "foreground_probs": foreground_probs,
            "affinity_logits": affinity_logits,
            "affinity_probs": affinity_probs,
            "pixel_features": projected,
            "neighbor_features": aggregated,
            "assignment_logits": assignment_logits,
            "assignment_probs": F.softmax(assignment_logits, dim=1),
            "instance_map": assignment_logits.argmax(dim=1),
        }
