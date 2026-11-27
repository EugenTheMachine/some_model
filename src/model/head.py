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


class SpatialConvGRU(nn.Module):
    """Shared spatial GRU cell used for recurrent affinity message passing."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("ConvGRU channel counts must be positive.")
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self, recurrent_input: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Apply one ConvGRU update to a full-resolution feature map."""
        combined = torch.cat((recurrent_input, hidden), dim=1)
        reset, update = self.gates(combined).chunk(2, dim=1)
        reset = reset.sigmoid()
        update = update.sigmoid()
        candidate = torch.tanh(
            self.candidate(
                torch.cat((recurrent_input, reset * hidden), dim=1)
            )
        )
        return (1.0 - update) * hidden + update * candidate


class AffinityGuidedRecurrentGrouping(nn.Module):
    """Exchange pixel features along learned affinity-weighted neighbors."""

    def __init__(
        self,
        pixel_channels: int,
        foreground_channels: int = 1,
        affinity_channels: int = 4,
        hidden_channels: int = 32,
        iterations: int = 3,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(pixel_channels, hidden_channels, iterations) <= 0:
            raise ValueError("Recurrent grouping dimensions must be positive.")
        if epsilon <= 0:
            raise ValueError("Message epsilon must be positive.")
        if affinity_channels != len(AFFINITY_OFFSETS):
            raise ValueError("The recurrent grouping requires four affinity channels.")

        self.iterations = iterations
        self.epsilon = epsilon
        recurrent_input_channels = (
            pixel_channels + foreground_channels + affinity_channels
        )
        self.input_projection = nn.Sequential(
            nn.Conv2d(recurrent_input_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.message_transforms = nn.ModuleList(
            [
                nn.Conv2d(hidden_channels, hidden_channels, 1)
                for _ in AFFINITY_OFFSETS
            ]
        )
        self.reverse_message_transforms = nn.ModuleList(
            [
                nn.Conv2d(hidden_channels, hidden_channels, 1)
                for _ in AFFINITY_OFFSETS
            ]
        )
        self.gru = SpatialConvGRU(
            input_channels=hidden_channels + recurrent_input_channels,
            hidden_channels=hidden_channels,
        )

    def forward(
        self,
        pixel_features: torch.Tensor,
        foreground_probs: torch.Tensor,
        affinity_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Refine full-resolution pixel features through recurrent messages."""
        if pixel_features.shape[0] != foreground_probs.shape[0]:
            raise ValueError("Pixel and foreground batch sizes must match.")
        if pixel_features.shape[-2:] != foreground_probs.shape[-2:]:
            raise ValueError("Pixel and foreground spatial dimensions must match.")
        recurrent_input = torch.cat(
            (pixel_features, foreground_probs, affinity_probs), dim=1
        )
        hidden = self.input_projection(recurrent_input)

        for _ in range(self.iterations):
            message = torch.zeros_like(hidden)
            weight_sum = torch.zeros(
                (hidden.shape[0], 1, *hidden.shape[-2:]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            for channel, (dy, dx) in enumerate(AFFINITY_OFFSETS):
                forward_affinity = affinity_probs[:, channel : channel + 1]
                forward_features, forward_valid = _shift(hidden, dy, dx)
                reverse_affinity, _ = _shift(
                    forward_affinity, -dy, -dx
                )
                reverse_features, reverse_valid = _shift(hidden, -dy, -dx)

                forward_weight = (
                    forward_affinity
                    * forward_valid.to(hidden.dtype)[None, None]
                )
                reverse_weight = (
                    reverse_affinity
                    * reverse_valid.to(hidden.dtype)[None, None]
                )
                message = message + (
                    self.message_transforms[channel](forward_features)
                    * forward_weight
                )
                message = message + (
                    self.reverse_message_transforms[channel](reverse_features)
                    * reverse_weight
                )
                weight_sum = weight_sum + forward_weight + reverse_weight
            message = message / (weight_sum + self.epsilon)
            gru_input = torch.cat((message, recurrent_input), dim=1)
            updated = self.gru(gru_input, hidden)
            hidden = updated
        return hidden


class RecurrentSlotGrouping(nn.Module):
    """Iteratively refine learned instance slots by aggregating pixel features."""

    def __init__(
        self,
        pixel_channels: int,
        num_slots: int,
        slot_iterations: int = 2,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if pixel_channels <= 0 or num_slots <= 0 or slot_iterations <= 0:
            raise ValueError("Slot grouping dimensions must be positive.")
        if epsilon <= 0:
            raise ValueError("Slot aggregation epsilon must be positive.")
        self.num_slots = num_slots
        self.slot_iterations = slot_iterations
        self.epsilon = epsilon
        self.slot_embeddings = nn.Parameter(
            torch.randn(1, num_slots, pixel_channels)
            / (pixel_channels**0.5)
        )
        self.query = nn.Conv2d(pixel_channels, pixel_channels, 1)
        self.key = nn.Linear(pixel_channels, pixel_channels)
        self.value = nn.Conv2d(pixel_channels, pixel_channels, 1)
        self.slot_gru = nn.GRUCell(pixel_channels, pixel_channels)
        self.background_classifier = nn.Sequential(
            nn.Conv2d(pixel_channels + 1, pixel_channels, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(pixel_channels, 1, 1),
        )
        self.slot_bias = nn.Parameter(torch.zeros(num_slots))

    def forward(
        self,
        recurrent_features: torch.Tensor,
        foreground_probs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict background and fixed-slot logits for every pixel."""
        batch_size, channels, height, width = recurrent_features.shape
        if foreground_probs.shape != (
            batch_size,
            1,
            height,
            width,
        ):
            raise ValueError(
                "Foreground probabilities must match recurrent spatial features."
            )
        scale = channels**0.5
        queries = self.query(recurrent_features).flatten(2).transpose(1, 2)
        values = self.value(recurrent_features).flatten(2).transpose(1, 2)
        slots = self.slot_embeddings.expand(batch_size, -1, -1)

        for _ in range(self.slot_iterations):
            keys = self.key(slots)
            compatibility = torch.bmm(
                queries,
                keys.transpose(1, 2),
            ) / scale
            assignment_weights = compatibility.softmax(dim=-1)
            foreground_weights = foreground_probs.flatten(2).transpose(1, 2)
            weighted = assignment_weights * foreground_weights
            slot_updates = torch.bmm(
                weighted.transpose(1, 2),
                values,
            ) / (weighted.sum(dim=1).unsqueeze(-1) + self.epsilon)
            slots = self.slot_gru(
                slot_updates.reshape(batch_size * self.num_slots, channels),
                slots.reshape(batch_size * self.num_slots, channels),
            ).reshape(batch_size, self.num_slots, channels)

        keys = self.key(slots)
        compatibility = torch.bmm(
            queries,
            keys.transpose(1, 2),
        ) / scale
        slot_logits = compatibility.transpose(1, 2).reshape(
            batch_size,
            self.num_slots,
            height,
            width,
        )
        slot_logits = slot_logits + self.slot_bias.view(1, -1, 1, 1)
        background_logits = self.background_classifier(
            torch.cat((recurrent_features, foreground_probs), dim=1)
        )
        assignment_logits = torch.cat(
            (background_logits, slot_logits),
            dim=1,
        )
        return {
            "slot_embeddings": slots,
            "assignment_logits": assignment_logits,
            "assignment_probs": assignment_logits.softmax(dim=1),
        }


class AffinityGuidedRecurrentHead(nn.Module):
    """Full recurrent grouping and learned fixed-slot assignment head."""

    def __init__(
        self,
        in_channels: int,
        num_slots: int = 32,
        feature_channels: int = 32,
        recurrent_hidden_channels: int = 32,
        recurrent_iterations: int = 3,
        slot_iterations: int = 2,
        minimum_instance_area: int = 0,
    ) -> None:
        super().__init__()
        if min(
            in_channels,
            num_slots,
            feature_channels,
            recurrent_hidden_channels,
            recurrent_iterations,
            slot_iterations,
        ) <= 0:
            raise ValueError("Recurrent head channel counts must be positive.")
        if minimum_instance_area < 0:
            raise ValueError("minimum_instance_area must be non-negative.")

        self.num_slots = num_slots
        self.minimum_instance_area = minimum_instance_area
        self.foreground_head = nn.Sequential(
            nn.Conv2d(in_channels, max(1, in_channels // 2), 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(1, in_channels // 2), 1, 1),
        )
        self.affinity_head = nn.Sequential(
            nn.Conv2d(in_channels, max(1, in_channels // 2), 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(1, in_channels // 2), len(AFFINITY_OFFSETS), 1),
        )
        self.feature_projection = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 1),
            nn.ReLU(inplace=False),
        )
        self.recurrent = AffinityGuidedRecurrentGrouping(
            pixel_channels=feature_channels,
            hidden_channels=recurrent_hidden_channels,
            iterations=recurrent_iterations,
        )
        self.slots = RecurrentSlotGrouping(
            pixel_channels=recurrent_hidden_channels,
            num_slots=num_slots,
            slot_iterations=slot_iterations,
        )

    def _compact_instance_map(
        self, raw_instance_map: torch.Tensor
    ) -> torch.Tensor:
        """Remove unused slots and compact positive IDs in first-seen slot order."""
        compacted = torch.zeros_like(raw_instance_map)
        for batch_index in range(raw_instance_map.shape[0]):
            positive_slots = torch.unique(raw_instance_map[batch_index])
            positive_slots = positive_slots[positive_slots > 0]
            next_id = 1
            for slot in positive_slots.tolist():
                mask = raw_instance_map[batch_index] == slot
                if (
                    self.minimum_instance_area
                    and int(mask.sum()) < self.minimum_instance_area
                ):
                    continue
                compacted[batch_index, mask] = next_id
                next_id += 1
        return compacted

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run dense predictions, recurrent grouping, and slot assignment."""
        foreground_logits = self.foreground_head(features)
        affinity_logits = self.affinity_head(features)
        foreground_probs = foreground_logits.sigmoid()
        affinity_probs = affinity_logits.sigmoid()
        pixel_features = self.feature_projection(features)
        recurrent_features = self.recurrent(
            pixel_features,
            foreground_probs,
            affinity_probs,
        )
        slot_output = self.slots(recurrent_features, foreground_probs)
        assignment_logits = slot_output["assignment_logits"]
        assignment_probs = slot_output["assignment_probs"]
        raw_instance_map = assignment_logits.argmax(dim=1)
        instance_map = self._compact_instance_map(raw_instance_map)
        return {
            "foreground_logits": foreground_logits,
            "foreground_probs": foreground_probs,
            "affinity_logits": affinity_logits,
            "affinity_probs": affinity_probs,
            "pixel_features": pixel_features,
            "recurrent_features": recurrent_features,
            "slot_embeddings": slot_output["slot_embeddings"],
            "assignment_logits": assignment_logits,
            "assignment_probs": assignment_probs,
            "instance_map": instance_map,
        }


ConvGRU2d = SpatialConvGRU
AffinityGuidedGrouping = AffinityGuidedRecurrentGrouping
InstanceSlotGrouping = RecurrentSlotGrouping
