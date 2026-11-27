"""Classical U-Net and training utilities for binary semantic segmentation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


class DoubleConv(nn.Module):
    """Two padded 3x3 convolutions with batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SemanticSegmentationUNet(nn.Module):
    """Encoder-decoder U-Net with skip concatenations and foreground logits."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if len(channels) < 2 or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain at least two positive widths.")

        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for width in channels:
            self.encoders.append(DoubleConv(current_channels, width))
            current_channels = width
        self.pools = nn.ModuleList(
            nn.MaxPool2d(kernel_size=2, stride=2)
            for _ in channels[:-1]
        )
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(channels[:-1]):
            self.upconvs.append(
                nn.ConvTranspose2d(current_channels, width, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(width * 2, width))
            current_channels = width
        self.output = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.input_channels = in_channels
        self.downsampling_factor = 2 ** (len(channels) - 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one foreground logit per input pixel."""
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"Input must have shape (N, {self.input_channels}, H, W)."
            )
        if not inputs.is_floating_point():
            raise TypeError("Model input must be a floating-point tensor.")
        if any(size % self.downsampling_factor for size in inputs.shape[-2:]):
            raise ValueError(
                "Input height and width must be divisible by "
                f"{self.downsampling_factor}."
            )

        skips: list[torch.Tensor] = []
        features = inputs
        for index, encoder in enumerate(self.encoders):
            features = encoder(features)
            if index < len(self.pools):
                skips.append(features)
                features = self.pools[index](features)

        for upconv, decoder, skip in zip(
            self.upconvs, self.decoders, reversed(skips)
        ):
            features = upconv(features)
            features = decoder(torch.cat((skip, features), dim=1))
        return self.output(features)


class SemanticSegmentationLoss(nn.Module):
    """Binary cross-entropy and soft Dice loss for foreground logits."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 3:
            labels = labels.unsqueeze(1)
        if logits.shape != labels.shape:
            raise ValueError("Semantic logits and masks must have equal shapes.")
        valid = labels >= 0
        targets = (labels > 0).to(dtype=logits.dtype)
        if not torch.any(valid):
            return logits.sum() * 0.0
        bce = F.binary_cross_entropy_with_logits(
            logits[valid], targets[valid]
        )
        probabilities = logits.sigmoid()
        intersection = ((probabilities * targets) * valid).sum()
        dice = 1.0 - (2.0 * intersection + 1e-6) / (
            (probabilities * valid).sum() + (targets * valid).sum() + 1e-6
        )
        return bce + dice


def eval_semantic_epoch(
    model: nn.Module,
    device: torch.device,
    eval_loader: Any,
    loss_fn: nn.Module,
    *,
    amp: bool = False,
    progress_description: str = "Validation",
) -> dict[str, float]:
    """Evaluate semantic segmentation loss and pixel-level metrics."""
    model.eval()
    losses: list[float] = []
    intersection = 0
    predicted_count = 0
    target_count = 0
    correct = 0
    pixel_count = 0
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
                logits = model(images)
                loss = loss_fn(logits, labels)
            losses.append(float(loss.detach()))
            predictions = logits.sigmoid() >= 0.5
            targets = labels.to(dtype=torch.bool)
            intersection += int((predictions & targets).sum())
            predicted_count += int(predictions.sum())
            target_count += int(targets.sum())
            correct += int((predictions == targets).sum())
            pixel_count += targets.numel()

    epsilon = 1e-8
    union = predicted_count + target_count - intersection
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "pixel_accuracy": correct / max(1, pixel_count),
        "dice": 2 * intersection / (predicted_count + target_count + epsilon),
        "iou": intersection / (union + epsilon),
    }
