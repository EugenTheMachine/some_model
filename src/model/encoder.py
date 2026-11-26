"""U-Net encoder blocks and convolution helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def conv_norm_act(
    in_channels: int,
    out_channels: int,
    sz: int,
    norm: str | None = "BATCH",
    act: str | None = "GELU",
    depthwise: bool = False,
) -> nn.Sequential:
    """Create a same-resolution convolution, normalization, and activation."""
    if in_channels <= 0 or out_channels <= 0 or sz <= 0:
        raise ValueError("Convolution dimensions must be positive.")
    if depthwise and in_channels != out_channels:
        raise ValueError("Depthwise convolution requires equal channel counts.")

    if norm is None or norm.lower() == "none":
        norm_layer: nn.Module = nn.Identity()
    elif norm.lower() == "batch":
        norm_layer = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
    elif norm.lower() == "instance":
        norm_layer = nn.InstanceNorm2d(
            out_channels, eps=1e-5, track_running_stats=False, affine=True
        )
    elif norm.lower() in {"layer", "group"}:
        norm_layer = nn.GroupNorm(1, out_channels, eps=1e-5)
    else:
        raise ValueError(f"Unsupported normalization: {norm}")

    if act is None or act.lower() == "none":
        act_layer: nn.Module = nn.Identity()
    elif act.lower() == "gelu":
        act_layer = nn.GELU()
    elif act.lower() == "relu":
        act_layer = nn.ReLU(inplace=False)
    elif act.lower() in {"leakyrelu", "leaky_relu"}:
        act_layer = nn.LeakyReLU(negative_slope=0.01, inplace=False)
    elif act.lower() in {"silu", "swish"}:
        act_layer = nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {act}")

    groups = in_channels if depthwise else 1
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            sz,
            padding=sz // 2,
            groups=groups,
            bias=norm is None or norm.lower() == "none",
        ),
        norm_layer,
        act_layer,
    )


class EncoderBlock(nn.Module):
    """Residual convolutional encoder stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True,
        norm: str | None = "BATCH",
        act: str | None = "GELU",
        shallow: bool = False,
    ) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = (
            nn.Identity()
            if shallow
            else conv_norm_act(out_channels, out_channels, 3, norm, act)
        )
        self.conv3 = (
            nn.Identity()
            if shallow
            else conv_norm_act(out_channels, out_channels, 3, norm, act)
        )
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the pooled residual convolutional stage."""
        inputs = self.maxpool(inputs)
        projected = self.conv0(inputs)
        features = self.conv1(inputs)
        features = projected + self.conv2(features)
        return features + self.conv4(self.conv3(features))


class UNetEncoder(nn.Module):
    """Build encoder stages from shallow to deep."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        norm: str | None = "BATCH",
        act: str | None = "GELU",
    ) -> None:
        super().__init__()
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("Encoder channels must be a non-empty positive sequence.")
        if in_channels <= 0:
            raise ValueError("Input channels must be positive.")

        stages: list[nn.Module] = []
        current_channels = in_channels
        for index, output_channels in enumerate(channels):
            stages.append(
                EncoderBlock(
                    current_channels,
                    output_channels,
                    pool=index > 0,
                    norm=norm,
                    act=act,
                )
            )
            current_channels = output_channels
        self.stages = nn.ModuleList(stages)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the bottleneck and shallow-to-deep skip tensors."""
        divisor = 2 ** (len(self.stages) - 1)
        if inputs.ndim != 4:
            raise ValueError("Encoder input must have shape (N, C, H, W).")
        if inputs.shape[1] != self.stages[0].conv0[0].in_channels:
            raise ValueError(
                f"Expected {self.stages[0].conv0[0].in_channels} input channels, "
                f"received {inputs.shape[1]}."
            )
        if not inputs.is_floating_point():
            raise TypeError("Encoder input must be a floating-point tensor.")
        if inputs.shape[-2] % divisor or inputs.shape[-1] % divisor:
            raise ValueError(
                f"Input height and width must be divisible by {divisor}."
            )

        skips: list[torch.Tensor] = []
        features = inputs
        for index, stage in enumerate(self.stages):
            features = stage(features)
            if index < len(self.stages) - 1:
                skips.append(features)
        return features, skips
