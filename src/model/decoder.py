"""U-Net decoder blocks and output projections."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import conv_norm_act


class DecoderBlock(nn.Module):
    """Upsample a feature map and fuse its matching encoder skip."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str | None = "BATCH",
        act: str | None = "ReLU",
        shallow: bool = False,
        upsample_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        self.conv_skip = (
            conv_norm_act(skip_channels, out_channels, 1, norm, act)
            if skip_channels > 0
            else None
        )
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = (
            nn.Identity()
            if shallow
            else conv_norm_act(out_channels, out_channels, 3, norm, act)
        )
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

    def forward(
        self, features: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Upsample, fuse an optional skip, and apply residual convolutions."""
        if self.upsample_mode == "bilinear":
            features = F.interpolate(
                features,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
        else:
            features = F.interpolate(
                features, scale_factor=2, mode="nearest"
            )
        projected = self.conv0(features)
        features = self.conv1(features)
        if skip is not None and self.conv_skip is not None:
            if skip.shape[-2:] != features.shape[-2:]:
                raise ValueError(
                    "Decoder skip spatial dimensions do not match the upsampled features."
                )
            features = projected + self.conv2(features + self.conv_skip(skip))
        else:
            features = projected + self.conv2(features)
        return features + self.conv4(self.conv3(features))


class Decoder(nn.Module):
    """Decode a bottleneck and emit one or more output projections."""

    def __init__(
        self,
        layers: Sequence[int],
        out_channels: Sequence[int],
        norm: str | None = "BATCH",
        act: str | None = "ReLU",
        decoder_block: type[nn.Module] = DecoderBlock,
        upsample_mode: str = "nearest",
    ) -> None:
        super().__init__()
        if len(layers) < 2:
            raise ValueError("Decoder requires at least two channel widths.")
        if not out_channels or any(channel <= 0 for channel in out_channels):
            raise ValueError("Decoder output channels must be positive.")
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    layers[index],
                    layers[index + 1],
                    layers[index + 1],
                    norm=norm,
                    act=act,
                    upsample_mode=upsample_mode,
                )
                if decoder_block is DecoderBlock
                else decoder_block(
                    layers[index],
                    layers[index + 1],
                    layers[index + 1],
                    norm=norm,
                    act=act,
                )
                for index in range(len(layers) - 1)
            ]
        )
        projection_norm = (
            None if norm is not None and norm.lower() == "instance" else norm
        )
        self.final_block = nn.ModuleList(
            [
                conv_norm_act(
                    layers[-1], output_channels, 1, projection_norm, act=None
                )
                for output_channels in out_channels
            ]
        )

    def forward(
        self, features: torch.Tensor, skips: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Decode skips and concatenate the requested output projections."""
        if len(skips) < len(self.decoder):
            raise ValueError(
                f"Decoder needs {len(self.decoder)} skip tensors; got {len(skips)}."
            )
        for layer, skip in zip(self.decoder, reversed(skips)):
            features = layer(features, skip)
        return torch.cat(
            [projection(features) for projection in self.final_block], dim=1
        )


class UNetDecoderFeatures(nn.Module):
    """Decoder that returns features rather than task-specific predictions."""

    def __init__(
        self,
        channels: Sequence[int],
        norm: str | None = "BATCH",
        act: str | None = "ReLU",
    ) -> None:
        super().__init__()
        descending = list(reversed(channels))
        if len(descending) < 2:
            raise ValueError("U-Net decoder requires at least two stages.")
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    descending[index],
                    descending[index + 1],
                    descending[index + 1],
                    norm=norm,
                    act=act,
                )
                for index in range(len(descending) - 1)
            ]
        )
        self.out_channels = channels[0]

    def forward(
        self, features: torch.Tensor, skips: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Decode the bottleneck and return the final feature representation."""
        if len(skips) != len(self.layers):
            raise ValueError(
                f"Decoder expected {len(self.layers)} skips; got {len(skips)}."
            )
        for layer, skip in zip(self.layers, reversed(skips)):
            features = layer(features, skip)
        return features
