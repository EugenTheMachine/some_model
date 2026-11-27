"""In-memory image/mask dataset and variable-channel batch collation."""

# pylint: disable=invalid-name

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import Augmentations


class Segmentation_Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Keep source arrays in CPU memory and augment on access."""

    def __init__(
        self,
        img: Sequence[np.ndarray | torch.Tensor],
        label: Sequence[np.ndarray | torch.Tensor],
        common_transforms: bool = True,
        metadata: Sequence[dict[str, Any] | None] | None = None,
        size: tuple[int, int] = (256, 256),
        augmentation_dict: dict[str, Any] | None = None,
        dim_in: int = 3,
        *,
        debug: bool = False,
        channel_invariant: bool = False,
        random_seed: int | None = None,
        mode: str = "train",
        augmentation_type: str = "minimal",
        binarize_masks: bool = False,
    ) -> None:
        self.X = list(img)
        self.Y = list(label)
        self.common_transforms = common_transforms
        self.binarize_masks = binarize_masks
        if len(self.X) != len(self.Y):
            raise ValueError("The number of images and labels must be the same.")
        self.metadata = (
            list(metadata)
            if metadata
            else [None] * len(self.X)
        )
        if len(self.X) != len(self.metadata):
            raise ValueError("The number of images and metadata must be the same.")
        if len(size) != 2 or any(dimension <= 0 for dimension in size):
            raise ValueError("size must contain two positive dimensions.")
        if mode not in {"train", "test"}:
            raise ValueError("mode must be either 'train' or 'test'.")
        self.size = tuple(int(dimension) for dimension in size)
        self.Augmenter = Augmentations(
            augmentation_dict=augmentation_dict,
            debug=debug,
            shape=self.size,
            dim_in=dim_in,
            channel_invariant=channel_invariant,
            random_seed=random_seed,
            mode=mode,
            augmentation_type=augmentation_type,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.X[index]
        label = self.Y[index]
        metadata = self.metadata[index]
        if self.common_transforms:
            data, label = self.Augmenter(data, label, metadata)
        else:
            data, label = self.Augmenter.to_tensor(data, label)

        if label.ndim == 2:
            label = label.unsqueeze(0)
        if data.ndim == 2:
            data = data.unsqueeze(0)
        if data.ndim != 3 or label.ndim != 3:
            raise ValueError("Transformed image and label must be CHW tensors.")
        if self.binarize_masks:
            label = (label != 0).to(dtype=torch.int32)
        if data.shape[-2:] != label.shape[-2:]:
            raise ValueError("Transformed image and label dimensions differ.")
        if not torch.isfinite(data).all():
            raise ValueError("Transformed image contains NaN or infinity.")
        if torch.is_floating_point(label) and not torch.isfinite(label).all():
            raise ValueError("Transformed label contains NaN or infinity.")
        return data.to(dtype=torch.float32), label.to(dtype=torch.int32)


def collate_fn(
    data: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a batch and pad channel counts up to the batch maximum."""
    if not data:
        raise ValueError("Cannot collate an empty batch.")
    images, labels = zip(*data)
    if any(image.ndim != 3 for image in images):
        raise ValueError("Each image must have shape (C, H, W).")
    if any(label.ndim == 2 for label in labels):
        labels = tuple(
            label.unsqueeze(0) if label.ndim == 2 else label for label in labels
        )
    if any(label.ndim != 3 or label.shape[0] != 1 for label in labels):
        raise ValueError("Each label must have shape (1, H, W).")
    height, width = images[0].shape[-2:]
    if any(image.shape[-2:] != (height, width) for image in images):
        raise ValueError("Images in a batch must have equal spatial dimensions.")
    if any(label.shape[-2:] != (height, width) for label in labels):
        raise ValueError("Labels and images in a batch must have equal dimensions.")
    lengths = torch.tensor([image.shape[0] for image in images], dtype=torch.int32)
    max_channels = int(lengths.max())
    batch_images = torch.zeros(
        (len(images), max_channels, height, width), dtype=torch.float32
    )
    for index, image in enumerate(images):
        batch_images[index, : image.shape[0]] = image.to(dtype=torch.float32)
    batch_labels = torch.stack(
        [label.to(dtype=torch.int32) for label in labels]
    )
    return batch_images, batch_labels, lengths
