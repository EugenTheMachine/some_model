"""Paired image/instance-mask preprocessing and spatial augmentation."""

# pylint: disable=invalid-name

from __future__ import annotations

import collections
import io
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _move_channel_axis(image: np.ndarray | torch.Tensor, to_back: bool = False):
    """Move the smallest 3D axis to channel-first or channel-last."""
    if image.ndim == 2:
        image = image.unsqueeze(0) if isinstance(image, torch.Tensor) else image[None]
    if image.ndim != 3:
        raise ValueError("Input image must be 2D or 3D.")
    channel_axis = int(np.argmin(image.shape))
    if isinstance(image, torch.Tensor):
        return image.movedim(channel_axis, -1 if to_back else 0)
    if to_back:
        return np.moveaxis(image, channel_axis, -1)
    return np.moveaxis(image, channel_axis, 0)


def get_augmentation_dict(
    dim_in: int,
    amount: float,
    *,
    pixel_size: float = 0.5,
    augmentation_type: str = "minimal",
    mean_diameter: float | None = None,
) -> dict[str, dict[str, collections.OrderedDict[str, list[Any]]]]:
    """Build the ordered train/test transforms for each supported modality."""
    del dim_in
    heavy_diameter = (
        (mean_diameter, 2**-2, 2**2) if mean_diameter is not None else None
    )
    light_diameter = (
        (mean_diameter, 2**-1, 2**1) if mean_diameter is not None else None
    )
    if augmentation_type in {"two_channel", "colourize"}:
        raise ValueError(f"Augmentation type {augmentation_type!r} is unavailable.")
    supported = {"minimal", "kornia_intensity", "heavy", "brightfield_only"}
    if augmentation_type not in supported:
        raise ValueError(f"Unsupported augmentation type: {augmentation_type!r}")

    def base_test(diameter: Any) -> collections.OrderedDict[str, list[Any]]:
        return collections.OrderedDict(
            [
                ("to_tensor", [1]),
                ("normalize", [1]),
                ("torch_rescale", [1, pixel_size, diameter]),
            ]
        )

    if augmentation_type in {"minimal", "kornia_intensity"}:
        train_bf = base_test(light_diameter)
        train_bf["flips"] = [1]
        train_bf["rotate"] = [1]
        test_bf = base_test(light_diameter)
        test_bf["flips"] = [1]
        if augmentation_type == "kornia_intensity":
            train_bf["kornia_base_augmentations"] = [1]
        train_pc = train_bf.copy()
        test_pc = test_bf.copy()
    elif augmentation_type == "heavy":
        train_bf = collections.OrderedDict(
            [
                ("to_tensor", [1]),
                ("normalize_HE_stains", [0.1, amount * 0]),
                ("extract_hematoxylin_stain", [0.1, amount * 0]),
                ("normalize", [1]),
                ("torch_rescale", [1, pixel_size, heavy_diameter]),
                ("randomJPEGcompression", [0.2, amount]),
                ("adjust_hue", [0.2, amount]),
                ("AdjustContrast", [0.2, amount]),
                ("flips", [1]),
                ("rotate", [1]),
                ("brightness_augment", [0.2, amount]),
                ("RandGaussianNoise", [0.2, amount]),
                ("perspective", [0, amount]),
            ]
        )
        train_pc = collections.OrderedDict(
            [
                ("to_tensor", [1]),
                ("normalize", [1]),
                ("torch_rescale", [1, pixel_size, heavy_diameter]),
                ("flips", [1]),
                ("rotate", [1]),
                ("kornia_base_augmentations", [1]),
            ]
        )
        test_bf = base_test(light_diameter)
        test_pc = base_test(light_diameter)
        test_bf["flips"] = [1]
        test_pc["flips"] = [1]
    else:
        train_bf = collections.OrderedDict(
            [
                ("to_tensor", [1]),
                ("normalize_HE_stains", [0.1, amount * 0]),
                ("extract_hematoxylin_stain", [0.1, amount * 0]),
                ("normalize", [1]),
                ("torch_rescale", [1, pixel_size, 0]),
                ("randomJPEGcompression", [0.2, amount]),
                ("adjust_hue", [0.2, amount]),
                ("AdjustContrast", [0.2, amount]),
                ("flips", [1]),
                ("rotate", [1]),
                ("brightness_augment", [0.2, amount]),
                ("RandGaussianNoise", [0.2, amount]),
                ("perspective", [0.1, amount]),
            ]
        )
        train_pc = train_bf.copy()
        test_bf = base_test(0)
        test_pc = base_test(0)
        test_bf["flips"] = [1]
        test_pc["flips"] = [1]
    return {
        "train": {"Brightfield": train_bf, "phase-contrast": train_pc},
        "test": {"Brightfield": test_bf, "phase-contrast": test_pc},
    }


class Augmentations:
    """Apply ordered, modality-aware transforms to an image and its mask."""

    def __init__(
        self,
        augmentation_dict: Mapping[str, Any] | None = None,
        debug: bool = False,
        shape: tuple[int, int] | None = (256, 256),
        dim_in: int = 3,
        channel_invariant: bool = False,
        random_seed: int | None = None,
        mode: str = "train",
        augmentation_type: str = "minimal",
        modality: str | None = None,
    ) -> None:
        if mode not in {"train", "test"}:
            raise ValueError("mode must be either 'train' or 'test'.")
        self.debug = debug
        self.shape = shape
        self.dim_in = dim_in
        self.channel_invariant = channel_invariant
        self.mode = mode
        self.modality = modality
        self.augmentation_dict = augmentation_dict or get_augmentation_dict(
            dim_in, 0.2, augmentation_type=augmentation_type
        )
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def _select_pipeline(self, modality: str) -> Mapping[str, list[Any]]:
        config: Any = self.augmentation_dict
        if "train" in config or "test" in config:
            config = config[self.mode]
        if modality in config:
            config = config[modality]
        return config

    def __call__(
        self,
        image: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = metadata or {}
        modality = (
            self.modality
            or metadata.get("image_modality")
            or metadata.get("modality")
        )
        if modality is None:
            modality = (
                "Brightfield"
                if image.ndim == 3 and 3 in image.shape
                else "phase-contrast"
            )
        channel_count = 1 if image.ndim == 2 else int(np.argmin(image.shape))
        if modality == "Brightfield" and (
            image.ndim != 3 or image.shape[channel_count] != 3
        ):
            modality = "phase-contrast"
        if modality not in {"Brightfield", "phase-contrast"}:
            raise ValueError(f"Unsupported image modality: {modality!r}")

        current_image: Any = image
        current_labels: Any = labels
        normalized = False
        for transform_name, values in self._select_pipeline(modality).items():
            probability, *arguments = values
            if np.random.random() >= probability:
                continue
            if transform_name in {
                "normalize",
                "normalize_HE_stains",
                "extract_hematoxylin_stain",
            }:
                if normalized:
                    continue
                previous_image = current_image
                current_image, current_labels = getattr(self, transform_name)(
                    current_image,
                    current_labels,
                    amount=arguments[0] if arguments else None,
                    metadata=metadata,
                )
                normalized = (
                    transform_name == "normalize"
                    or current_image is not previous_image
                )
            elif transform_name == "torch_rescale":
                requested_pixel_size = arguments[0] if arguments else None
                diameter_range = arguments[1] if len(arguments) > 1 else None
                current_image, current_labels = self.torch_rescale(
                    current_image,
                    current_labels,
                    requested_pixel_size=requested_pixel_size,
                    diameter_range=diameter_range,
                    metadata=metadata,
                    modality=modality,
                )
            else:
                amount = arguments[0] if arguments else None
                current_image, current_labels = getattr(self, transform_name)(
                    current_image,
                    current_labels,
                    amount=amount,
                    metadata=metadata,
                )
        current_image, current_labels = self.to_tensor(
            current_image, current_labels
        )
        current_image, current_labels = self.duplicate_grayscale_channels(
            current_image, current_labels, metadata
        )
        if self.shape is not None and current_image.shape[-2:] != self.shape:
            current_image, current_labels = self._resize_crop_pad(
                current_image, current_labels, modality
            )
        if current_image.var() > 1e2:
            current_image = torch.clamp(current_image, min=-1, max=5)
        if not torch.isfinite(current_image).all():
            raise ValueError("Augmentation produced invalid image values.")
        return current_image.float(), current_labels.to(torch.int32)

    def to_tensor(
        self,
        image: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor | None = None,
        normalize: bool = False,
        amount: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Convert image and labels to channel-first tensors."""
        del amount, metadata
        if isinstance(image, np.ndarray):
            image_tensor = torch.as_tensor(
                np.array(_move_channel_axis(image), copy=True)
            )
        elif isinstance(image, torch.Tensor):
            image_tensor = _move_channel_axis(image)
        else:
            raise TypeError("Image must be a NumPy array or PyTorch tensor.")
        image_tensor = image_tensor.to(dtype=torch.float32)
        label_tensor: torch.Tensor | None = None
        if labels is not None:
            if isinstance(labels, np.ndarray):
                label_tensor = torch.as_tensor(np.array(labels, copy=True))
            elif isinstance(labels, torch.Tensor):
                label_tensor = labels.detach().clone()
            else:
                raise TypeError("Labels must be a NumPy array or PyTorch tensor.")
            if label_tensor.dtype == torch.bool or torch.is_floating_point(label_tensor):
                raise TypeError("Instance labels must use an integer dtype.")
            label_tensor = label_tensor.to(torch.int64)
            if torch.any(label_tensor < -1) or torch.any(
                label_tensor > torch.iinfo(torch.int32).max
            ):
                raise ValueError("Instance labels must fit int32 and be at least -1.")
            if label_tensor.ndim == 3 and label_tensor.shape[0] == 1:
                pass
            elif label_tensor.ndim == 3:
                label_tensor = _move_channel_axis(label_tensor)
            elif label_tensor.ndim == 2:
                label_tensor = label_tensor.unsqueeze(0)
            else:
                raise ValueError("Labels must be a 2D instance map.")
            label_tensor = label_tensor.to(dtype=torch.int32)
            if not torch.any(label_tensor == -1):
                ids = torch.unique(label_tensor)
                positive_ids = ids[ids > 0]
                remapped = torch.zeros_like(label_tensor)
                for new_id, old_id in enumerate(positive_ids, start=1):
                    remapped[label_tensor == old_id] = new_id
                label_tensor = remapped
        if image_tensor.ndim != 3:
            raise ValueError("Image must resolve to a (C, H, W) tensor.")
        if normalize:
            image_tensor = self._percentile_normalize(image_tensor)
        return image_tensor.contiguous(), label_tensor

    @staticmethod
    def _percentile_normalize(image: torch.Tensor) -> torch.Tensor:
        """Normalize every image channel using robust intensity percentiles."""
        normalized = torch.empty_like(image, dtype=torch.float32)
        for channel in range(image.shape[0]):
            pixels = image[channel].float()
            lower = torch.quantile(pixels, 0.001)
            upper = torch.quantile(pixels, 0.999)
            normalized[channel] = (pixels - lower) / (upper - lower).clamp_min(1e-3)
        return normalized

    def normalize(
        self,
        image: Any,
        labels: Any = None,
        amount: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Apply percentile normalization while retaining the instance map."""
        del amount, metadata, kwargs
        tensor, labels = self.to_tensor(image, labels)
        return self._percentile_normalize(tensor), labels

    def normalize_HE_stains(
        self, image: Any, labels: Any = None, amount: Any = None, metadata: Any = None
    ) -> tuple[Any, Any]:
        """Normalize H&E stains with a perturbed Macenko reference."""
        return self._stain_transform(
            image, labels, amount, metadata, hematoxylin_only=False
        )

    def extract_hematoxylin_stain(
        self, image: Any, labels: Any = None, amount: Any = None, metadata: Any = None
    ) -> tuple[Any, Any]:
        """Extract hematoxylin with a perturbed Macenko reference."""
        return self._stain_transform(
            image, labels, amount, metadata, hematoxylin_only=True
        )

    @staticmethod
    def _stain_transform(
        image: Any,
        labels: Any,
        amount: Any,
        metadata: Any,
        *,
        hematoxylin_only: bool,
    ) -> tuple[Any, Any]:
        """Apply Macenko normalization and leave input unchanged on failure."""
        if (
            metadata is None
            or metadata.get("image_modality", metadata.get("modality"))
            != "Brightfield"
            or not isinstance(image, torch.Tensor)
            or image.ndim != 3
            or image.shape[0] != 3
        ):
            return image, labels
        try:
            import torchstain

            stain_image = image / (image.max() + 0.001) * 255
            stain_image = stain_image.clamp(0, 255)
            normalizer = torchstain.normalizers.MacenkoNormalizer(
                backend="torch"
            )
            normalizer.fit(stain_image)
            perturbation = float(amount or 0)
            normalizer.HERef += (
                torch.rand_like(normalizer.HERef) - 0.5
            ) * normalizer.HERef * perturbation
            normalizer.maxCRef += (
                torch.rand_like(normalizer.maxCRef) - 0.5
            ) * normalizer.maxCRef * perturbation
            normalized, hematoxylin, _ = normalizer.normalize(
                stain_image, stains=True
            )
            result = hematoxylin if hematoxylin_only else normalized
            if result.ndim == 3 and result.shape[-1] == 3:
                result = result.movedim(-1, 0)
            return result / 255, labels
        except Exception:  # pylint: disable=broad-exception-caught
            return image, labels

    def torch_rescale(
        self,
        image: Any,
        labels: Any = None,
        amount: float | None = 0.0,
        current_pixel_size: float | None = None,
        requested_pixel_size: float | None = None,
        crop: bool = True,
        random_seed: int | None = None,
        metadata: dict[str, Any] | None = None,
        diameter_range: tuple[float, float, float] | None = None,
        modality: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Rescale and crop/pad an image and its mask with aligned geometry."""
        del crop, random_seed
        image_tensor, label_tensor = self.to_tensor(image, labels)
        scale = 1.0
        if (
            current_pixel_size is None
            and metadata is not None
            and "pixel_size" in metadata
        ):
            current_pixel_size = metadata["pixel_size"]
        if current_pixel_size is not None and requested_pixel_size is not None:
            scale = current_pixel_size / requested_pixel_size
        if amount:
            scale *= 1 + (torch.rand(1).item() - 0.5) * amount
        if diameter_range is not None and label_tensor is not None:
            object_areas = [
                int((label_tensor == instance_id).sum())
                for instance_id in torch.unique(label_tensor)
                if instance_id > 0
            ]
            if object_areas:
                original_diameter = 2 * np.sqrt(
                    float(np.median(object_areas)) / np.pi
                )
                lower, upper = np.log2(diameter_range[1:])
                target = diameter_range[0] * 2 ** random.uniform(lower, upper)
                scale = target / max(original_diameter, 1e-6)
        if scale != 1.0:
            output_size = (
                max(1, round(image_tensor.shape[-2] * scale)),
                max(1, round(image_tensor.shape[-1] * scale)),
            )
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=output_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            if label_tensor is not None:
                label_tensor = F.interpolate(
                    label_tensor.float().unsqueeze(0),
                    size=output_size,
                    mode="nearest",
                )[0].to(torch.int32)
        if self.shape is not None:
            image_tensor, label_tensor = self._resize_crop_pad(
                image_tensor, label_tensor, modality or "phase-contrast"
            )
        return image_tensor, label_tensor

    def _resize_crop_pad(
        self,
        image: torch.Tensor,
        labels: torch.Tensor | None,
        modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pad and crop paired tensors to the configured output dimensions."""
        if self.shape is None:
            return image, labels
        target_height, target_width = self.shape
        height, width = image.shape[-2:]
        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)
        if pad_height or pad_width:
            top = pad_height // 2
            bottom = pad_height - top
            left = pad_width // 2
            right = pad_width - left
            fill = image.max() if modality == "Brightfield" else image.min()
            image = F.pad(
                image, (left, right, top, bottom), value=float(fill)
            )
            if labels is not None:
                labels = F.pad(labels, (left, right, top, bottom), value=0)
            height, width = image.shape[-2:]
        if height > target_height:
            top = random.randint(0, height - target_height)
        else:
            top = 0
        if width > target_width:
            left = random.randint(0, width - target_width)
        else:
            left = 0
        image = image[:, top : top + target_height, left : left + target_width]
        if labels is not None:
            labels = labels[
                :, top : top + target_height, left : left + target_width
            ]
        return image, labels

    def flips(
        self, image: torch.Tensor, labels: torch.Tensor, amount: Any = None, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same random horizontal and vertical flips to both tensors."""
        del amount, metadata
        if random.random() > 0.5:
            image = image.flip(-1)
            labels = labels.flip(-1)
        if random.random() > 0.5:
            image = image.flip(-2)
            labels = labels.flip(-2)
        return image, labels

    def rotate(
        self, image: torch.Tensor, labels: torch.Tensor, amount: Any = None, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate paired tensors by a random multiple of 90 degrees."""
        del amount, metadata
        if self.mode == "train":
            turns = int(np.random.choice((0, 1, 2, 3)))
            image = torch.rot90(image, turns, (-2, -1))
            labels = torch.rot90(labels, turns, (-2, -1))
        return image, labels

    def randomJPEGcompression(
        self, image: torch.Tensor, labels: torch.Tensor, amount: float = 0.2, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply JPEG compression to an RGB image without changing its mask."""
        del metadata
        if image.shape[0] != 3:
            return image, labels
        maximum = float(image.max())
        minimum = float(image.min())
        scaled = ((image - minimum) / max(maximum - minimum, 1e-6) * 255).byte()
        pil_image = Image.fromarray(scaled.permute(1, 2, 0).cpu().numpy())
        buffer = io.BytesIO()
        quality = random.randrange(
            int(100 * (1 - amount) - 1), 100
        )
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        restored = torch.from_numpy(
            np.asarray(Image.open(buffer).convert("RGB")).copy()
        ).permute(2, 0, 1)
        return restored.float() / 255 * (maximum - minimum) + minimum, labels

    def brightness_augment(
        self, image: torch.Tensor, labels: torch.Tensor, amount: float = 0.2, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a random per-channel brightness multiplier."""
        del metadata
        factors = 1 + (torch.rand(image.shape[0]) - 0.5) * amount
        return image * factors[:, None, None], labels

    def RandGaussianNoise(
        self, image: torch.Tensor, labels: torch.Tensor, amount: float = 0.2, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add zero-mean Gaussian noise to the image."""
        del metadata
        return image + torch.randn_like(image) * (amount / 5), labels

    def RandStdShiftIntensity(
        self, image: torch.Tensor, labels: torch.Tensor, amount: Any = None, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a random intensity offset independently to each channel."""
        del amount, metadata
        from monai.transforms import RandStdShiftIntensity

        transform = RandStdShiftIntensity(10)
        return torch.stack([transform(channel) for channel in image]), labels

    def AdjustContrast(
        self, image: torch.Tensor, labels: torch.Tensor, amount: float = 1.0, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adjust contrast around each channel's spatial mean."""
        del metadata
        mean = image.mean(dim=(-2, -1), keepdim=True)
        gamma = 1.0 if amount is None else float(amount)
        return (image - mean) * gamma + mean, labels

    def adjust_hue(
        self, image: torch.Tensor, labels: torch.Tensor, amount: float = 0.2, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shift the hue of RGB images while leaving labels unchanged."""
        del metadata
        if image.shape[0] != 3:
            return image, labels
        if amount is None:
            return image, labels
        from torchvision.transforms import functional as vision_functional

        hue_factor = (torch.rand(1).item() - 0.5) * amount * 0.5
        transformed = vision_functional.adjust_hue(image, hue_factor)
        return torch.nan_to_num(transformed), labels

    def perspective(
        self, image: torch.Tensor, labels: torch.Tensor, amount: float = 0.0, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a shared perspective transform with nearest-neighbor labels."""
        modality = (
            metadata.get("image_modality", metadata.get("modality"))
            if metadata is not None
            else None
        )
        if not amount:
            return image, labels
        height, width = image.shape[-2:]
        from torchvision.transforms import InterpolationMode, RandomPerspective
        from torchvision.transforms import functional as vision_functional

        startpoints, endpoints = RandomPerspective.get_params(
            width, height, distortion_scale=amount / 2
        )
        image_out = vision_functional.perspective(
            image,
            startpoints,
            endpoints,
            interpolation=InterpolationMode.BILINEAR,
            fill=float(
                image.max() if modality == "Brightfield" else image.min()
            ),
        )
        label_out = vision_functional.perspective(
            labels,
            startpoints,
            endpoints,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        return image_out, label_out

    def gradient(
        self, image: torch.Tensor, labels: torch.Tensor, amount: float = 0.2, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a smooth spatial intensity gradient to the image."""
        del metadata
        height, width = image.shape[-2:]
        x = torch.linspace(0, 0.5, width, device=image.device)
        y = torch.linspace(0, 0.5, height, device=image.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        if random.random() > 0.5:
            grid_x = grid_x.flip(-1)
        if random.random() > 0.5:
            grid_x = grid_x.flip(-2)
        if random.random() > 0.5:
            grid_y = grid_y.flip(-1)
        if random.random() > 0.5:
            grid_y = grid_y.flip(-2)
        gradient = random.random() * grid_x + random.random() * grid_y
        return image + amount * gradient.unsqueeze(0) * image.max(), labels

    def inversion(
        self, image: torch.Tensor, labels: torch.Tensor, amount: Any = None, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert image intensity without modifying instance labels."""
        del amount, metadata
        return 1 - image, labels

    def kornia_base_augmentations(
        self, image: torch.Tensor, labels: torch.Tensor, amount: Any = None, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the configured Kornia intensity sequence to one image."""
        del amount, metadata
        from kornia import augmentation

        low, high = image.amin(), image.amax()
        normalized = (image - low) / (high - low).clamp_min(1e-6)
        pipeline = torch.nn.Sequential(
            augmentation.RandomLinearIllumination(gain=(0.1, 0.4), p=1),
            augmentation.RandomPlasmaContrast(p=0.5),
            augmentation.RandomSaltAndPepperNoise(),
            augmentation.RandomMedianBlur(p=0.2),
            augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
            augmentation.RandomBoxBlur((3, 3), p=0.2),
            augmentation.RandomSharpness(sharpness=1, p=0.5),
            augmentation.RandomContrast(contrast=(1, 1.5), p=0.5),
            augmentation.RandomGamma(
                gamma=(0.5, 0.5), gain=(0.5, 1.5), p=0.5
            ),
        )
        augmented = pipeline(normalized.unsqueeze(0))[0]
        return augmented * (high - low) + low, labels

    def duplicate_grayscale_channels(
        self, image: torch.Tensor, labels: torch.Tensor, metadata: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Duplicate grayscale input channels or validate the configured width."""
        del metadata
        if self.dim_in == 1 and image.shape[0] == 3:
            image = (
                image[0:1] * 0.299
                + image[1:2] * 0.587
                + image[2:3] * 0.114
            )
        if self.channel_invariant and image.shape[0] == self.dim_in:
            return image, labels
        if self.channel_invariant:
            raise ValueError(
                f"Channel-invariant input has {image.shape[0]} channels, "
                f"but the model expects {self.dim_in}."
            )
        if image.shape[0] == 1 and self.dim_in != 1:
            image = image.repeat(self.dim_in, 1, 1)
        elif image.shape[0] != self.dim_in:
            raise ValueError(
                f"Image has {image.shape[0]} channels, but model expects {self.dim_in}."
            )
        return image, labels
