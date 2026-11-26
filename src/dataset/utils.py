"""Dataset discovery, image/mask loading, and deterministic split utilities."""

# pylint: disable=no-member

from __future__ import annotations

import os
import random
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
from skimage import io as skimage_io

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SUPPORTED_MODALITIES = {"Brightfield", "phase-contrast"}
SPLIT_NAMES = {"train", "val", "test"}


def seed_worker(worker_id: int) -> None:
    """Seed each loader worker's Python, NumPy, and PyTorch generators."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(path)
    else:
        image = skimage_io.imread(path)
    image = np.asarray(image)
    if image.ndim not in {2, 3}:
        raise ValueError(f"Image must be 2D or 3D: {path} (shape {image.shape})")
    return image


def _read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".png":
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Could not read mask: {path}")
    else:
        mask = skimage_io.imread(path)
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Mask must be a 2D integer instance map: {path}")
    return mask.astype(np.int32, copy=False)


def _image_spatial_shape(image: np.ndarray) -> tuple[int, int]:
    if image.ndim == 2:
        return image.shape
    channel_axis = int(np.argmin(image.shape))
    spatial = [size for index, size in enumerate(image.shape) if index != channel_axis]
    return int(spatial[0]), int(spatial[1])


def _pair_files(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    masks = sorted(
        path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    mask_by_name = {path.name: path for path in masks}
    mask_by_stem = {path.stem: path for path in masks}
    pairs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for image_path in images:
        mask_path = mask_by_name.get(image_path.name)
        if mask_path is None:
            mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            mask_path = mask_by_name.get(f"mask_{image_path.stem}.png")
        if mask_path is None:
            missing.append(image_path)
        else:
            pairs.append((image_path, mask_path))
    if missing and len(images) == len(masks):
        already_paired = {image for image, _ in pairs}
        remaining_images = [image for image in images if image not in already_paired]
        used_masks = {mask for _, mask in pairs}
        remaining_masks = [mask for mask in masks if mask not in used_masks]
        pairs.extend(zip(remaining_images, remaining_masks))
    elif missing:
        raise ValueError(
            "Could not pair every image with a mask and image/mask counts differ: "
            f"{image_dir}, {mask_dir}"
        )
    if not pairs and (images or masks):
        raise ValueError(f"No valid image/mask pairs found in {image_dir} and {mask_dir}.")
    return sorted(pairs, key=lambda pair: pair[0].name)


def _resolve_split_dirs(split_dir: Path) -> tuple[Path, Path] | None:
    image_dir = split_dir / "images"
    if not image_dir.is_dir():
        return None
    for mask_name in ("masks", "labels"):
        mask_dir = split_dir / mask_name
        if mask_dir.is_dir():
            return image_dir, mask_dir
    return None


def _directory_pairs(
    root: Path, configured_modality: str | None
) -> dict[str, list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]]:
    found: dict[str, list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]] = {}
    for split in SPLIT_NAMES:
        split_dirs = _resolve_split_dirs(root / split)
        if split_dirs is not None:
            found[split] = _pair_files(*split_dirs)
    if not found:
        split_dirs = _resolve_split_dirs(root)
        if split_dirs is None:
            raise FileNotFoundError(
                f"Expected split/images with masks or labels directories under {root}."
            )
        found["train"] = _pair_files(*split_dirs)

    loaded: dict[str, list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]] = {}
    for split, pairs in found.items():
        loaded[split] = []
        for image_path, mask_path in pairs:
            image = _read_image(image_path)
            mask = _read_mask(mask_path)
            if _image_spatial_shape(image) != mask.shape:
                raise ValueError(
                    f"Image/mask dimensions differ for {image_path.name}: "
                    f"{_image_spatial_shape(image)} versus {mask.shape}"
                )
            inferred_modality = (
                "Brightfield"
                if image.ndim == 3 and 3 in image.shape
                else "phase-contrast"
            )
            modality = configured_modality or inferred_modality
            if modality not in SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unsupported modality {modality!r} for {image_path.name}."
                )
            if modality == "Brightfield" and (
                image.ndim != 3
                or image.shape[int(np.argmin(image.shape))] != 3
            ):
                modality = "phase-contrast"
            metadata = {
                "parent_dataset": root.name,
                "modality": modality,
                "image_modality": modality,
                "pixel_size": None,
                "name": image_path.name,
            }
            loaded[split].append((image, mask, metadata))
    return loaded


def _load_torch_dataset(
    dataset_path: Path,
    dataset_name: str,
    data_root: Path,
    source_dataset: list[str] | tuple[str, ...],
) -> dict[str, list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]]:
    if dataset_path.is_dir():
        dataset_path = (
            dataset_path / f"{dataset_name}_dataset.pth"
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"PyTorch dataset file not found: {dataset_path}")
    try:
        loaded = torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(dataset_path, map_location="cpu")
    if isinstance(loaded, dict) and any(
        split in loaded for split in SPLIT_NAMES
    ):
        raw_splits = {
            split: loaded.get(split, []) for split in SPLIT_NAMES
        }
    elif isinstance(loaded, dict):
        raw_splits = {"train": loaded.get("items", list(loaded.values()))}
        raw_splits["val"] = []
        raw_splits["test"] = []
    elif isinstance(loaded, (list, tuple)):
        raw_splits = {"train": loaded, "val": [], "test": []}
    else:
        raise ValueError(f"Unsupported dataset object in {dataset_path}.")

    result: dict[str, list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]] = {}
    for split, items in raw_splits.items():
        result[split] = []
        for item in items:
            if not isinstance(item, dict) or "cell_masks" not in item:
                continue
            if item.get("duplicate", False):
                continue
            metadata = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "image",
                    "cell_masks",
                    "nucleus_masks",
                    "class_masks",
                }
            }
            parent_dataset = metadata.get("parent_dataset")
            if list(source_dataset) != ["all"] and parent_dataset not in source_dataset:
                continue
            modality = metadata.get(
                "image_modality", metadata.get("modality")
            )
            if modality not in SUPPORTED_MODALITIES:
                raise ValueError(f"Unsupported dataset modality: {modality!r}")
            metadata["modality"] = modality
            metadata["image_modality"] = modality
            image_value = item["image"]
            if isinstance(image_value, (str, Path)):
                image_path = Path(image_value)
                if not image_path.is_absolute():
                    image_path = data_root / image_path
                if not image_path.is_file():
                    _inflate_referenced_file(image_path, data_root)
                image = _read_image(image_path)
            elif isinstance(image_value, torch.Tensor):
                image = image_value.detach().cpu().numpy()
            else:
                image = np.asarray(image_value)
            mask_value = item["cell_masks"]
            if isinstance(mask_value, torch.Tensor):
                mask = mask_value.detach().cpu().numpy()
            else:
                mask = np.asarray(mask_value)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 2:
                raise ValueError("Each .pth cell_masks value must be a 2D map.")
            if _image_spatial_shape(image) != mask.shape:
                raise ValueError("A .pth image and its cell_masks have different dimensions.")
            result[split].append(
                (image, mask.astype(np.int32, copy=False), metadata)
            )
    return result


def _inflate_referenced_file(image_path: Path, data_root: Path) -> None:
    """Extract only the requested member from a matching archive."""
    for archive_path in sorted(data_root.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            member = next(
                (
                    name
                    for name in archive.namelist()
                    if Path(name).name == image_path.name
                ),
                None,
            )
            if member is None:
                continue
            image_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, image_path.open("wb") as target:
                target.write(source.read())
            return
    raise FileNotFoundError(
        f"Referenced image {image_path} was not found or present in a matching zip."
    )


def _take_training_fraction(
    items: list[Any], ratio: float, seed: int
) -> list[Any]:
    if not 0 < ratio <= 1:
        raise ValueError("train_data_ratio must be in (0, 1].")
    if ratio == 1 or not items:
        return items
    count = max(1, int(round(len(items) * ratio)))
    selected = np.random.RandomState(seed).permutation(len(items))[:count]
    return [items[int(index)] for index in sorted(selected)]


def _split_training_validation(
    train_items: list[Any], validation_ratio: float, seed: int
) -> tuple[list[Any], list[Any]]:
    if not 0 <= validation_ratio <= 1:
        raise ValueError("val_ratio must be in [0, 1].")
    if len(train_items) <= 1:
        return train_items, []
    validation_count = max(1, round(len(train_items) * validation_ratio))
    validation_count = min(validation_count, len(train_items) - 1)
    permutation = np.random.RandomState(seed).permutation(len(train_items))
    validation_indices = set(int(index) for index in permutation[:validation_count])
    train = [item for index, item in enumerate(train_items) if index not in validation_indices]
    validation = [item for index, item in enumerate(train_items) if index in validation_indices]
    return train, validation


def _pack_items(
    items: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    return (
        [item[0] for item in items],
        [item[1] for item in items],
        [item[2] for item in items],
    )


def load_dataset_splits(
    config: dict[str, Any]
) -> dict[str, tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]]:
    """Preload all retained images and masks into CPU memory before loaders."""
    data_path = Path(config["data_dir"]).expanduser()
    if not data_path.is_absolute():
        data_path = (Path(__file__).resolve().parents[2] / data_path).resolve()
    dataset_argument = str(config.get("dataset", ""))
    if data_path.suffix.lower() == ".pth" or dataset_argument.endswith(".pth"):
        default_root = Path(os.environ.setdefault("INSTANSEG_DATASET_PATH", str(data_path.parent)))
        dataset_path = (
            data_path
            if data_path.suffix.lower() == ".pth"
            else default_root / dataset_argument
        )
        if dataset_path.suffix.lower() != ".pth":
            dataset_path = default_root / f"{dataset_argument}_dataset.pth"
        items_by_split = _load_torch_dataset(
            dataset_path,
            dataset_argument,
            default_root,
            config.get("source_dataset", ["all"]),
        )
    else:
        if data_path.name in SPLIT_NAMES and (data_path / "images").is_dir():
            root = data_path.parent
        else:
            root = data_path
        items_by_split = _directory_pairs(root, config.get("modality"))

    seed = int(config.get("seed", 42))
    train_items = _take_training_fraction(
        items_by_split.get("train", []),
        float(config.get("train_data_ratio", 1.0)),
        seed,
    )
    if "val" not in items_by_split or not items_by_split["val"]:
        train_items, val_items = _split_training_validation(
            train_items, float(config.get("val_ratio", 0.2)), seed
        )
    else:
        val_items = items_by_split["val"]
    test_items = items_by_split.get("test") or val_items
    packed = {
        "train": _pack_items(train_items),
        "val": _pack_items(val_items),
        "test": _pack_items(test_items),
    }
    if not packed["train"][0]:
        raise ValueError("No training image/mask pairs were found.")
    if data_path.suffix.lower() == ".pth":
        for split, (images, _, _) in packed.items():
            if split != "train" and not images:
                raise ValueError(f"The requested {split} .pth split is empty.")
    return packed
