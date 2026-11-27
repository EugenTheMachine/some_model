"""Classical U-Net foreground prediction with distance-transform watershed reconstruction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .semantic import SemanticSegmentationUNet


def _as_probability_map(probability_map: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(probability_map, torch.Tensor):
        array = probability_map.detach().cpu().numpy()
    else:
        array = np.asarray(probability_map)
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        else:
            raise ValueError("A probability map must contain exactly one channel.")
    if array.ndim != 2:
        raise ValueError("A probability map must be two-dimensional.")
    return array.astype(np.float32, copy=False)


def _connectivity_footprint(connectivity: int) -> np.ndarray:
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be either 4 or 8.")
    if connectivity == 4:
        return np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    return np.ones((3, 3), dtype=bool)


def extract_watershed_markers(
    distance_map: torch.Tensor | np.ndarray,
    *,
    mask: torch.Tensor | np.ndarray | None = None,
    min_distance: int = 5,
    min_height: float | None = None,
    prominence: float | None = None,
    gaussian_sigma: float = 0.0,
    connectivity: int = 8,
    minimum_instance_area: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique integer marker labels from a distance map."""
    if isinstance(distance_map, torch.Tensor):
        distance = distance_map.detach().cpu().numpy()
    else:
        distance = np.asarray(distance_map)
    if distance.ndim != 2:
        raise ValueError("distance_map must be two-dimensional.")
    distance = distance.astype(np.float32, copy=False)
    if isinstance(min_distance, bool) or not isinstance(min_distance, int) or min_distance < 1:
        raise ValueError("min_distance must be an integer >= 1.")
    if gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be non-negative.")
    if isinstance(minimum_instance_area, bool) or not isinstance(minimum_instance_area, int) or minimum_instance_area < 0:
        raise ValueError("minimum_instance_area must be an integer >= 0.")

    if gaussian_sigma > 0:
        distance = gaussian_filter(distance, sigma=gaussian_sigma)

    if mask is None:
        mask_array = distance > 0
    else:
        if isinstance(mask, torch.Tensor):
            mask_array = mask.detach().cpu().numpy()
        else:
            mask_array = np.asarray(mask)
        mask_array = mask_array.astype(bool)
    if mask_array.ndim == 3:
        mask_array = mask_array.reshape(mask_array.shape[-2:])
    if mask_array.shape != distance.shape:
        raise ValueError("mask and distance_map must have equal shapes.")

    peaks = peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_abs=min_height,
        labels=mask_array.astype(np.uint8),
        footprint=_connectivity_footprint(connectivity),
    )
    if prominence is not None:
        if prominence < 0:
            raise ValueError("prominence must be non-negative.")
        peaks = np.asarray(
            [
                peak
                for peak in peaks
                if distance[tuple(peak)]
                >= prominence
            ],
            dtype=np.int64,
        ).reshape((-1, 2))

    markers = np.zeros(distance.shape, dtype=np.int32)
    for marker_id, (row, column) in enumerate(peaks, start=1):
        if minimum_instance_area == 0 or distance[row, column] > 0:
            markers[row, column] = marker_id
    return markers, distance


def distance_transform_watershed(
    probability_map: torch.Tensor | np.ndarray,
    *,
    threshold: float = 0.5,
    foreground_threshold: float | None = None,
    min_distance: int = 5,
    peak_min_distance: int | None = None,
    min_peak_distance: int | None = None,
    min_height: float | None = None,
    peak_min_height: float | None = None,
    prominence: float | None = None,
    peak_prominence: float | None = None,
    gaussian_sigma: float = 0.0,
    minimum_instance_area: int = 0,
    connectivity: int = 8,
) -> np.ndarray:
    """Generate a labelled instance map from a foreground probability map."""
    probability = _as_probability_map(probability_map)
    if foreground_threshold is not None:
        threshold = foreground_threshold
    if peak_min_distance is not None:
        min_distance = peak_min_distance
    if min_peak_distance is not None:
        min_distance = min_peak_distance
    if peak_min_height is not None:
        min_height = peak_min_height
    if peak_prominence is not None:
        prominence = peak_prominence
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")

    binary_mask = probability > threshold
    if not binary_mask.any():
        return np.zeros(probability.shape, dtype=np.int32)

    distance = distance_transform_edt(binary_mask)
    markers, smoothed_distance = extract_watershed_markers(
        distance,
        mask=binary_mask,
        min_distance=min_distance,
        min_height=min_height,
        prominence=prominence,
        gaussian_sigma=gaussian_sigma,
        connectivity=connectivity,
        minimum_instance_area=minimum_instance_area,
    )
    if markers.max() == 0:
        return np.zeros(probability.shape, dtype=np.int32)

    labels = watershed(
        -smoothed_distance,
        markers,
        mask=binary_mask,
        connectivity=_connectivity_footprint(connectivity),
    )

    if minimum_instance_area > 0:
        for label_id in np.unique(labels):
            if label_id == 0:
                continue
            if np.count_nonzero(labels == label_id) < minimum_instance_area:
                labels[labels == label_id] = 0
        labels = _relabel_instance_map(labels)
    return labels.astype(np.int32, copy=False)


def _relabel_instance_map(labels: np.ndarray) -> np.ndarray:
    result = np.zeros(labels.shape, dtype=np.int32)
    next_label = 1
    for label_id in np.unique(labels):
        if label_id > 0:
            result[labels == label_id] = next_label
            next_label += 1
    return result


def reconstruct_instances(
    foreground_probability: torch.Tensor | np.ndarray,
    **kwargs: object,
) -> np.ndarray:
    """Public alias for distance-transform watershed reconstruction."""
    return distance_transform_watershed(foreground_probability, **kwargs)


extract_distance_transform_markers = extract_watershed_markers
reconstruct_distance_transform_watershed = reconstruct_instances


class UNet_DistanceTransform_Watershed(SemanticSegmentationUNet):
    """Conventional U-Net with configurable distance-transform watershed post-processing."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        *,
        foreground_threshold: float = 0.5,
        peak_min_distance: int = 5,
        peak_min_height: float | None = None,
        peak_prominence: float | None = None,
        gaussian_sigma: float = 0.0,
        minimum_instance_area: int = 0,
        connectivity: int = 8,
        **aliases: object,
    ) -> None:
        super().__init__(in_channels=in_channels, channels=channels)
        threshold = aliases.pop("threshold", foreground_threshold)
        peak_min_distance = aliases.pop(
            "min_distance",
            aliases.pop("min_peak_distance", peak_min_distance),
        )
        peak_min_height = aliases.pop(
            "min_height", aliases.pop("min_peak_height", peak_min_height)
        )
        peak_prominence = aliases.pop(
            "prominence", aliases.pop("peak_prominence", peak_prominence)
        )
        gaussian_sigma = aliases.pop("sigma", gaussian_sigma)
        minimum_instance_area = aliases.pop(
            "minimum_area", aliases.pop("min_instance_area", minimum_instance_area)
        )
        if aliases:
            raise TypeError(f"Unknown watershed parameters: {sorted(aliases)}")
        if not 0 <= float(threshold) <= 1:
            raise ValueError("foreground_threshold must be in [0, 1].")
        if gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative.")
        if connectivity not in {4, 8}:
            raise ValueError("connectivity must be either 4 or 8.")
        self.foreground_threshold = float(threshold)
        self.peak_min_distance = int(peak_min_distance)
        self.peak_min_height = peak_min_height
        self.peak_prominence = peak_prominence
        self.gaussian_sigma = float(gaussian_sigma)
        self.minimum_instance_area = int(minimum_instance_area)
        self.connectivity = int(connectivity)

    def reconstruct_instances(
        self, foreground_probability: torch.Tensor | np.ndarray
    ) -> np.ndarray:
        """Apply the five-stage distance-transform watershed pipeline."""
        if isinstance(foreground_probability, torch.Tensor):
            probability = foreground_probability.detach().sigmoid()
        else:
            probability = np.asarray(foreground_probability)
        if isinstance(probability, torch.Tensor):
            probability = probability.detach().cpu().numpy()
        else:
            probability = np.asarray(probability)
        if probability.ndim == 4:
            if probability.shape[1] == 1:
                probability = probability[:, 0]
            else:
                raise ValueError("Expected one foreground channel.")
        if probability.ndim == 3:
            instance_maps = [
                distance_transform_watershed(
                    sample,
                    threshold=self.foreground_threshold,
                    min_distance=self.peak_min_distance,
                    min_height=self.peak_min_height,
                    prominence=self.peak_prominence,
                    gaussian_sigma=self.gaussian_sigma,
                    minimum_instance_area=self.minimum_instance_area,
                    connectivity=self.connectivity,
                )
                for sample in probability
            ]
            if len(instance_maps) == 1:
                return instance_maps[0]
            return np.stack(
                instance_maps
            )
        return distance_transform_watershed(
            probability,
            threshold=self.foreground_threshold,
            min_distance=self.peak_min_distance,
            min_height=self.peak_min_height,
            prominence=self.peak_prominence,
            gaussian_sigma=self.gaussian_sigma,
            minimum_instance_area=self.minimum_instance_area,
            connectivity=self.connectivity,
        )


DistanceTransformWatershedUNet = UNet_DistanceTransform_Watershed
