# Dataset Specification

## Purpose

This document specifies the data loading and preprocessing contract for the current InstanSeg implementation. It is sufficient to reconstruct the dataset reader, dataset object, augmentation pipeline, collate behavior, and `__getitem__()` output without reverse engineering the repository.

The implementation supports **cell instance segmentation only**. It does not support separate nuclei/cell heads or the former `cells_and_nuclei` / `target_segmentation` modes.

Supported source modalities are:

- `Brightfield`
- `phase-contrast`

Fluorescence-like image/mask pairs must be rejected.

## 1. Input Data Formats

### 1.1 Directory layout

The primary training format is a directory with the following layout:

```text
<data_dir>/
  train/
    images/
      image_000.tif
      image_001.tif
      ...
    masks/
      mask_000.png
      mask_001.png
      ...
  val/                    # optional
    images/
    masks/
  test/                   # optional
    images/
    masks/
```

The `labels/` directory name is accepted as an alias for `masks/`.

The reader may also be given `<data_dir>/train/...`, `<data_dir>/val/...`, or `<data_dir>/test/...`. When there is no split structure, it falls back to:

```text
<data_dir>/
  images/
  masks/
```

If `val` or `test` is absent, it is created from the training split. The default split values are:

- validation ratio: `0.2`
- test ratio: `0.2`
- default seed: `42`

### 1.2 Image files

Supported extensions are `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, and `.bmp`.

Images may be:

- 2D grayscale: shape `(H, W)`
- HWC RGB: shape `(H, W, 3)`
- CHW: shape `(C, H, W)`
- TIFFs with one of those layouts
- single-channel grayscale, for duplication to `dim_in` channels
- multiple channels whose channel count matches `dim_in`, unless channel-invariant mode is enabled

TIFF files are read with `tifffile.imread`; PNG/JPEG/BMP files are read with `skimage.io.imread`.

When channel-invariant mode is not enabled, the post-read image must eventually have exactly `dim_in` channels. The normal `dim_in` value is 3.

### 1.3 Mask files

Masks are 2D integer instance maps. Their spatial dimensions must match the source image spatial dimensions.

```text
0       = background
1..N    = distinct cell instance labels
-1      = pixels to ignore, if present
```

Mask values must be renumbered to contiguous instance IDs after conversion to a tensor. Instance IDs are not semantic class IDs.

Masks are read:

- from PNGs with OpenCV `IMREAD_UNCHANGED`, cast to `int32`
- from all other formats with `skimage.io.imread(...).astype(int32)`

### 1.4 Image-mask pairing

Pairing is resolved in this order:

1. exact image filename matches a mask filename
2. exact stem matches a mask stem
3. `mask_<stem>` in PNG format
4. positional fallback only when image and mask counts are equal

Use a sorted filename list for both directories. Do not pair pairs based on arbitrary directories or insertion order.

### 1.5 Metadata

Every image/mask pair has this metadata dictionary:

```python
{
    "parent_dataset": "<dataset-directory-name>",
    "modality": "Brightfield" | "phase-contrast",
    "image_modality": "Brightfield" | "phase-contrast",
    "pixel_size": None | float,
    "name": "<image filename>",
}
```

For `.pth` datasets, preserve the item metadata but remove `image`, `cell_masks`, `nucleus_masks`, and `class_masks`; normalize `modality` to `image_modality` if only `modality` exists.

For directory datasets, `pixel_size` is `None`.

## 2. Split and Filtering Rules

### 2.1 Directory reader

The directory reader must:

1. discover `train`, `val`, and `test` directories
2. read valid, paired image/mask files
3. reject unsupported modalities
4. keep at most `train_data_ratio` fraction of the training list
5. create the validation split from training data if no validation directory exists
6. use the validation list as test data if no test directory exists

`train_data_ratio` is in `(0, 1]`. Validation and test ratios are in `[0, 1]`.

### 2.2 Validation fallback

If no validation directory exists:

- if training size `n <= 1`, leave no validation split
- otherwise calculate `num_val = max(1, round(n * val_ratio))`
- use `numpy RandomState(seed).permutation(n)` to choose the validation indices
- remove those indices from training

For reproducibility, perform this permutation independently and exactly with `RandomState(seed)`, not `np.random.default_rng(seed)`.

The current implementation must additionally guard `n == 1` so it does not create a zero-sized training/validation split accidentally.

### 2.3 `.pth` dataset reader

The `.pth` reader, if used, must:

- set `INSTANSEG_DATASET_PATH` to the default dataset path when unset
- load a file named `<dataset>_dataset.pth`, or the exact path when the dataset argument contains `.pth`
- attempt `torch.load(..., weights_only=False)` first and fall back to `torch.load(...)`
- keep items only if they contain `cell_masks`
- reject items marked duplicate
- filter by `parent_dataset` if `source_dataset != ["all"]`
- reject modalities other than Brightfield and phase-contrast
- resolve string image references relative to `INSTANSEG_DATASET_PATH`
- inflate a matching zip archive if a referenced file has not yet been inflated
- assert that each requested split is non-empty

## 3. CPU Preloading Requirement

**All retained source images and masks must be preloaded into CPU memory before creating the PyTorch `DataLoader`.**

The required flow is:

```text
disk files
  -> NumPy arrays / in-memory TIFF objects
  -> X = list of CPU images
  -> Y = list of CPU masks
  -> metadata = list of metadata dictionaries
  -> Segmentation_Dataset
  -> DataLoader
```

The dataset constructor must retain these lists:

```python
class Segmentation_Dataset:
    self.X = img
    self.Y = label
    self.metadata = metadata
```

Requirements:

1. `len(X) == len(Y) == len(metadata)`.
2. The full retained image and mask lists are loaded before training/evaluation.
3. `__getitem__(i)` performs augmentation in the worker or main process, but must read `X[i]` and `Y[i]` directly from memory.
4. It must not reopen image or mask files in `__getitem__()`.
5. `DataLoader` workers receive only indexing/augmentation work; disk loading is not in the hot loop.
6. If CUDA is available, enable `pin_memory=True`; otherwise use `pin_memory=False`.
7. Use `num_workers=0` for Windows/local runs unless explicitly configured otherwise.
8. Enable `persistent_workers=True` when `num_workers > 0`.
9. Set `prefetch_factor=2` when `num_workers > 0` and leave it unset when there are no workers.
10. Use a CPU-GPU pinned-memory transfer for the final batch: after CPU collation, `.to(device)` occurs in the training/evaluation loop.

This is an intentional performance tradeoff: all data must fit in CPU RAM, but random crops and augmentation become faster and GPU transfer avoids repeated disk I/O.

## 4. Dataset and Collate Implementation

The complete required dataset shape is:

```python
class Segmentation_Dataset:
    def __init__(
        self,
        img,
        label,
        common_transforms=True,
        metadata=None,
        size=(256, 256),
        augmentation_dict=None,
        dim_in=3,
        *,
        debug=False,
        channel_invariant=False,
        random_seed=None,
    ):
        self.X = img
        self.Y = label
        self.common_transforms = common_transforms

        assert len(self.X) == len(self.Y), (
            "The number of images and labels must be the same"
        )

        if not metadata:
            self.metadata = [None] * len(self.X)
        else:
            self.metadata = metadata

        assert len(self.X) == len(self.metadata), (
            "The number of images and metadata must be the same"
        )

        self.size = size
        self.Augmenter = Augmentations(
            augmentation_dict=augmentation_dict,
            debug=debug,
            shape=self.size,
            dim_in=dim_in,
            channel_invariant=channel_invariant,
            random_seed=random_seed,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]
        label = self.Y[i]
        meta = self.metadata[i]

        if self.common_transforms:
            data, label = self.Augmenter(data, label, meta)

        if len(label.shape) == 2:
            label = label[None, :]
        if len(data.shape) == 2:
            data = data[None, :]

        assert not data.isnan().any(), "Transformed image contains NaN"
        assert not label.isnan().any(), "Transformed label contains NaN"

        return data.float(), label
```

The collate function must accept a batch of `(image, label)` tuples and return:

```python
images, labels, lengths
```

where:

- `images`: `(batch, dim_in, height, width)`
- `labels`: `(batch, 1, height, width)`
- `lengths`: `(batch,)` channel counts, as `int32`

Padding shape is determined by the first item:

```python
imgs, labels = zip(*data)
lengths = [img.shape[0] for img in imgs]
max_len = max(lengths)
C, H, W = data[0][0].shape
images = torch.zeros((len(data), max_len, H, W), dtype=torch.float32)
labels = torch.stack(labels)
lengths = torch.tensor(lengths)

for i, img in enumerate(imgs):
    images[i, :len(img)] = img

return images, labels, lengths.int()
```

`C` is currently unused, but must remain `3` in normal Brightfield/phase-contrast configuration.

## 5. Augmentation Pipeline

### 5.1 Modality dispatch

`Augmentations.__call__` selects modality as follows:

1. use explicit `self.modality` if configured
2. otherwise use `meta["image_modality"]` or `meta["modality"]`
3. otherwise estimate the modality from image and labels
4. reject any modality outside:
   - `Brightfield`
   - `phase-contrast`

For metadata that identifies Brightfield but provides a 3-channel input, Brightfield is retained. Brightfield metadata with a non-3-channel image is normalized to phase-contrast.

The augmentation dictionary is selected by the observed modality.

### 5.2 Ordered augmentation dictionary

Use `collections.OrderedDict` and evaluate entries in this order.

```python
import collections

def get_augmentation_dict(
    dim_in,
    amount,
    *,
    pixel_size=0.5,
    augmentation_type="minimal",
    mean_diameter=None,
):
    if mean_diameter is not None:
        mean_diameter_heavy = (mean_diameter, 2**(-2), 2**(2))
        mean_diameter_light = (mean_diameter, 2**(-1), 2**(1))
    else:
        mean_diameter_heavy = None
        mean_diameter_light = None

    if augmentation_type == "minimal":
        train_bf = collections.OrderedDict([
            ("to_tensor", [1]),
            ("normalize", [1]),
            ("torch_rescale", [1, pixel_size, mean_diameter_light]),
            ("flips", [1]),
            ("rotate", [1]),
        ])
        test_bf = collections.OrderedDict([
            ("to_tensor", [1]),
            ("normalize", [1]),
            ("torch_rescale", [1, pixel_size, mean_diameter_light]),
            ("flips", [1]),
        ])

    elif augmentation_type == "kornia_intensity":
        train_bf = collections.OrderedDict([
            ("to_tensor", [1]),
            ("normalize", [1]),
            ("torch_rescale", [1, pixel_size, mean_diameter_light]),
            ("flips", [1]),
            ("rotate", [1]),
            ("kornia_base_augmentations", [1]),
        ])
        test_bf = collections.OrderedDict([
            ("to_tensor", [1]),
            ("normalize", [1]),
            ("torch_rescale", [1, pixel_size, mean_diameter_light]),
            ("flips", [1]),
        ])

    elif augmentation_type == "heavy":
        train_bf = collections.OrderedDict([
            ("to_tensor", [1]),
            ("normalize_HE_stains", [0.1, amount * 0]),
            ("extract_hematoxylin_stain", [0.1, amount * 0]),
            ("normalize", [1]),
            ("torch_rescale", [1, pixel_size, mean_diameter_heavy]),
            ("randomJPEGcompression", [0.2, amount]),
            ("adjust_hue", [0.2, amount]),
            ("AdjustContrast", [0.2, amount]),
            ("flips", [1]),
            ("rotate", [1]),
            ("brightness_augment", [0.2, amount]),
            ("RandGaussianNoise", [0.2, amount]),
            ("perspective", [0, amount]),
        ])
        train_pc = collections.OrderedDict([
            ("to_tensor", [1]),
            ("normalize", [1]),
            ("torch_rescale", [1, pixel_size, mean_diameter_heavy]),
            ("flips", [1]),
            ("rotate", [1]),
            ("kornia_base_augmentations", [1]),
        ])
        test_bf = collections.OrderedDict([
            ("to_tensor", [1]),
            ("normalize", [1]),
            ("torch_rescale", [1, pixel_size, mean_diameter_light]),
            ("flips", [1]),
        ])
        test_pc = collections.OrderedDict([
            ("to_tensor", [1]),
            ("normalize", [1]),
            ("torch_rescale", [1, pixel_size, mean_diameter_light]),
            ("flips", [1]),
        ])

    elif augmentation_type == "brightfield_only":
        train_bf = collections.OrderedDict([
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
        ])
        test_bf = collections.OrderedDict([
            ("to_tensor", [1]),
            ("normalize", [1]),
            ("torch_rescale", [1, pixel_size, 0]),
            ("flips", [1]),
        ])

    elif augmentation_type in {"two_channel", "colourize"}:
        raise ValueError(
            f"Augmentation type {augmentation_type!r} is unavailable"
        )
    else:
        raise ValueError(
            "Invalid augmentation type. Supported options are 'minimal', "
            "'kornia_intensity', 'heavy', and 'brightfield_only'."
        )

    if augmentation_type in {"minimal", "kornia_intensity"}:
        train_pc = train_bf
        test_pc = test_bf
    if augmentation_type == "brightfield_only":
        train_pc = train_bf
        test_pc = test_bf

    return {
        "train": {"Brightfield": train_bf, "phase-contrast": train_pc},
        "test": {"Brightfield": test_bf, "phase-contrast": test_pc},
    }
```

The heavy Brightfield test path and the phase-contrast test path use the listed test dictionaries; no test-time perspective, hue, intensity, stain, or Kornia intensity augmentation is applied in the default paths.

### 5.3 Pipeline execution

For each modality-specific ordered entry:

```python
for augmentation, values in augmentation_dict.items():
    if np.random.random() < values[0]:
        if augmentation in [
            "normalize_HE_stains",
            "extract_hematoxylin_stain",
            "normalize",
        ]:
            if not has_been_normalized:
                amount = values[1] if len(values) > 1 else None
                image, labels = getattr(self, augmentation)(
                    image,
                    labels,
                    amount=amount,
                    metadata=metadata,
                )
                has_been_normalized = True
        elif augmentation == "torch_rescale":
            _, requested_pixel_size, diameter_range = values
            if not (
                isinstance(diameter_range, tuple)
                and len(diameter_range) == 3
            ):
                diameter_range = None
            image, labels = self.torch_rescale(
                image,
                labels,
                current_pixel_size=None,
                requested_pixel_size=requested_pixel_size,
                diameter_range=diameter_range,
                metadata=metadata,
            )
        else:
            p, *rest = values
            amount = rest[0] if len(rest) == 1 else None
            image, labels = getattr(self, augmentation)(
                image,
                labels,
                amount=amount,
                metadata=metadata,
            )
        assert not image.isnan().any()

image, labels = self.duplicate_grayscale_channels(
    image, labels, metadata=metadata
)

if image.var() > 1e2:
    image = torch.clip(image, min=-1, max=5)
```

Normalization state is global to a sample: once `normalize`, `normalize_HE_stains`, or `extract_hematoxylin_stain` has run, later normalization entries are skipped. The heavy pipeline lists stain operations before standard normalization.

### 5.4 Channel-axis conversion

`_move_channel_axis` must move an image's smallest-dimension channel axis to channel-first or channel-back.

```python
def _move_channel_axis(img, to_back=False):
    img = img.squeeze()
    if img.ndim != 3:
        if img.ndim == 2:
            if isinstance(img, np.ndarray):
                img = img[None, :]
            else:
                img = img[None, :]
        elif img.ndim != 3:
            raise ValueError("Input array should be 2D or 3D")

    ch = np.argmin(img.shape)
    if isinstance(img, np.ndarray):
        if to_back:
            return np.rollaxis(img, ch, 3)
        return np.rollaxis(img, ch, 0)
    if to_back:
        return img.movedim(ch, -1)
    return img.movedim(ch, 0)
```

The implementation assumes the smallest dimension is the channel dimension. Ambiguous cubic data should be explicitly CHW/HWC-normalized before entering the pipeline.

### 5.5 Complete augmentation method implementations

#### Tensor conversion

```python
def to_tensor(self, image, labels=None, normalize=False, amount=None, metadata=None):
    if isinstance(image, np.ndarray):
        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float32)
        if normalize:
            image, _ = self.normalize(image)
        out = torch.tensor(_move_channel_axis(image), dtype=torch.float32)
    elif isinstance(image, torch.Tensor):
        out = _move_channel_axis(image).float()
        if normalize:
            out, _ = self.normalize(out)
    else:
        raise TypeError("Image must be NumPy or PyTorch")

    if labels is not None:
        if isinstance(labels, np.ndarray):
            labels = np.atleast_3d(labels)
            labels = _move_channel_axis(labels)
            for n, lab in enumerate(labels):
                if not (lab == -1).any():
                    labels[n] = fastremap.renumber(lab)[0]
            labels = torch.tensor(labels.astype(np.int32), dtype=torch.int32)
        labels = torch.atleast_3d(labels)
        labels = _move_channel_axis(labels)

    out = out.squeeze()
    out = _move_channel_axis(torch.atleast_3d(out))
    return out, labels
```

#### Percentile normalization

```python
def percentile_normalize(
    img,
    percentile=0.1,
    subsampling_factor=1,
    epsilon=1e-3,
):
    img = torch.atleast_3d(img)
    channel_axis = np.argmin(img.shape)
    img = _move_channel_axis(img, to_back=True)

    for c in range(img.shape[-1]):
        im_temp = img[::subsampling_factor, ::subsampling_factor, c]
        if img.is_cuda or img.is_mps:
            q = torch.tensor(
                [percentile / 100, (100 - percentile) / 100],
                device=im_temp.device,
            )
            p_min, p_max = torch.quantile(im_temp, q)
        else:
            p_min, p_max = np.percentile(
                im_temp.cpu(),
                [percentile, 100 - percentile],
            )
        img[:, :, c] = (
            img[:, :, c] - p_min
        ) / max(epsilon, p_max - p_min)

    return img.movedim(2, channel_axis)

def normalize(
    self,
    image,
    labels=None,
    amount=0.0,
    subsampling_factor=1,
    percentile=0.1,
    metadata=None,
):
    return percentile_normalize(
        image,
        subsampling_factor=subsampling_factor,
        percentile=percentile,
    ), labels
```

#### Flips

```python
def flips(self, image, labels, amount=0.5, metadata=None):
    if random.random() > 0.5:
        image = TF.hflip(image)
        labels = TF.hflip(labels)
    if random.random() > 0.5:
        image = TF.vflip(image)
        labels = TF.vflip(labels)
    return image, labels
```

The pipeline passes `amount=[1]`, so the effective flip probability is hardcoded to `0.5` per axis.

#### 90-degree rotations

```python
def rotate(self, image, labels, amount=0, metadata=None):
    angle = int(np.random.choice([180, 90, 270, 0]))
    return TF.rotate(image, angle), TF.rotate(labels, angle)
```

This produces one of exactly 0, 90, 180, or 270 degrees.

#### Rescale and random crop

```python
def torch_rescale(
    self,
    image,
    labels=None,
    amount=0,
    current_pixel_size=None,
    requested_pixel_size=None,
    crop=True,
    random_seed=None,
    metadata=None,
    diameter_range=None,
    modality=None,
):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    assert labels is None or image.shape[-2:] == labels.shape[-2:]

    shape = (
        torch.tensor(image.shape[-2:])
        * (1 + ((torch.rand(1) - 0.5) * amount))
    ).int().tolist()

    if metadata is not None:
        if current_pixel_size is None and "pixel_size" in metadata:
            current_pixel_size = metadata["pixel_size"]
        if modality is None and "image_modality" in metadata:
            modality = metadata["image_modality"]

    if current_pixel_size is not None and requested_pixel_size is not None:
        scale = current_pixel_size / requested_pixel_size
        shape = (
            torch.tensor(image.shape[-2:]) * scale
        ).int().tolist()

    if diameter_range is not None and labels is not None:
        scale = resize_with_log_scale(
            labels,
            mean_diameter=diameter_range[0],
            min_scale=diameter_range[1],
            max_scale=diameter_range[2],
        )
        if scale == 0:
            scale = 1
        shape = (
            torch.tensor(image.shape[-2:]) * scale
        ).int().tolist()

    resized_data = Resize(size=shape, antialias=True)(image)
    if labels is not None:
        resized_labels = Resize(
            size=shape,
            interpolation=InterpolationMode.NEAREST,
        )(labels)

    while (
        self.shape is not None
        and np.any(np.array(resized_data.shape[-2:]) < self.shape)
        and crop
    ):
        pad = int((self.shape[0] - min(resized_data.shape[-2:])) / 2) + 3
        pad = torch.tensor(
            [pad, resized_data.shape[-2], resized_data.shape[-1]]
        ).min().int().item() - 1
        fill = (
            image.max()
            if modality == "Brightfield"
            else resized_data.min()
        )
        resized_data = torch.nn.functional.pad(
            resized_data, (pad, pad, pad, pad),
            mode="constant", value=fill,
        )
        if labels is not None:
            resized_labels = torch.nn.functional.pad(
                resized_labels,
                (pad, pad, pad, pad),
                mode="constant",
                value=min(abs(resized_labels.min()), 0),
            ).to(labels.dtype)

    if not crop:
        return resized_data, resized_labels if labels is not None else None

    cropper = RandomCrop(size=self.shape)
    i, j, h, w = cropper.get_params(resized_data, output_size=self.shape)
    out_image = resized_data[:, i:i+h, j:j+w].float()
    if labels is None:
        return out_image, None
    out_labels = resized_labels[:, i:i+h, j:j+w]
    assert out_image.shape[-2:] == self.shape
    assert out_labels.shape[-2:] == self.shape
    return out_image, out_labels
```

The current implementation's scale formula uses image median diameter when `diameter_range` is present. Otherwise, requested pixel size takes precedence. With no pixel-size information, size remains the source dimensions unless augmentation amount contributes an independent resize.

Log-scale diameter rescaling is:

```python
def resize_with_log_scale(
    lab,
    mean_diameter=30,
    min_scale=0.25,
    max_scale=4,
):
    original_diameter = 2 * np.sqrt(
        measure_average_instance_area(lab) / np.pi
    )
    scale_factor = 2 ** random.uniform(
        np.log2(min_scale),
        np.log2(max_scale),
    )
    target_diameter = mean_diameter * scale_factor
    return target_diameter / original_diameter
```

#### Random JPEG

Only RGB images (`C == 3`) are altered. The quality factor is:

```python
qf = random.randrange(
    int(100 * (1 - amount) - 1),
    100,
)
```

Convert the image to PIL, JPEG-compress to memory, and convert back with `ToTensor`.

#### Brightness

```python
rand = 1 + ((torch.rand(len(image)) - 0.5) * amount)
out = (image.permute(1, 2, 0) * rand).permute(2, 0, 1)
```

One random factor is generated per channel.

#### Gaussian noise

Use MONAI `RandGaussianNoise(prob=1, mean=0.0, std=amount / 5)`.

#### Histogram intensity shift

Use MONAI `RandStdShiftIntensity(10)` independently over each channel.

#### Contrast

Use MONAI `AdjustContrast(gamma=amount)`.

#### Hue

For exactly 3-channel Brightfield input only:

```python
hue_factor = (torch.rand(1) - 0.5) * amount * 0.5
out = TF.adjust_hue(image, hue_factor=hue_factor)
```

Replace NaNs with zero.

#### Perspective

Apply a torchvision `RandomPerspective(distortion_scale=amount/2, p=1.0, interpolation=NEAREST)` to labels with the saved RNG state, and apply the same perspective transform to image data. Set label fill to zero. The image fill value is:

- `image.max()` for Brightfield
- `image.min()` for phase-contrast

The label transform must use the exact RNG state captured immediately after the image transform.

#### Gradient

Generate an XY grid in `[0, 0.5]`; randomly flip it horizontally and/or vertically for each of X and Y; then add:

```python
out = (
    amount
    * (np.random.random() * x + np.random.random() * y)
    * image.max()
) + image
```

#### Inversion

```python
out = 1 - image
```

#### Macenko stain operations

These run only for Brightfield metadata. Heavily augmented stain operations must use:

```text
tensor = (image / (image.max() + 0.001)) * 255
tensor = clamp(tensor, 0, 255)
normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
HERef += (torch.rand_like(HERef) - 0.5) * HERef * amount
maxCRef += (torch.rand_like(maxCRef) - 0.5) * maxCRef * amount
```

In the current configuration, both stain amounts are `amount * 0`, so both are effectively no-op for their reference matrices, unless the configured amount logic is changed. If `normalize_HE_stains` succeeds, return `norm / 255`. If `extract_hematoxylin_stain` succeeds, return `H / 255`. On any exception, return the input unchanged.

#### Kornia intensity augmentation

For phase-contrast/heavy and Kornia-intensity pipelines, normalize image minimum/maximum to `[0, 1]`, run this sequential augmentation stack, and denormalize:

```python
torch.nn.Sequential(
    kornia.augmentation.RandomLinearIllumination(gain=(0.1, 0.4), p=1),
    kornia.augmentation.RandomPlasmaContrast(p=0.5),
    kornia.augmentation.RandomSaltAndPepperNoise(),
    kornia.augmentation.RandomMedianBlur(p=0.2),
    kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
    kornia.augmentation.RandomBoxBlur((3, 3), p=0.2),
    kornia.augmentation.RandomSharpness(sharpness=1, p=0.5),
    kornia.augmentation.RandomContrast(contrast=(1, 1.5), p=0.5),
    kornia.augmentation.RandomGamma(
        gamma=(0.5, 0.5),
        gain=(0.5, 1.5),
        p=0.5,
    ),
)(image).squeeze(0)
```

Labels are unchanged.

#### Grayscale channel duplication

```python
def duplicate_grayscale_channels(self, image, labels, metadata=None):
    if image.shape[0] == 1 and self.dim_in != 1 and not self.channel_invariant:
        return image.repeat(self.dim_in, 1, 1), labels
    if image.shape[0] != self.dim_in and not self.channel_invariant:
        raise ValueError(
            f"Image has {image.shape[0]} channels, "
            f"but model expects {self.dim_in}"
        )
    return image, labels
```

This finalizes the channel count after every earlier transform.

## 6. Reproducibility and Randomness

The global seed is `preprocessing.seed`, default `42`.

At training start, seed every source:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Additional required seed behavior:

- dataset validation permutation: `numpy.RandomState(seed)`
- each dataset object's augmenter:
  - `torch.manual_seed(random_seed)`
  - `np.random.seed(random_seed)`
  - `random.seed(random_seed)` must also be fixed for complete reproducibility
- `DataLoader` worker initialization:

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
```

- the loader's `generator` should be manually seeded with the same seed
- `RandomSampler` should use a seeded loader generator
- every source of randomness in augmentations must be seeded:
  - NumPy array/tensor draws
  - Python `random` draws
  - PyTorch CPU/CUDA draws
  - Kornia/MONAI transforms
  - torchvision random crops and transforms
- train/validation split and train/evaluation `RandomSampler` randomness must derive from the fixed seed
- model initialization must also receive the same seed
- loss-time instance sampling (`torch.randperm`) must use the seeded PyTorch RNG

For exact repeatability across workers, prefer an explicitly seeded per-worker or per-sample generator over relying on global RNG state.

Without fixing `random.seed(...)`, `random.random()` and Python `random.randrange()` draws remain outside NumPy/PyTorch control and can diverge between repeated runs.

## 7. `__getitem__()` / Preprocessed Output Shape

In the final implementation, `__getitem__(i)` returns:

```python
(
    image: torch.float32,
    label: torch.int32,
)
```

### 7.1 Image

The image shape is:

```text
dim_in x tile_size x tile_size
```

Normally:

```text
3 x H x W
```

where `H == W == imgsz` (`512` by default). The final shape is guaranteed by the random crop/pad/rescale pipeline.

Examples:

```text
3 x 256 x 256
3 x 512 x 512
1 x 512 x 512       channel-invariant/single-channel model
C x 512 x 512       if C is the configured dim_in
```

The image dtype is `torch.float32`. Pixels are generally percentile-normalized to approximately `[0, 1]`, but no hard `[0, 1]` bound is guaranteed after augmentation.

### 7.2 Label

The label shape is:

```text
1 x tile_size x tile_size
```

for example:

```text
1 x 512 x 512
```

The label dtype is `torch.int32`. It contains:

- `0`: background
- positive: instance IDs
- optionally `-1`: ignored pixels

After NumPy conversion, labels are renumbered with `fastremap.renumber` unless at least one `-1` pixel forces the ignore convention.

### 7.3 Batch shape

After collating one batch of size `B`:

```text
images: (B, dim_in, tile_size, tile_size)
labels: (B, 1, tile_size, tile_size)
lengths: (B,)
```

For normal use:

```text
images: (B, 3, H, W)
labels: (B, 1, H, W)
lengths: (B,)
```

The model receives `images`; labels are sent to the loss and evaluation functions.

## 8. Loader Configuration

The training and test loaders should be built as:

```python
train_sampler = RandomSampler(
    train_data,
    num_samples=args.length_of_epoch,
) if args.length_of_epoch is not None else RandomSampler(train_data)

test_sampler = RandomSampler(
    test_data,
    num_samples=max(1, int(args.length_of_epoch * 0.2)),
) if args.length_of_epoch is not None else RandomSampler(test_data)

train_loader = DataLoader(
    train_data,
    collate_fn=collate_fn,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    sampler=train_sampler,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(args.num_workers > 0),
    prefetch_factor=2 if args.num_workers > 0 else None,
    worker_init_fn=seed_worker,
)
```

The default test/validation loader is randomly sampled too. It is therefore not a deterministic positional iteration unless the random sampler and seed are fixed; even with augmentation, deterministic test behavior requires stable seeds.
