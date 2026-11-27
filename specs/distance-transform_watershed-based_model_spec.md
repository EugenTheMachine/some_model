# Model Specification 1: Classical U-Net with Distance-Transform Watershed

## 1. Overview

**Model name:** `UNet_DistanceTransform_Watershed`

**Architecture:** Classical U-Net for semantic foreground segmentation, followed by distance-transform-based marker extraction and watershed instance reconstruction.

**Core concept:** The neural network predicts a binary foreground probability map. The distance transform is subsequently calculated from the predicted binary mask using a conventional image-processing algorithm. Local maxima of the distance transform are used as markers for seeded watershed segmentation.

The model does not predict instance identities, feature embeddings, affinities, or distance maps directly.

## 2. Neural Network Architecture

### 2.1. Backbone

Use a conventional U-Net architecture consisting of:

- Contracting encoder with convolutional blocks and downsampling.
- Bottleneck.
- Expanding decoder with upsampling.
- Skip connections between corresponding encoder and decoder stages.
- Final convolutional prediction head.

The backbone should remain identical across both watershed experiments to ensure a fair comparison.

No attention mechanisms, transformers, additional feature aggregation modules, or architectural modifications are required.

### 2.2. Output Head

The network produces one output channel:

| Channel | Description | Activation |
|---|---|---|
| 1 | Foreground probability | Sigmoid |

Output tensor:

`[B, 1, H, W]`

Each pixel represents the probability of belonging to a cell rather than the background.

## 3. Training

### 3.1. Ground Truth

The original instance-indexed segmentation masks are converted into binary foreground masks:

- 0: Background.
- 1: Any cell instance.

Individual instance identities are not required for training this model.

### 3.2. Loss Function

Use a combination of Binary Cross-Entropy and Dice loss:

`L_total = L_BCE + L_Dice`

The loss is calculated between the predicted foreground probability map and the binary ground-truth mask.

### 3.3. Optimization

Use the same optimizer, learning-rate schedule, batch size, augmentation pipeline, and training configuration as the other U-Net experiments.

## 4. Inference and Instance Reconstruction

The inference pipeline consists of five stages.

### Stage 1: Foreground Prediction

Pass the input image through U-Net and obtain the foreground probability map.

### Stage 2: Binary Mask Generation

Apply a configurable threshold to obtain a binary foreground mask:

`binary_mask = foreground_probability > threshold`

Initial threshold: 0.5.

### Stage 3: Distance Transform

Calculate the Euclidean distance transform of the binary foreground mask.

For every foreground pixel, calculate the distance to the nearest background pixel:

`D(x, y) = min distance((x, y), background)`

Use `scipy.ndimage.distance_transform_edt`.

The resulting distance map represents the approximate interior geometry of the predicted objects.

Pixels located near the center of an object generally have higher distance values, while pixels near its boundaries have lower values.

### Stage 4: Marker Extraction

Identify local maxima in the distance map.

These maxima represent candidate cell centers and are used as watershed markers.

Recommended implementation:

- `skimage.feature.peak_local_max`
- Alternatively, `skimage.morphology.h_maxima` followed by connected-component labeling.

Configurable parameters:

- Minimum distance between peaks.
- Minimum peak prominence or height.
- Minimum object size.
- Optional Gaussian smoothing of the distance map.

The resulting markers must receive unique integer labels.

### Stage 5: Watershed Reconstruction

Apply seeded watershed using the negative distance transform as the topographical surface:

`labels = watershed(-distance_map, markers, mask=binary_mask)`

The negative distance transform is used because the centers of objects should represent the lowest points of the watershed landscape.

The binary foreground mask constrains the flooding procedure to predicted cell regions.

The resulting label map contains:

- 0: Background.
- 1, 2, 3, ...: Individual cell instances.

## 5. Required Dependencies

- PyTorch
- NumPy
- SciPy (`scipy.ndimage.distance_transform_edt`)
- scikit-image (`peak_local_max`, `h_maxima`, `watershed`)

## 6. Important Hyperparameters

| Parameter | Initial value |
|---|---:|
| Foreground threshold | 0.5 |
| Peak minimum distance | 5 px |
| Peak minimum height | Validation-dependent |
| Peak prominence | Validation-dependent |
| Gaussian smoothing sigma | 0–1 px |
| Minimum instance area | Validation-dependent |
| Connectivity | 8-connected |

All post-processing parameters must be optimized on the validation set, not the test set.

## 7. Advantages

- Very simple neural architecture.
- Only one prediction channel.
- No additional distance-map training objectives.
- Low computational overhead.
- No clustering or learned assignment mechanism.
- Easy to implement and debug.

## 8. Limitations

- Distance information is derived from the predicted binary mask, so segmentation errors directly affect instance reconstruction.
- Touching cells may produce insufficiently distinct distance maxima.
- Incorrect foreground connections can cause merging.
- Multiple local maxima inside a single cell can cause over-segmentation.
- Peak detection parameters may be sensitive to cell morphology.

## 9. Expected Experimental Role

This model should serve as the simplest watershed-based baseline.

Its primary purpose is to establish how effectively a conventional U-Net foreground prediction, combined with classical distance-transform processing, can separate individual cells without additional neural-network outputs.