# Model Specification 2: Classical U-Net with Learned-Distance Watershed

## 1. Overview

**Model name:** `UNet_LearnedDistance_Watershed`

**Architecture:** Classical U-Net with a three-channel prediction head, followed by marker extraction and seeded watershed reconstruction.

**Core concept:** Instead of calculating all geometric information from the predicted binary mask, the neural network learns to predict three complementary representations:

1. Foreground probability.
2. Distance to the instance center.
3. Distance to the instance boundary.

The predicted distance representations are subsequently used to identify reliable watershed markers and reconstruct individual cell instances.

This approach is inspired by the Automatic Instance Segmentation (AIS) procedure described in *Segment Anything for Microscopy*.

## 2. Neural Network Architecture

### 2.1. Backbone

Use exactly the same classical U-Net backbone as in Specification 1:

- Contracting encoder.
- Bottleneck.
- Expanding decoder.
- Skip connections.
- Final convolutional prediction head.

The backbone must remain unchanged to isolate the effect of learned geometric representations.

### 2.2. Output Head

The network produces three output channels:

| Channel | Description | Activation |
|---|---|---|
| 1 | Foreground probability | Sigmoid |
| 2 | Center-distance representation | Sigmoid |
| 3 | Boundary-distance representation | Sigmoid |

Output tensor:

`[B, 3, H, W]`

The distance channels are normalized to the range [0, 1].

### 2.3. Output Channel Semantics

**Foreground probability**

Indicates whether a pixel belongs to a cell.

Higher values indicate a higher probability of foreground membership.

**Center-distance representation**

Measures the distance of a pixel from the center of its corresponding instance.

Lower values indicate proximity to the instance center.

Higher values indicate increasing distance from the center.

**Boundary-distance representation**

Measures the distance of a pixel from the boundary of its corresponding instance.

Lower values indicate proximity to the cell boundary.

Higher values indicate that the pixel is located deeper inside the cell.

These representations provide complementary information:

- Center distance helps identify candidate instance centers.
- Boundary distance helps distinguish interior regions from narrow inter-cell connections.

## 3. Ground-Truth Target Generation

The original instance-indexed masks are used to generate all three training targets.

Let `M_i` represent the binary mask of instance i.

### 3.1. Foreground Target

Merge all instance masks into a single binary foreground mask:

`FG = union(M_i)`

Background pixels have a value of 0; foreground pixels have a value of 1.

### 3.2. Center-Distance Target

For every instance, determine its center using the maximum of its Euclidean distance transform.

Use this point as the instance center.

For each pixel belonging to the instance, calculate its Euclidean distance to that center.

Normalize the distances within the individual instance:

`D_center(x, y) = distance_to_center / maximum_instance_distance`

Consequently:

- The center has a value close to 0.
- The outer regions have values closer to 1.

This produces a separate center-distance representation for every instance.

### 3.3. Boundary-Distance Target

For every instance, calculate its Euclidean distance transform:

`D_boundary(x, y) = distance_to_nearest_instance_boundary`

Normalize the distances within the individual instance:

`D_boundary_norm = D_boundary / maximum_instance_distance`

Consequently:

- Boundary pixels have values close to 0.
- Interior pixels have higher values.
- Deep interior regions approach 1.

### 3.4. Background Handling

Distance targets are supervised only inside foreground instances.

Background pixels are excluded from the distance-channel losses using a foreground mask.

The foreground prediction channel is supervised across the entire image.

## 4. Training

### 4.1. Loss Function

Use a multi-task loss consisting of three components:

`L_total = L_foreground + λ_center * L_center + λ_boundary * L_boundary`

Where:

- `L_foreground`: Binary Cross-Entropy + Dice loss.
- `L_center`: Regression loss for center-distance prediction.
- `L_boundary`: Regression loss for boundary-distance prediction.

Recommended initial implementation:

- Foreground: BCE + Dice.
- Center distance: Masked Smooth L1 loss.
- Boundary distance: Masked Smooth L1 loss.

The distance losses are calculated only for foreground pixels.

Initial loss weights:

| Component | Weight |
|---|---:|
| Foreground | 1.0 |
| Center distance | 1.0 |
| Boundary distance | 1.0 |

The weights should be adjusted based on validation performance and the relative magnitude of the individual losses.

### 4.2. Training Targets

Each training sample produces three target maps with identical spatial dimensions:

- Binary foreground map.
- Normalized center-distance map.
- Normalized boundary-distance map.

No instance-classification head, affinity head, or embedding head is required.

## 5. Inference and Instance Reconstruction

The inference pipeline consists of five stages.

### Stage 1: Neural Prediction

Pass the input image through U-Net and obtain:

- Foreground probability.
- Center-distance prediction.
- Boundary-distance prediction.

### Stage 2: Foreground Mask Generation

Threshold the foreground probability map:

`binary_mask = foreground_probability > threshold`

Initial threshold: 0.5.

### Stage 3: Marker Extraction

Use both predicted distance representations to identify reliable instance seeds.

A candidate seed region must satisfy two conditions:

1. Center-distance prediction is below a configurable threshold.
2. Boundary-distance prediction is above a configurable threshold.

Conceptually:

`seed_candidates = (center_distance < center_threshold) AND (boundary_distance > boundary_threshold)`

The candidate regions are restricted to the foreground mask.

Apply connected-component labeling to the seed candidates to assign unique marker IDs.

If necessary, apply small-region filtering to remove spurious markers.

### Stage 4: Watershed Surface Construction

Construct a topographical surface using the predicted distance representations.

The initial implementation should use the center-distance prediction as the primary watershed heightmap.

Since lower center-distance values represent proximity to an instance center, the predicted center-distance map naturally provides a landscape with candidate minima.

An optional combined heightmap can be investigated:

`H = α * center_distance + β * (1 - boundary_distance)`

Where α and β control the relative influence of the two representations.

The combined representation should be treated as an experimental variant rather than an obligatory component.

### Stage 5: Watershed Reconstruction

Apply seeded watershed using:

- Markers generated from Stage 3.
- Heightmap generated from Stage 4.
- Binary foreground mask as the watershed constraint.

Conceptually:

`labels = watershed(heightmap, markers, mask=binary_mask)`

The result is an integer instance-label map.

## 6. Required Dependencies

- PyTorch
- NumPy
- SciPy (`scipy.ndimage`)
- scikit-image (`skimage.segmentation.watershed`, connected-component utilities)

## 7. Important Hyperparameters

| Parameter | Initial value |
|---|---|
| Foreground threshold | 0.5 |
| Center-distance threshold | Validation-dependent |
| Boundary-distance threshold | Validation-dependent |
| Minimum marker area | Validation-dependent |
| Center heightmap weight | 1.0 |
| Boundary heightmap weight | 0.0 initially |
| Watershed connectivity | 8-connected |

The center and boundary thresholds must be tuned using the validation set.

## 8. Advantages

- Learns geometric information directly from the microscopy images.
- Does not depend exclusively on the geometry of the binary foreground mask.
- Uses two complementary distance representations.
- Can distinguish narrow connections between adjacent instances.
- Can reduce incorrect splitting of irregular or non-convex instances.
- Retains a lightweight classical U-Net backbone.
- Avoids embedding clustering and affinity graph optimization.

## 9. Limitations

- Requires additional training targets and loss functions.
- Incorrect distance predictions can generate unreliable markers.
- Threshold selection can affect instance reconstruction.
- The three output channels require more supervision than the binary baseline.
- Watershed remains a conventional non-differentiable post-processing operation.

## 10. Expected Experimental Role

This model should serve as the primary learned-geometric watershed architecture.

Its purpose is to investigate whether explicitly learning instance-center and instance-boundary representations improves instance separation compared with calculating distance information directly from the predicted foreground mask.

The model should be evaluated against Specification 1 using identical backbone configurations, data splits, augmentation policies, training budgets, and evaluation metrics.