# Cell Segmentation

This project implements a single-pass U-Net instance-segmentation model for
microscopy images. It predicts foreground, four directional affinities, and
per-pixel assignments to a fixed number of instance slots. Hungarian matching
is used only to construct training targets; inference uses a per-pixel slot
argmax without clustering or connected-component post-processing.

It also includes `affinity_guided_recurrent`, which adds affinity-gated
ConvGRU message passing and recurrent slot refinement before dense assignment.

The `dela_net` model adds normalized discriminative pixel embeddings and a
learned, image-conditioned slot assignment head. It uses Hungarian matching
only during training, with inference producing instance IDs through the
assignment argmax. Configure `num_slots` for the training set; training reports
the instance-count distribution and fails explicitly rather than truncating
images that exceed slot capacity.

It also provides a classical U-Net binary semantic-segmentation baseline.
Select `model_name="semantic_unet"` and configure `unet_channels` to train it
through the same data-loading and checkpointing facade. In this mode, all
nonzero ground-truth mask pixels are converted to foreground value 1.

The distance-transform watershed baseline is available as
`model_name="distance_transform_watershed"` (also
`UNet_DistanceTransform_Watershed`). It uses the classical U-Net for binary
foreground prediction, then extracts EDT local maxima and seeded watershed
instances. Configure `foreground_threshold`, `peak_min_distance`,
`peak_min_height`, `peak_prominence`, `gaussian_sigma`,
`minimum_instance_area`, and `connectivity`.
Validation and test metrics include instance-level precision, recall, and F1
(`object_precision`, `object_recall`, and `object_f1`), with a true positive
defined by a one-to-one instance match at IoU >= 0.5. The logs also report
`object_aji`, `object_pq`, and `object_boundary_f1`.

The learned-geometry watershed model is available as
`model_name="learned_distance_watershed"` (also
`UNet_LearnedDistance_Watershed`). It preserves instance IDs during training
and learns foreground probability, normalized center distance, and normalized
boundary distance with a U-Net matching the baseline architecture. Its
validation and inference stage extracts markers from the learned distances and
uses the predicted center distance as the seeded watershed surface. Tune
`center_distance_threshold`, `boundary_distance_threshold`, and
`minimum_marker_area` on the validation split.

## Setup

Install the dependencies from `requirements.txt`. Configure the dataset and
experiment in [`src/config.yaml`](./src/config.yaml). The directory reader
accepts `train/images` and `train/masks` (or `train/labels`) folders, with
optional `val` and `test` splits. Images and masks are loaded into CPU memory
before data loaders are created.

## Use

Run from the repository root with `src` on `PYTHONPATH`:

```python
from model import InstanSegModel

model = InstanSegModel()
history = model.train(epochs=2, batch_size=1, data_dir="./data")
metrics = model.eval(subset="test", data_dir="./data")
```

`InstanSegModel` also accepts a YAML configuration or a PyTorch checkpoint.
Each run saves its effective configuration, log, metrics CSV, and atomic
`last.pt` / `best.pt` checkpoints under `runs/<experiment_name>/`.

The public modules are available as `src.model` and `src.dataset` when using
the repository root as the Python import path.