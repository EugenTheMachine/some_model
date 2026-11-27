# Cell Segmentation

This project implements a single-pass U-Net instance-segmentation model for
microscopy images. It predicts foreground, four directional affinities, and
per-pixel assignments to a fixed number of instance slots. Hungarian matching
is used only to construct training targets; inference uses a per-pixel slot
argmax without clustering or connected-component post-processing.

It also provides a classical U-Net binary semantic-segmentation baseline.
Select `model_name="semantic_unet"` and configure `unet_channels` to train it
through the same data-loading and checkpointing facade. In this mode, all
nonzero ground-truth mask pixels are converted to foreground value 1.

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