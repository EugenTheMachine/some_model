# Training Procedure Specification

## 1. Scope and design

This document describes the reusable training and evaluation procedure implemented by this project. It specifies the lifecycle around a model without prescribing a particular model architecture, objective function, input modality, output representation, or task-specific postprocessor.

Keep task-specific behavior behind explicit interfaces:

- a model factory builds the configured model;
- a loss function computes the training objective from model outputs and targets;
- dataset and preprocessing code returns batches in the model's expected format;
- an evaluator converts predictions and targets into the quantities required by the selected task's metrics.

The training lifecycle should work with those components without embedding their domain-specific assumptions. The implementation uses PyTorch and is organized around:

- `train_epoch()` for one optimizer epoch;
- `eval_epoch()` for one evaluation pass;
- `<ModelClass>.train()` for end-to-end experiment orchestration;
- `<ModelClass>.eval()` for end-to-end evaluation on a selected data split.

The existing code has a reusable `train_epoch()` helper and end-to-end `train()` / `eval()` methods on its model wrapper. Validation and final test loops are currently inline; `eval_epoch()` is the common reusable interface they should share.

## 2. Configuration

### 2.1 Storage and precedence

Store defaults in a project-root `config.yaml`, grouped into these sections:

```yaml
meta: {}
preprocessing: {}
training: {}
model: {}
optimizer: {}
```

Load the YAML mapping, then apply non-null keyword overrides supplied during construction or to `.train()` / `.eval()`. Flat public argument names map to the appropriate section. Validate the merged configuration before creating datasets or models. Invalid values must raise a clear error rather than being silently clamped or replaced.

Before a run starts, save the effective merged configuration to `runs/<experiment_name>/config.yaml`. Log that same configuration so the persisted values, logs, and actual run agree.

The following table documents the general-purpose settings used by the current project. Model- or task-specific settings belong in their own section or component and should not be required by the generic training loop.

### 2.2 `meta`

| Parameter | Type and default | Meaning and allowed values |
|---|---|---|
| `data_dir` | string, `"../data"` | Root location of the dataset. Must resolve to readable data accepted by the selected dataset reader. |
| `experiment_name` | string or null, `null` | Name of the run directory under `runs/`. If omitted, use the project's documented generated default. Treat it as a folder name and reject unsafe path values. |

### 2.3 `preprocessing`

| Parameter | Type and default | Meaning and allowed values |
|---|---|---|
| `imgsz` | integer, `512` | Target spatial size or preprocessing scale consumed by the dataset pipeline. Its precise geometric meaning belongs to the dataset/preprocessing specification. Require a positive integer when applicable. |
| `train_data_ratio` | float, `1.0` | Fraction of the training split to use. Require `0 < value <= 1`; `1.0` uses the full split. Selection must be deterministic for a fixed seed and split ordering. |
| `seed` | integer, `42` | Root seed for initialization, splitting, sampling, data-loader workers, and other stochastic operations. |

### 2.4 `training`

| Parameter | Type and default | Meaning and allowed values |
|---|---|---|
| `epochs` | integer, `10` | Total target number of epochs. Require a positive integer. On resume, this is the final target epoch, not the number of additional epochs. |
| `patience` | integer, `5` | Number of validation checks without improvement in the selected validation objective before early stopping. Require a non-negative integer. |
| `batch_size` | integer, `4` | Samples per DataLoader batch. Require a positive integer. |
| `val_ratio` | float, `0.2` | Fraction reserved for validation when the dataset reader must create a split. Require `0 <= value <= 1`; validate split ratios together against dataset size and split policy. |
| `test_ratio` | float, `0.2` | Fraction reserved for testing when the dataset reader must create a split. Require `0 <= value <= 1`; validate split ratios together against dataset size and split policy. |
| `val_interval` | integer, `1` | Validate every N epochs and always validate on the final epoch. Require a positive integer. |
| `warmup_epochs` | integer, `2` | Number of initial epochs to linearly ramp the optimizer learning rate from zero to its configured base value. Require a non-negative integer; zero disables warmup. At epoch `e` during warmup, use `base_lr * e / warmup_epochs`. |
| `resume` | boolean or string, `false` | `false` starts a fresh run; `true` selects the latest checkpoint in the current experiment; a non-empty path selects an explicit checkpoint. The missing-checkpoint policy must be explicit and logged. Current code logs the absence and starts from scratch. |
| `amp` | boolean, `false` | Enable automatic mixed precision where supported. Current implementation uses CUDA FP16 autocast and gradient scaling; if CUDA is unavailable, it logs the fallback and runs full precision. |

### 2.5 `model`

| Parameter | Type and default | Meaning and allowed values |
|---|---|---|
| `model_name` | string, selected project default | Identifier resolved by the model factory. The allowed set depends on registered model implementations and their installed dependencies. Validate against that registry and report unsupported identifiers; the generic training loop must not contain a hard-coded list of task-specific architectures. |
| `num_classes` | integer, `2` | Intended number of semantic categories for model/task configurations that use this concept. It is not necessarily the model's output-channel count. Require a positive integer when the chosen model uses it; otherwise it may be omitted or treated as informational. |
| `freeze_backbone_epochs` | integer, `0` | Optional number of initial epochs during which a model component designated as its backbone is frozen. Require a non-negative integer. Apply only when the selected model exposes such a component; otherwise document that it is ignored or reject it. |
| `pretrained_backbone` | boolean, `true` | Whether to initialize supported model components from pretrained weights. It is relevant only for model factories that implement pretrained initialization. A full training checkpoint takes precedence during resume. |
| `dropout` | float, `0.0` | Dropout probability for models that support dropout. Require `0 <= value < 1`. |

### 2.6 `optimizer`

| Parameter | Type and default | Meaning and allowed values |
|---|---|---|
| `type` | string, `"AdamW"` | Optimizer implementation. Current code supports `"SGD"` and otherwise constructs AdamW. A strict configuration contract should allow only `"AdamW"` or `"SGD"` (case-insensitive) and reject other values instead of silently selecting a fallback. |
| `learning_rate` | float, `0.001` | Base learning rate. Require a finite value greater than zero. |
| `weight_decay` | float, `0.0001` | Optimizer weight decay. Require a finite, non-negative value. |
| `momentum` | float, `0.9` | SGD momentum; ignored by AdamW. Require `0 <= value < 1` when using SGD. |
| `backbone_lr_mult` | float, `0.1` | Optional multiplier used to create a lower learning rate for a separately identified model component: `component_lr = learning_rate * backbone_lr_mult`. Require a finite, non-negative value. Use only if the selected model defines such parameter groups. |

### 2.7 Separation of generic and task-specific configuration

The training loop owns run-level options such as epochs, optimizer settings, validation cadence, checkpointing, and reproducibility. Model construction may define additional model-specific values; data loading and preprocessing may define additional data-specific values; evaluation may define task-specific metric parameters.

Do not add task-specific output layouts, labels, loss names, modalities, or postprocessing controls to the generic loop. Pass them to the relevant model, dataset, loss, or evaluator through documented interfaces.

## 3. Reproducibility

At the start of a fresh run, seed all global random-number generators before data splitting, model construction, or stochastic loader creation:

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Seed DataLoader workers' Python, NumPy, and PyTorch generators from each worker's PyTorch seed. Use explicit sampler generators. Ensure data splitting and any subset selection are deterministic for the configured seed.

For exact continuation across interruption, preserve or deterministically derive:

- Python, NumPy, PyTorch CPU, and all CUDA RNG states;
- sampler/generator state and the epoch number used to derive shuffle order;
- any stateful dataset or augmentation RNG, or deterministic per-sample random seeds;
- model, optimizer, and mixed-precision scaler states;
- current epoch, best validation objective, early-stopping counter, and complete metrics history;
- effective configuration and dataset/split identity.

Restore these before the next stochastic epoch iteration. Restoring model weights alone is initialization for evaluation/fine-tuning, not a faithful training resume. If a learning-rate scheduler is added, checkpoint and restore its state and step count as well.

The current checkpoint utility stores Python, NumPy, PyTorch CPU, and CUDA RNG state. The training wrapper restores these and the optimizer/scaler/history/epoch state. It does not explicitly store separate sampler state or a split manifest, so exact continuation should be tested for each sampler and worker setup rather than assumed. Even with deterministic settings, bitwise equality is not guaranteed across hardware, library versions, or nondeterministic operations.

## 4. Training and evaluation components

### 4.1 `train_epoch()`

Implement a reusable epoch-level function in the training utilities. One suitable interface is:

```python
train_epoch(
    model,
    device,
    train_loader,
    loss_fn,
    optimizer,
    *,
    scaler=None,
    amp=False,
    accumulation_steps=1,
    gradient_clip_norm=20.0,
) -> tuple[float, float]
```

Responsibilities:

1. Set `model.train()`.
2. Iterate through batches with a tqdm progress bar.
3. Move model inputs and targets to the selected device.
4. Run forward and loss computation under autocast only when mixed precision is enabled and supported.
5. Divide loss by `accumulation_steps` before backpropagation.
6. At each accumulation boundary and after the final incomplete accumulation group, unscale gradients if needed, clip the gradient norm, update optimizer/scaler, and clear gradients.
7. Return mean per-batch loss and elapsed time. The current implementation averages batch losses without weighting by batch size.

The repository's current `train_epoch()` performs these steps using an argument bundle and optional scaler. Its signature may remain as-is or be made more explicit, provided call sites preserve these semantics.

### 4.2 `eval_epoch()`

Use one shared evaluation helper for validation, final testing, and standalone evaluation:

```python
eval_epoch(
    model,
    device,
    eval_loader,
    loss_fn,
    metric_fn,
    *,
    amp=False,
    progress_description="Validation",
) -> dict
```

Responsibilities:

1. Set `model.eval()` and run under `torch.no_grad()`.
2. Compute loss using the same objective and compatible precision policy as training.
3. Pass outputs and targets to the task-provided `metric_fn`; the helper must not assume a particular output representation or metric-matching algorithm.
4. Display a progress bar for the evaluation loader.
5. Return the mean loss and the evaluator's documented metrics in a stable mapping.

Use the same evaluation helper for validation and test to keep prediction handling, loss aggregation, metrics, and progress reporting consistent.

### 4.3 `<ModelClass>.train()`

The model class's public `.train(**overrides)` method runs an entire experiment and returns a per-epoch history. It is an orchestration method, not the epoch-level function.

Run stages in this order:

1. Merge overrides into the loaded configuration and validate all values.
2. Seed random generators and configure deterministic backend settings.
3. Create `runs/<experiment_name>/` and its `checkpoints/` subdirectory.
4. Initialize console and file logging.
5. Save and log the effective configuration.
6. Load and split the data; log successful loading, split sizes, and split source.
7. Build data loaders and initialize preprocessing/augmentation; log completion.
8. Build the model, objective, and metric evaluator.
9. Create optimizer parameter groups, optional scheduler, optional component-freezing policy, and mixed-precision scaler.
10. If resuming, restore the latest selected checkpoint and all continuation state; otherwise log model initialization.
11. For each epoch from `start_epoch` through the total configured epoch target:
    - log `EPOCH <current>/<total>`;
    - run `train_epoch()`;
    - validate when `(epoch % val_interval == 0) or (epoch == epochs)`;
    - append and persist the epoch's metrics record;
    - update the best validation objective and patience counter on validation epochs;
    - save the latest checkpoint and, when improved, the best checkpoint;
    - apply early stopping after the configured number of non-improving validation checks.
12. Load the best checkpoint for final test evaluation, falling back to the latest checkpoint only if no best checkpoint exists.
13. Evaluate the held-out test split without gradient updates and log loss and metrics.
14. Return the complete epoch history.

For epochs without validation, represent validation values as `"N/A"` or a consistently documented null value; do not report stale validation results as if measured in that epoch.

### 4.4 `<ModelClass>.eval()`

The public `.eval(subset="test", **overrides)` method evaluates a loaded or configured model on a selected data split and returns a mapping containing loss and task metrics.

- Support explicitly named evaluation splits such as `"val"` and `"test"` as appropriate for the dataset interface.
- Reject unknown split names instead of silently substituting another split.
- Build an evaluation-only loader with no gradient updates.
- Initialize the model from configuration if needed and load supplied checkpoint weights when applicable.
- Delegate computation to `eval_epoch()`.
- Do not update optimizer state, checkpoints, or training history.

Use the same metric schema as the final test stage in `.train()`.

## 5. Logging

Use a named Python logger at `INFO` level, with timestamps, writing to both the console and an append-mode UTF-8 `train.log` in the experiment directory. Remove existing handlers for that logger before adding new handlers so repeated runs in the same process do not duplicate records.

Log these major events:

1. Experiment name, artifact directory, and effective configuration.
2. Model initialized or checkpoint path loaded.
3. Resume selection, restored epoch, and any missing or incompatible state.
4. Data loaded successfully, split sizes, and split source.
5. Data-loader and preprocessing initialization.
6. Device, optimizer, learning rates, and mixed-precision mode or fallback.
7. Warmup learning rates and each `EPOCH n/N` start.
8. Training progress via tqdm.
9. Validation start and progress bar on validation epochs.
10. Per-epoch mean training loss, validation loss if evaluated, and validation metrics.
11. Checkpoint writes and best-checkpoint updates.
12. Early-stopping event and stopping epoch.
13. Final test progress bar, mean loss, and metrics.
14. Completion summary, elapsed time, and artifact paths.

Progress bars supplement structured lifecycle logs. If they are captured in a log file, use a logging-compatible progress writer or separately log batch counts so progress rendering does not corrupt log records. Do not log full data batches.

## 6. Checkpointing and resume

### 6.1 Experiment artifacts

Maintain the following artifacts:

```text
runs/<experiment_name>/
  config.yaml
  train.log
  metrics.csv
  last.pt
  best.pt
  checkpoints/
    last.pt
    best.pt
```

The experiment-root checkpoint copies support existing callers; `checkpoints/last.pt` is the canonical resume location.

`metrics.csv` contains one row per completed epoch, with at least:

```text
epoch,train_loss,val_loss,precision,recall,accuracy,f1
```

Metric columns may be extended by a task evaluator, but their meaning and aggregation must be documented. Persist the complete history after each epoch so resumed runs retain earlier rows.

### 6.2 Checkpoint contents and cadence

Each checkpoint should include:

- `epoch`: last fully completed epoch, one-based;
- `model_state_dict`;
- `optimizer_state_dict`;
- `scheduler_state_dict` when a scheduler is used;
- `scaler_state_dict` when mixed precision is active;
- best validation objective and early-stopping counter;
- complete metrics history;
- global RNG states;
- sampler/generator and stateful data-pipeline state, or deterministic derivation state;
- effective config and dataset/split identity.

Save `last.pt` after every completed epoch. On validation epochs, compare the selected validation objective (current code uses lower validation loss as better). If it improves, reset patience and update `best.pt`; otherwise increment patience. Only validation checks change the patience counter.

Write checkpoints crash-safely: serialize to a temporary file in the same directory and atomically replace the target after serialization succeeds.

### 6.3 Resume procedure

When resume is requested:

1. Resolve an explicit checkpoint path first, then the current experiment's `checkpoints/last.pt`, then supported experiment-root or constructor-supplied checkpoint paths.
2. Construct compatible model and optimizer objects, then restore model weights, optimizer state, scaler/scheduler state when present, and continuation metadata.
3. Validate checkpoint/config/model compatibility. Fail clearly for incompatible model state; log any optional optimizer state that cannot be restored.
4. Restore RNG, sampler, and stateful data-pipeline state before creating the next epoch's stochastic iteration.
5. Set `start_epoch = checkpoint["epoch"] + 1`; treat configured `epochs` as the total target.
6. Continue writing into the same experiment directory without truncating metrics history.

If no checkpoint is found, the current implementation logs the condition and starts from scratch. Keep that behavior explicit in logs and tests, or choose a strict error policy consistently.

The reproducibility acceptance criterion is that a fixed run to N epochs matches a run to K epochs followed by resume to N, within a documented numerical tolerance, for both metric history and final parameters.

## 7. Metrics

Metrics are produced by a task-provided evaluator and calculated on held-out evaluation data, not from training batches. The generic training loop requires the evaluator to return stable scalar values for:

- `precision`
- `recall`
- `accuracy`
- `f1`

The evaluator must document how it derives TP, FP, and FN for its task and how predictions are matched to targets. The training loop must not assume pixel, object, or class matching semantics.

For the count-based convention used by the current metric utility, aggregate TP, FP, and FN over the evaluation dataset, then compute:

```text
precision = TP / (TP + FP + epsilon)
recall    = TP / (TP + FN + epsilon)
accuracy  = TP / (TP + FP + FN + epsilon)
F1        = 2 * TP / (2 * TP + FP + FN + epsilon)
```

Here `epsilon` is a small positive value preventing division by zero. This `accuracy` definition is the requested TP/FP/FN-based score; it is not the conventional classification accuracy formula involving true negatives. F1 is the harmonic mean of precision and recall.

The current metric utility aggregates counts globally across the dataset; it does not average per-sample metric values. Empty evaluation input currently yields zero for all four metrics. Preserve a documented policy for empty inputs and zero denominators. If the task's evaluator uses a different metric convention, it must expose that convention clearly rather than changing the meaning silently.

## 8. Required validation tests

At minimum, test:

1. Config defaults, override precedence, and rejection of invalid ranges and unsupported discrete values.
2. `train_epoch()` performs updates, averages losses, clips gradients, handles a final partial accumulation group, and enables mixed precision only on supported devices.
3. `eval_epoch()` does not update model parameters and returns a stable metric schema and loss.
4. Checkpoint creation after an epoch, best-checkpoint selection, history preservation, and loading on CPU or accelerator.
5. Resume starts at the next epoch and reaches the total configured epoch target.
6. A deterministic N-epoch run matches a K-then-resume-to-N run within the declared tolerance.
7. Validation follows `val_interval` and occurs on the final epoch; patience changes only after validation.
8. Metric formulas, dataset aggregation, zero-denominator and empty-input behavior, and task-specific matching semantics.
9. Public `.eval()` accepts supported split names, rejects unknown names, returns loss and metrics, and does not mutate training artifacts.
10. Console and file logs contain required lifecycle events without duplicate handlers.
