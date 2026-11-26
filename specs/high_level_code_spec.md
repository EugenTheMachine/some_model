# High-Level Model API and Code Paradigm

## 1. Purpose

This specification describes the high-level model API pattern used in this project and general rules for reusing it in another codebase. The goal is to let a user initialize, train, resume, and evaluate a model through a small, stable interface without manually orchestrating internal model, data, optimizer, logging, and checkpoint components.

The public API is a façade around the lower-level implementation:

```text
configuration + optional checkpoint
                |
                v
          model wrapper
          /           \
  train pipeline    evaluation pipeline
          \           /
           internal model
```

In this repository, `InstanSegModel` is a plain Python wrapper that owns configuration, device selection, model construction/loading, and the end-to-end `train()` / `eval()` workflows. The actual PyTorch module is stored as `self.model`. The wrapper is exported from the package's public `__init__.py`.

The approach keeps normal usage concise while leaving specialized operations available internally. It should not bind the façade to any one architecture or task: model creation, data loading, loss calculation, and metric calculation remain replaceable components.

## 2. Public API shape

A transferable wrapper should provide:

```python
class ModelFacade:
    def __init__(self, config_or_checkpoint=None, **overrides):
        ...

    def train(self, **overrides):
        """Run an end-to-end training experiment and return its history."""
        ...

    def eval(self, subset="test", **overrides):
        """Evaluate the model on a selected dataset split and return metrics."""
        ...
```

The wrapper should make the common path convenient, while optional arguments support configuration files, checkpoints, and small per-run overrides. Complex implementation details should remain behind the wrapper rather than being duplicated in notebooks, command-line tools, or application code.

### Naming note

The current wrapper is not a subclass of `torch.nn.Module`, so its `.train()` and `.eval()` methods represent complete workflows. If a future wrapper subclasses `torch.nn.Module`, preserve the framework's `.train(mode)` / `.eval()` behavior-switching contract. In that case, use a distinct name such as `.fit()` and `.evaluate()` for end-to-end workflows rather than shadowing those standard methods.

## 3. Quick usage

### 3.1 Minimal initialization and training

When project defaults are suitable:

```python
model = ModelFacade()
history = model.train()
```

The call to `.train()` should read configured defaults, create the run's output directory, execute training, save artifacts, and return a structured history. The caller should not need to construct the optimizer or loop over epochs.

### 3.2 Small configuration overrides

Common values can be overridden without editing the shared config file:

```python
model = ModelFacade()
history = model.train(
    epochs=20,
    batch_size=4,
    learning_rate=1e-3,
    data_dir="./data",
    experiment_name="baseline_run",
)
```

The overrides apply to this wrapper instance/run and are persisted in the run's effective configuration. They should not silently mutate the repository's default configuration file.

### 3.3 Load configuration or a checkpoint

The constructor can accept a YAML configuration path or a model checkpoint path:

```python
configured_model = ModelFacade("./configs/experiment.yaml")
```

```python
pretrained_model = ModelFacade("./runs/baseline_run/checkpoints/best.pt")
```

When loading a checkpoint, use the matching architecture/configuration to construct the module before loading its weights. If an adjacent saved configuration exists, it can provide those settings. Report missing or incompatible files clearly.

### 3.4 Evaluation

Use the public evaluation method for one named split:

```python
metrics = pretrained_model.eval(subset="test")
print(metrics["loss"], metrics["accuracy"])
```

Evaluation should load or reuse the configured model, build an evaluation-only data loader, disable gradient updates, and return a stable mapping. It should not modify optimizer state, training history, or checkpoints.

### 3.5 Concrete repository example

The following illustrates the current public API with the repository's model wrapper. Data and run names are examples and can be changed:

```python
from instanseg import InstanSegModel

model = InstanSegModel()
history = model.train(
    epochs=2,
    batch_size=1,
    data_dir="./data",
    experiment_name="quick_run",
)

test_metrics = model.eval(subset="test", data_dir="./data")
```

The wrapper is also used in integration tests by creating a model, training it, checking that the run configuration was saved, and then evaluating the test split. See [test_train_eval_wrapper.py](../tests/test_train_eval_wrapper.py) for a representative API-level example.

## 4. Configuration-first construction

Use one configuration mapping as the source of truth:

1. Load project defaults from a configuration file.
2. Load an optional user-supplied configuration file or checkpoint-adjacent configuration.
3. Apply explicit constructor overrides.
4. Apply method-level overrides for a specific `.train()` or `.eval()` call.
5. Validate the fully merged result before expensive work begins.
6. Save the effective configuration with the run artifacts.

Define precedence once and apply it consistently. A common rule is that explicit method overrides take precedence over constructor overrides, which take precedence over loaded configuration, which takes precedence over defaults.

Organize settings by responsibility rather than by individual function:

- run metadata: input location, output/run name;
- data/preprocessing: split, input transformations, reproducibility seed;
- training: epochs, batch size, validation cadence, resume behavior;
- model: architecture and model-specific options;
- optimizer: optimizer type and optimization parameters.

Keep architecture-specific options in the model section or model factory. Keep task-specific loss, label, and metric settings with their owning component. Avoid letting the generic training wrapper accumulate assumptions about a particular output format or dataset.

## 5. Initialization lifecycle

Initialization should be cheap when a caller only needs a model object, but deterministic and explicit:

1. Load defaults and optional configuration.
2. Merge caller-provided overrides.
3. Validate configuration.
4. Select the execution device.
5. If a checkpoint is supplied, resolve its configuration, construct the matching module, and load the weights.
6. Otherwise, construct the model when needed for training or evaluation.

The repository wrapper builds the model from a configuration dictionary and moves it to the selected device. Its constructor supports loading a configuration path or checkpoint path. A transferable design should keep those mechanisms behind a single constructor/factory rather than requiring users to call internal builders directly.

Separate model construction from run execution: constructing a wrapper should not accidentally start training, create run artifacts, or download unrelated data. Any expensive pretrained-weight acquisition should be an explicit, documented part of model initialization.

## 6. End-to-end training lifecycle

The public `.train()` method orchestrates components in a predictable order:

1. Merge method overrides and validate the configuration.
2. Set random seeds and deterministic backend options.
3. Create an experiment directory and initialize logging.
4. Save and log the effective configuration.
5. Load data and construct train/validation loaders.
6. Construct the model, loss, optimizer, and optional precision/scheduler helpers.
7. Restore a checkpoint when resume is requested.
8. Repeatedly call `train_epoch()` and, at the configured cadence, `eval_epoch()`.
9. Record metrics, update early-stopping state, and save checkpoints.
10. Load the selected best/latest model and evaluate the held-out test data.
11. Return the complete training history.

The public wrapper should delegate the single-epoch operation to a reusable helper rather than embedding all batch-level logic in `.train()`. The epoch helper owns `module.train()`, forward/loss calculation, gradient accumulation, optimizer steps, and the training progress bar. The wrapper owns run-wide decisions and artifact management.

This separation makes it easier to:

- test one epoch independently of directory setup and checkpoint policy;
- call the same evaluation implementation during validation, final testing, and standalone evaluation;
- replace a model, data reader, or loss without rewriting the run lifecycle;
- keep notebook examples short and readable.

## 7. End-to-end evaluation lifecycle

The public `.eval(subset=..., **overrides)` should:

1. Resolve and validate the requested data split.
2. Load or reuse the configured model and weights.
3. Construct the evaluation data loader with evaluation-time transforms.
4. Set the underlying module to evaluation mode and disable gradient calculation.
5. Delegate batch execution and metric aggregation to a shared `eval_epoch()` helper.
6. Return loss and metrics as a stable mapping.

Unknown split names should be rejected rather than silently mapped to a different split. Evaluation should be side-effect-light: it may read data and load weights, but should not update training state or overwrite run checkpoints.

Keep task-specific prediction decoding and metrics behind an evaluator interface. The wrapper should report metrics but should not define their task semantics.

## 8. Results, logging, and artifacts

Each training run should be self-describing and isolated:

```text
runs/<experiment_name>/
  config.yaml
  train.log
  metrics.csv
  checkpoints/
    last.pt
    best.pt
```

The public training call should return an in-memory history as well as persist it. Each history record should identify its epoch and include the training result plus validation values when validation ran. The saved CSV should contain the same history so interrupted runs can be inspected without re-running code.

Log lifecycle events (configuration, model initialization or loading, data readiness, epoch start, training/validation results, checkpoint writes, early stopping, final evaluation, and completion). Progress bars complement lifecycle logs; they should not replace them.

Users should be able to identify which configuration and checkpoint produced a reported evaluation result. Store or reference the effective config beside the checkpoint, and log checkpoint paths used for loading and saving.

## 9. Extending the pattern to another project

To transfer this paradigm:

1. Implement a small façade class around the framework module and training components.
2. Keep the public path limited to construction, `.train()`, and evaluation.
3. Represent run choices in validated configuration and support explicit overrides.
4. Put model/data/task-specific logic behind factories or helper interfaces.
5. Keep epoch-level training and evaluation independently callable.
6. Save configuration, metrics, logs, checkpoints, and enough state for resume.
7. Expose stable return types from training and evaluation.
8. Write an integration test that exercises initialization, a short training run, artifact creation, and evaluation.
9. Keep examples focused on user choices; do not make users reproduce orchestration code in each notebook.

The pattern is useful when a project has multiple models, training entry points, notebooks, or evaluation workflows. A thin façade reduces duplicated orchestration while retaining the ability to swap implementations behind its stable API.

## 10. References in this repository

- [instanseg_model.py](../instanseg/instanseg_model.py) — wrapper construction, configuration merging, and end-to-end methods.
- [__init__.py](../instanseg/__init__.py) — public package exports.
- [training_spec.md](./training_spec.md) — training loop, checkpointing, resume, and metrics contract.

