"""Configuration, reproducibility, logging, and checkpoint utilities."""

from __future__ import annotations

import csv
import logging
import os
import random
import re
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config.yaml"

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "meta": {"data_dir": "../data", "experiment_name": None},
    "preprocessing": {
        "imgsz": 512,
        "train_data_ratio": 1.0,
        "seed": 42,
        "dim_in": 1,
        "augmentation_type": "minimal",
        "dataset": None,
        "source_dataset": ["all"],
        "modality": None,
    },
    "training": {
        "epochs": 10,
        "patience": 5,
        "batch_size": 4,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "val_interval": 1,
        "warmup_epochs": 2,
        "resume": False,
        "amp": False,
        "num_workers": 0,
        "length_of_epoch": None,
    },
    "model": {
        "model_name": "affinity_guided_mlp",
        "num_classes": 2,
        "freeze_backbone_epochs": 0,
        "pretrained_backbone": False,
        "dropout": 0.0,
        "input_channels": 1,
        "backbone_channels": [64, 128, 256, 512, 1024],
        "unet_channels": [32, 64, 128, 256, 512],
        "feature_channels": 32,
        "num_slots": 32,
        "mlp_channels": [64, 32],
        "use_aggregation": True,
        "recurrent_hidden_channels": 32,
        "recurrent_iterations": 3,
        "slot_iterations": 2,
        "minimum_instance_area": 0,
    },
    "optimizer": {
        "type": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "backbone_lr_mult": 0.1,
    },
}

FLAT_OVERRIDE_SECTIONS = {
    "data_dir": "meta",
    "experiment_name": "meta",
    "imgsz": "preprocessing",
    "train_data_ratio": "preprocessing",
    "seed": "preprocessing",
    "dim_in": "preprocessing",
    "augmentation_type": "preprocessing",
    "dataset": "preprocessing",
    "source_dataset": "preprocessing",
    "modality": "preprocessing",
    "epochs": "training",
    "patience": "training",
    "batch_size": "training",
    "val_ratio": "training",
    "test_ratio": "training",
    "val_interval": "training",
    "warmup_epochs": "training",
    "resume": "training",
    "amp": "training",
    "num_workers": "training",
    "length_of_epoch": "training",
    "model_name": "model",
    "num_classes": "model",
    "freeze_backbone_epochs": "model",
    "pretrained_backbone": "model",
    "dropout": "model",
    "input_channels": "model",
    "backbone_channels": "model",
    "unet_channels": "model",
    "feature_channels": "model",
    "num_slots": "model",
    "mlp_channels": "model",
    "use_aggregation": "model",
    "recurrent_hidden_channels": "model",
    "recurrent_iterations": "model",
    "recurrent_hidden_dim": "model",
    "hidden_channels": "model",
    "slot_iterations": "model",
    "minimum_instance_area": "model",
    "num_recurrent_iterations": "model",
    "slot_update_iterations": "model",
    "optimizer_type": "optimizer",
    "learning_rate": "optimizer",
    "weight_decay": "optimizer",
    "momentum": "optimizer",
    "backbone_lr_mult": "optimizer",
}


def _merge_mapping(base: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            _merge_mapping(base[key], value)
        else:
            base[key] = value


def load_config(
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load defaults, a YAML mapping, and explicit non-null overrides."""
    config = {
        section: values.copy() for section, values in DEFAULT_CONFIG.items()
    }
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if config_path is not None and not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    if path.is_file():
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Configuration must be a mapping: {path}")
        _merge_mapping(config, loaded)

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key in config and isinstance(config[key], Mapping) and isinstance(value, Mapping):
            _merge_mapping(config[key], value)
        elif key in FLAT_OVERRIDE_SECTIONS:
            section = FLAT_OVERRIDE_SECTIONS[key]
            config[section][
                "type" if key == "optimizer_type" else key
            ] = value
        else:
            raise ValueError(f"Unknown configuration override: {key}")
    validate_config(config)
    return config


def _require_int(value: Any, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _require_finite(value: Any, name: str, minimum: float, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    if not np.isfinite(value) or value < minimum or (
        maximum is not None and value > maximum
    ):
        bounds = f" in [{minimum}, {maximum}]" if maximum is not None else f" >= {minimum}"
        raise ValueError(f"{name} must be finite and{bounds}.")


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject invalid configuration values before creating expensive resources."""
    required_sections = {"meta", "preprocessing", "training", "model", "optimizer"}
    if not required_sections.issubset(config):
        raise ValueError("Configuration is missing required sections.")
    meta = config["meta"]
    preprocessing = config["preprocessing"]
    training = config["training"]
    model = config["model"]
    optimizer = config["optimizer"]

    if not isinstance(meta.get("data_dir"), (str, Path)) or not str(meta["data_dir"]):
        raise ValueError("meta.data_dir must be a non-empty path.")
    meta["data_dir"] = str(meta["data_dir"])
    experiment_name = meta.get("experiment_name")
    if experiment_name is not None and (
        not isinstance(experiment_name, str)
        or not experiment_name
        or Path(experiment_name).name != experiment_name
        or experiment_name in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9._-]+", experiment_name)
    ):
        raise ValueError("experiment_name must be a safe folder name.")

    _require_int(preprocessing["imgsz"], "imgsz", 1)
    _require_int(preprocessing["seed"], "seed", 0)
    _require_int(preprocessing["dim_in"], "dim_in", 1)
    _require_finite(
        preprocessing["train_data_ratio"], "train_data_ratio", 0.0, 1.0
    )
    if preprocessing["train_data_ratio"] == 0:
        raise ValueError("train_data_ratio must be greater than zero.")
    if preprocessing.get("augmentation_type") not in {
        "minimal",
        "kornia_intensity",
        "heavy",
        "brightfield_only",
    }:
        raise ValueError("Unsupported augmentation_type.")
    if preprocessing.get("modality") not in {
        None,
        "Brightfield",
        "phase-contrast",
    }:
        raise ValueError("modality must be Brightfield or phase-contrast.")
    if not isinstance(preprocessing.get("source_dataset"), (list, tuple)):
        raise ValueError("source_dataset must be a list of dataset names.")

    for key in ("epochs", "batch_size", "val_interval"):
        _require_int(training[key], key, 1)
    for key in ("patience", "warmup_epochs", "num_workers"):
        _require_int(training[key], key, 0)
    for key in ("val_ratio", "test_ratio"):
        _require_finite(training[key], key, 0.0, 1.0)
    if training["val_ratio"] + training["test_ratio"] > 1:
        raise ValueError("val_ratio and test_ratio must sum to at most 1.")
    resume = training["resume"]
    if not isinstance(resume, bool) and (
        not isinstance(resume, str) or not resume.strip()
    ):
        raise ValueError("resume must be a boolean or non-empty checkpoint path.")
    for key in ("amp",):
        if not isinstance(training[key], bool):
            raise ValueError(f"{key} must be a boolean.")
    if training["length_of_epoch"] is not None:
        _require_int(training["length_of_epoch"], "length_of_epoch", 1)

    if model["model_name"] not in {
        "affinity_guided_mlp",
        "affinity_guided_recurrent",
        "affinity_guided_recurrent_instance_segmentation",
        "affinity_guided_recurrent_net",
        "affinity_guided_recurrent_unet",
        "instanseg_unet",
        "semantic_unet",
    }:
        raise ValueError(f"Unsupported model_name: {model['model_name']!r}")
    for key in ("num_classes", "input_channels", "feature_channels", "num_slots"):
        _require_int(model[key], key, 1)
    if "recurrent_hidden_dim" in model:
        model["recurrent_hidden_channels"] = model.pop("recurrent_hidden_dim")
    if "hidden_channels" in model:
        model["recurrent_hidden_channels"] = model.pop("hidden_channels")
    if "num_recurrent_iterations" in model:
        model["recurrent_iterations"] = model.pop("num_recurrent_iterations")
    if "slot_update_iterations" in model:
        model["slot_iterations"] = model.pop("slot_update_iterations")
    for key in (
        "recurrent_hidden_channels",
        "recurrent_iterations",
        "slot_iterations",
    ):
        _require_int(model[key], key, 1)
    _require_int(model["minimum_instance_area"], "minimum_instance_area", 0)
    _require_int(model["freeze_backbone_epochs"], "freeze_backbone_epochs", 0)
    _require_finite(model["dropout"], "dropout", 0.0, 1.0)
    if model["dropout"] == 1:
        raise ValueError("dropout must be less than 1.")
    if (
        not isinstance(model["backbone_channels"], (list, tuple))
        or len(model["backbone_channels"]) < 2
    ):
        raise ValueError("backbone_channels must contain at least two widths.")
    for channel in model["backbone_channels"]:
        _require_int(channel, "backbone channel", 1)
    if (
        not isinstance(model["unet_channels"], (list, tuple))
        or len(model["unet_channels"]) < 2
    ):
        raise ValueError("unet_channels must contain at least two widths.")
    for channel in model["unet_channels"]:
        _require_int(channel, "U-Net channel", 1)
    if model["model_name"] == "semantic_unet":
        downsampling_factor = 2 ** (len(model["unet_channels"]) - 1)
        if preprocessing["imgsz"] % downsampling_factor:
            raise ValueError(
                "imgsz must be divisible by the semantic U-Net downsampling factor "
                f"({downsampling_factor})."
            )
    if (
        not isinstance(model["mlp_channels"], (list, tuple))
        or len(model["mlp_channels"]) != 2
    ):
        raise ValueError("mlp_channels must contain exactly two values.")
    for channel in model["mlp_channels"]:
        _require_int(channel, "MLP channel", 1)
    if not isinstance(model["use_aggregation"], bool):
        raise ValueError("use_aggregation must be a boolean.")
    if not isinstance(model["pretrained_backbone"], bool):
        raise ValueError("pretrained_backbone must be a boolean.")

    if not isinstance(optimizer["type"], str):
        raise ValueError("optimizer.type must be a string.")
    optimizer_type = optimizer["type"].lower()
    if optimizer_type not in {"adamw", "sgd"}:
        raise ValueError("optimizer.type must be AdamW or SGD.")
    _require_finite(optimizer["learning_rate"], "learning_rate", 0.0)
    if optimizer["learning_rate"] == 0:
        raise ValueError("learning_rate must be greater than zero.")
    _require_finite(optimizer["weight_decay"], "weight_decay", 0.0)
    if optimizer_type == "sgd":
        _require_finite(optimizer["momentum"], "momentum", 0.0, 1.0)
        if optimizer["momentum"] == 1:
            raise ValueError("momentum must be less than one.")
    _require_finite(optimizer["backbone_lr_mult"], "backbone_lr_mult", 0.0)
    optimizer["type"] = "SGD" if optimizer_type == "sgd" else "AdamW"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state() -> dict[str, Any]:
    """Capture process-wide random-number-generator state for checkpoints."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore the RNG states captured by :func:`rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def create_logger(log_path: Path, name: str = "some_model") -> logging.Logger:
    """Create a fresh console and append-mode UTF-8 file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def generated_experiment_name() -> str:
    """Return a filesystem-safe name for a new experiment."""
    return datetime.now().strftime("ag_mlp_%Y%m%d_%H%M%S")


def save_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    """Atomically save a checkpoint in the target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(checkpoint), temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a trusted local PyTorch checkpoint with a clear path error."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    return checkpoint


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    """Persist full epoch history as a stable CSV table."""
    metric_names = sorted(
        {
            key
            for record in history
            for key in record
            if key not in {"epoch", "train_loss", "train_time", "val_loss"}
        }
    )
    fieldnames = ["epoch", "train_loss", "val_loss", "train_time", *metric_names]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(
                {
                    key: record.get(key, "")
                    for key in fieldnames
                }
            )
