"""Affinity-guided U-Net and its high-level training/evaluation façade."""

# pylint: disable=invalid-name

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from .decoder import Decoder, UNetDecoderFeatures
from .encoder import UNetEncoder
from .head import (
    AffinityGuidedRecurrentHead,
    DiscriminativeEmbeddingLearnedAssignmentHead,
    InstanceSegmentationHead,
)
from .semantic import (
    SemanticSegmentationLoss,
    SemanticSegmentationUNet,
    eval_semantic_epoch,
)
from .utils import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    create_logger,
    generated_experiment_name,
    load_checkpoint,
    load_config,
    restore_rng_state,
    rng_state,
    save_checkpoint,
    seed_everything,
    validate_config,
    write_history,
)


class UNetBackbone(nn.Module):
    """Four-downsampling-stage U-Net feature extractor."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (64, 128, 256, 512, 1024),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("U-Net backbone needs at least two channel stages.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.encoder = UNetEncoder(
            in_channels, channels, norm="BATCH", act="GELU"
        )
        self.decoder = UNetDecoderFeatures(channels, norm="BATCH", act="ReLU")
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.out_channels = channels[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return full-resolution U-Net features."""
        bottleneck, skips = self.encoder(inputs)
        return self.dropout(self.decoder(bottleneck, skips))


class AffinityGuidedMLPNet(nn.Module):
    """One-pass cell instance segmentation with affinity-guided assignment."""

    def __init__(
        self,
        in_channels: int = 1,
        backbone_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        feature_channels: int = 32,
        num_slots: int = 32,
        mlp_channels: Sequence[int] = (64, 32),
        dropout: float = 0.0,
        use_aggregation: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = UNetBackbone(
            in_channels=in_channels,
            channels=backbone_channels,
            dropout=dropout,
        )
        self.head = InstanceSegmentationHead(
            in_channels=self.backbone.out_channels,
            num_slots=num_slots,
            feature_channels=feature_channels,
            hidden_channels=mlp_channels,
            use_aggregation=use_aggregation,
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return prediction logits, probabilities, and assigned instance IDs."""
        if inputs.ndim != 4:
            raise ValueError("Model input must have shape (N, C, H, W).")
        if not inputs.is_floating_point():
            raise TypeError("Model input must be a floating-point tensor.")
        features = self.backbone(inputs)
        return self.head(features)

    @torch.no_grad()
    def predict_instances(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return background ID 0 and positive slot IDs for a batch."""
        self.eval()
        return self(inputs)["instance_map"]


class DiscriminativeEmbeddingLearnedAssignmentNet(nn.Module):
    """U-Net instance segmentation with embeddings and learned slot assignment."""

    def __init__(
        self,
        in_channels: int = 1,
        backbone_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        num_slots: int = 32,
        embedding_dim: int = 8,
        assignment_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = UNetBackbone(
            in_channels=in_channels,
            channels=backbone_channels,
            dropout=dropout,
        )
        self.head = DiscriminativeEmbeddingLearnedAssignmentHead(
            in_channels=self.backbone.out_channels,
            num_slots=num_slots,
            embedding_dim=embedding_dim,
            assignment_dim=assignment_dim,
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return full-resolution DELA predictions and instance assignments."""
        if inputs.ndim != 4:
            raise ValueError("Model input must have shape (N, C, H, W).")
        if not inputs.is_floating_point():
            raise TypeError("Model input must be a floating-point tensor.")
        return self.head(self.backbone(inputs))

    @torch.no_grad()
    def predict_instances(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return background ID 0 and positive slot IDs for a batch."""
        self.eval()
        return self(inputs)["instance_map"]


class AffinityGuidedRecurrentNet(nn.Module):
    """Affinity-guided recurrent pixel grouping with fixed instance slots."""

    def __init__(
        self,
        in_channels: int = 1,
        backbone_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        feature_channels: int = 32,
        num_slots: int = 32,
        recurrent_hidden_channels: int = 32,
        recurrent_iterations: int = 3,
        slot_iterations: int = 2,
        dropout: float = 0.0,
        minimum_instance_area: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        recurrent_hidden_channels = kwargs.pop(
            "recurrent_hidden_dim",
            kwargs.pop("hidden_channels", recurrent_hidden_channels),
        )
        recurrent_iterations = kwargs.pop(
            "num_recurrent_iterations", recurrent_iterations
        )
        slot_iterations = kwargs.pop(
            "slot_update_iterations",
            kwargs.pop("num_slot_iterations", slot_iterations),
        )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown recurrent model arguments: {unknown}")
        self.backbone = UNetBackbone(
            in_channels=in_channels,
            channels=backbone_channels,
            dropout=dropout,
        )
        self.head = AffinityGuidedRecurrentHead(
            in_channels=self.backbone.out_channels,
            num_slots=num_slots,
            feature_channels=feature_channels,
            recurrent_hidden_channels=recurrent_hidden_channels,
            recurrent_iterations=recurrent_iterations,
            slot_iterations=slot_iterations,
            minimum_instance_area=minimum_instance_area,
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return recurrent predictions and a dense instance label map."""
        if inputs.ndim != 4:
            raise ValueError("Model input must have shape (N, C, H, W).")
        if not inputs.is_floating_point():
            raise TypeError("Model input must be a floating-point tensor.")
        features = self.backbone(inputs)
        return self.head(features)

    @torch.no_grad()
    def predict_instances(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a compact positive-integer instance map."""
        self.eval()
        return self(inputs)["instance_map"]


AffinityGuidedRecurrentUNet = AffinityGuidedRecurrentNet
RecurrentInstanceSegmentationNet = AffinityGuidedRecurrentNet


class InstanSeg_UNet(nn.Module):
    """Baseline U-Net matching the encoder/decoder module specifications."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | Sequence[int] | Sequence[Sequence[int]] = ((2, 2, 1),),
        layers: Sequence[int] = (256, 128, 64, 32),
        norm: str | None = "BATCH",
        dropout: float = 0.0,
        act: str = "GELU",
    ) -> None:
        super().__init__()
        if in_channels in (0, None):
            in_channels = 3
        encoder_channels = list(reversed(layers))
        if not encoder_channels:
            raise ValueError("layers must contain at least one width.")
        self.encoder = UNetEncoder(
            in_channels, encoder_channels, norm=norm, act=act
        )
        decoder_layers = list(reversed(encoder_channels))
        if isinstance(out_channels, int):
            output_specs = [[out_channels]]
        elif out_channels and isinstance(out_channels[0], int):
            output_specs = [list(out_channels)]  # type: ignore[arg-type]
        else:
            output_specs = [list(spec) for spec in out_channels]  # type: ignore[arg-type]
        if not output_specs or any(not spec for spec in output_specs):
            raise ValueError("out_channels must include at least one output.")
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    decoder_layers,
                    output_spec,
                    norm=norm,
                    act="ReLU",
                )
                for output_spec in output_specs
            ]
        )
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.in_channels = in_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return concatenated task-specific decoder projections."""
        bottleneck, skips = self.encoder(inputs)
        return torch.cat(
            [decoder(self.dropout(bottleneck), skips) for decoder in self.decoders],
            dim=1,
        )


class InstanSegModel:
    """Configuration-first façade for training, checkpointing, and evaluation."""

    def __init__(
        self,
        config_or_checkpoint: str | Path | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        self._config_path: Path | None = None
        self._checkpoint_path: Path | None = None
        self._checkpoint_config: Mapping[str, Any] | None = None
        self._constructor_overrides = dict(overrides)
        if isinstance(config_or_checkpoint, Mapping):
            self._constructor_overrides = {
                **dict(config_or_checkpoint),
                **self._constructor_overrides,
            }
        elif config_or_checkpoint is not None:
            supplied_path = Path(config_or_checkpoint).expanduser().resolve()
            if supplied_path.suffix.lower() in {".yaml", ".yml"}:
                self._config_path = supplied_path
            elif supplied_path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
                self._checkpoint_path = supplied_path
                adjacent_config = supplied_path.parent / "config.yaml"
                if adjacent_config.is_file():
                    self._config_path = adjacent_config
                checkpoint = load_checkpoint(supplied_path, torch.device("cpu"))
                stored_config = checkpoint.get("config")
                if isinstance(stored_config, Mapping):
                    self._checkpoint_config = stored_config
            else:
                raise ValueError(
                    "Expected a YAML configuration or PyTorch checkpoint path."
                )

        config_source = self._config_path or DEFAULT_CONFIG_PATH
        base_overrides = dict(self._checkpoint_config or {})
        if self._constructor_overrides:
            for key, value in self._constructor_overrides.items():
                if isinstance(value, Mapping) and isinstance(
                    base_overrides.get(key), Mapping
                ):
                    base_overrides[key] = {
                        **base_overrides[key],
                        **value,
                    }
                else:
                    base_overrides[key] = value
        self.config = load_config(config_source, base_overrides)
        validate_config(self.config)
        if self.config["preprocessing"]["dim_in"] != self.config["model"]["input_channels"]:
            raise ValueError(
                "preprocessing.dim_in must match model.input_channels."
            )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: nn.Module | None = None
        self.history: list[dict[str, Any]] = []
        self.logger: logging.Logger | None = None
        self._initial_checkpoint: dict[str, Any] | None = None
        if self._checkpoint_path is not None:
            self._initial_checkpoint = load_checkpoint(
                self._checkpoint_path, self.device
            )

    def _load_effective_config(
        self, overrides: Mapping[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Merge checkpoint, constructor, prior-run, and call-time settings."""
        effective_overrides: dict[str, Any] = dict(self._checkpoint_config or {})
        for update in (self.config, self._constructor_overrides, overrides):
            for key, value in update.items():
                if isinstance(value, Mapping) and isinstance(
                    effective_overrides.get(key), Mapping
                ):
                    effective_overrides[key] = {
                        **effective_overrides[key],
                        **value,
                    }
                else:
                    effective_overrides[key] = value
        config = load_config(
            self._config_path or DEFAULT_CONFIG_PATH, effective_overrides
        )
        validate_config(config)
        if config["preprocessing"]["dim_in"] != config["model"]["input_channels"]:
            raise ValueError(
                "preprocessing.dim_in must match model.input_channels."
            )
        return config

    def _build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        """Construct the registered model selected by configuration."""
        effective = config or self.config
        model_config = effective["model"]
        if model_config["model_name"] == "instanseg_unet":
            return InstanSeg_UNet(
                in_channels=model_config["input_channels"],
                out_channels=model_config.get("out_channels", (1,)),
                layers=model_config.get("layers", (256, 128, 64, 32)),
                dropout=model_config["dropout"],
            )
        if model_config["model_name"] == "semantic_unet":
            return SemanticSegmentationUNet(
                in_channels=model_config["input_channels"],
                channels=model_config["unet_channels"],
            )
        if model_config["model_name"] in {
            "dela_net",
            "discriminative_embedding_learned_assignment",
        }:
            return DiscriminativeEmbeddingLearnedAssignmentNet(
                in_channels=model_config["input_channels"],
                backbone_channels=model_config["backbone_channels"],
                num_slots=model_config["num_slots"],
                embedding_dim=model_config["embedding_dim"],
                assignment_dim=model_config["assignment_dim"],
                dropout=model_config["dropout"],
            )
        if model_config["model_name"] in {
            "affinity_guided_recurrent",
            "affinity_guided_recurrent_instance_segmentation",
            "affinity_guided_recurrent_net",
            "affinity_guided_recurrent_unet",
        }:
            return AffinityGuidedRecurrentNet(
                in_channels=model_config["input_channels"],
                backbone_channels=model_config["backbone_channels"],
                feature_channels=model_config["feature_channels"],
                num_slots=model_config["num_slots"],
                recurrent_hidden_channels=model_config["recurrent_hidden_channels"],
                recurrent_iterations=model_config["recurrent_iterations"],
                slot_iterations=model_config["slot_iterations"],
                dropout=model_config["dropout"],
                minimum_instance_area=model_config["minimum_instance_area"],
            )
        return AffinityGuidedMLPNet(
            in_channels=model_config["input_channels"],
            backbone_channels=model_config["backbone_channels"],
            feature_channels=model_config["feature_channels"],
            num_slots=model_config["num_slots"],
            mlp_channels=model_config["mlp_channels"],
            dropout=model_config["dropout"],
            use_aggregation=model_config["use_aggregation"],
        )

    def _prepare_data(
        self, config: Mapping[str, Any], *, training: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            from ..dataset.dataset import Segmentation_Dataset
            from ..dataset.utils import load_dataset_splits
        except ImportError:
            from dataset.dataset import Segmentation_Dataset
            from dataset.utils import load_dataset_splits

        data_config = {
            **config["meta"],
            **config["preprocessing"],
            **config["training"],
        }
        splits = load_dataset_splits(data_config)
        size = (config["preprocessing"]["imgsz"],) * 2
        datasets: dict[str, Any] = {}
        for split_name, split_data in splits.items():
            images, masks, metadata = split_data
            if not images:
                datasets[split_name] = None
                continue
            datasets[split_name] = Segmentation_Dataset(
                images,
                masks,
                common_transforms=True,
                metadata=metadata,
                size=size,
                dim_in=config["preprocessing"]["dim_in"],
                random_seed=config["preprocessing"]["seed"],
                mode="train" if training and split_name == "train" else "test",
                augmentation_type=config["preprocessing"]["augmentation_type"],
                channel_invariant=config["preprocessing"]["dim_in"] == 1,
                binarize_masks=config["model"]["model_name"] == "semantic_unet",
            )
        return splits, datasets

    def _make_loaders(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, Any],
        *,
        training: bool,
    ) -> tuple[dict[str, Any], dict[str, torch.Generator]]:
        from torch.utils.data import DataLoader, RandomSampler

        try:
            from ..dataset.dataset import collate_fn
            from ..dataset.utils import seed_worker
        except ImportError:
            from dataset.dataset import collate_fn
            from dataset.utils import seed_worker

        num_workers = config["training"]["num_workers"]
        batch_size = config["training"]["batch_size"]
        seed = config["preprocessing"]["seed"]
        loaders: dict[str, Any] = {}
        generators: dict[str, torch.Generator] = {}
        for index, (split, dataset) in enumerate(datasets.items()):
            if dataset is None:
                loaders[split] = None
                continue
            generator = torch.Generator()
            generator.manual_seed(seed + index)
            generators[split] = generator
            sample_count = None
            if split == "train" and training:
                sample_count = config["training"]["length_of_epoch"]
            if split == "test" and config["training"]["length_of_epoch"] is not None:
                sample_count = max(
                    1, int(config["training"]["length_of_epoch"] * 0.2)
                )
            sampler = RandomSampler(
                dataset, replacement=sample_count is not None,
                num_samples=sample_count, generator=generator,
            )
            loader_kwargs: dict[str, Any] = {
                "dataset": dataset,
                "batch_size": batch_size,
                "sampler": sampler,
                "generator": generator,
                "collate_fn": collate_fn,
                "num_workers": num_workers,
                "pin_memory": torch.cuda.is_available(),
                "persistent_workers": num_workers > 0,
                "worker_init_fn": seed_worker,
            }
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = 2
            loaders[split] = DataLoader(**loader_kwargs)
        return loaders, generators

    def train(self, **overrides: Any) -> list[dict[str, Any]]:
        """Run the full configured training experiment."""
        from .eval import InstanceSegmentationMetrics, eval_epoch
        from .loss import (
            AffinityGuidedLoss,
            DiscriminativeEmbeddingLearnedAssignmentLoss,
        )
        from .train import train_epoch

        config = self._load_effective_config(overrides)
        semantic_mode = config["model"]["model_name"] == "semantic_unet"
        dela_mode = config["model"]["model_name"] in {
            "dela_net",
            "discriminative_embedding_learned_assignment",
        }
        seed_everything(config["preprocessing"]["seed"])
        if config["meta"]["experiment_name"] is None:
            config["meta"]["experiment_name"] = generated_experiment_name()
        run_dir = PROJECT_ROOT / "runs" / config["meta"]["experiment_name"]
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = create_logger(run_dir / "train.log")
        self.logger.info("Experiment directory: %s", run_dir)
        self.logger.info("Effective configuration: %s", config)
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)

        started = time.monotonic()
        split_data, datasets = self._prepare_data(config, training=True)
        if not split_data["train"][0]:
            raise ValueError("The training split contains no image/mask pairs.")
        self.logger.info(
            "Data loaded: train=%d val=%d test=%d",
            len(split_data["train"][0]),
            len(split_data["val"][0]),
            len(split_data["test"][0]),
        )
        if dela_mode:
            instance_counts = [
                int(np.unique(mask[mask > 0]).size)
                for mask in split_data["train"][1]
            ]
            slot_capacity = config["model"]["num_slots"]
            over_capacity = sum(
                count > slot_capacity for count in instance_counts
            )
            self.logger.info(
                "DELA slot capacity K=%d; training instance counts "
                "p50=%.1f p99=%.1f max=%d; over-capacity images=%d/%d (%.4f)",
                slot_capacity,
                np.percentile(instance_counts, 50),
                np.percentile(instance_counts, 99),
                max(instance_counts),
                over_capacity,
                len(instance_counts),
                over_capacity / len(instance_counts),
            )
            if over_capacity:
                raise ValueError(
                    f"DELA slot capacity K={slot_capacity} is below the "
                    f"instance count in {over_capacity}/{len(instance_counts)} "
                    "training images. Increase num_slots or use a documented "
                    "tiling strategy; ground-truth instances are never truncated."
                )
        loaders, generators = self._make_loaders(
            config, datasets, training=True
        )
        self.logger.info("Data loaders and preprocessing initialized.")

        self.model = self._build_model(config).to(self.device)
        if semantic_mode:
            loss_fn = SemanticSegmentationLoss()
        elif dela_mode:
            loss_fn = DiscriminativeEmbeddingLearnedAssignmentLoss(
                matched_mask_weight=config["model"]["matched_mask_weight"]
            )
        else:
            loss_fn = AffinityGuidedLoss()
        base_lr = config["optimizer"]["learning_rate"]
        backbone = getattr(self.model, "backbone", None)
        backbone_parameters = list(backbone.parameters()) if backbone else []
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        other_parameters = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in backbone_ids
        ]
        parameter_groups = []
        if backbone_parameters:
            parameter_groups.append(
                {
                    "params": backbone_parameters,
                    "lr": base_lr * config["optimizer"]["backbone_lr_mult"],
                    "group_name": "backbone",
                }
            )
        if other_parameters:
            parameter_groups.append(
                {"params": other_parameters, "lr": base_lr, "group_name": "head"}
            )
        optimizer_config = config["optimizer"]
        if optimizer_config["type"] == "SGD":
            optimizer = torch.optim.SGD(
                parameter_groups,
                lr=base_lr,
                momentum=optimizer_config["momentum"],
                weight_decay=optimizer_config["weight_decay"],
            )
        else:
            optimizer = torch.optim.AdamW(
                parameter_groups,
                lr=base_lr,
                weight_decay=optimizer_config["weight_decay"],
            )
        amp_enabled = config["training"]["amp"] and self.device.type == "cuda"
        if config["training"]["amp"] and not amp_enabled:
            self.logger.info("AMP requested but CUDA is unavailable; using full precision.")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.logger.info(
            "Device=%s optimizer=%s learning_rate=%s AMP=%s",
            self.device,
            optimizer_config["type"],
            base_lr,
            amp_enabled,
        )

        history: list[dict[str, Any]] = []
        best_val_loss = float("inf")
        patience_count = 0
        start_epoch = 1
        checkpoint_path: Path | None = None
        resume_setting = config["training"]["resume"]
        if isinstance(resume_setting, str):
            checkpoint_path = Path(resume_setting).expanduser().resolve()
        elif resume_setting:
            checkpoint_path = checkpoint_dir / "last.pt"
        if checkpoint_path is None and self._initial_checkpoint is not None:
            model_state = self._initial_checkpoint.get(
                "model_state_dict", self._initial_checkpoint.get("state_dict")
            )
            if model_state is not None:
                self.model.load_state_dict(model_state)
            self.logger.info("Initialized model from %s", self._checkpoint_path)
        elif checkpoint_path is not None and checkpoint_path.is_file():
            checkpoint = load_checkpoint(checkpoint_path, self.device)
            model_state = checkpoint.get("model_state_dict")
            if model_state is None:
                raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
            self.model.load_state_dict(model_state)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            history = list(checkpoint.get("history", []))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            patience_count = int(checkpoint.get("patience_count", 0))
            if "generator_states" in checkpoint:
                for split, state in checkpoint["generator_states"].items():
                    if split in generators:
                        generators[split].set_state(state)
            if "rng_state" in checkpoint:
                restore_rng_state(checkpoint["rng_state"])
            self.logger.info(
                "Resumed from %s at completed epoch %d",
                checkpoint_path,
                start_epoch - 1,
            )
        elif checkpoint_path is not None:
            self.logger.info(
                "Resume checkpoint not found at %s; starting from scratch.",
                checkpoint_path,
            )

        metric_evaluator = InstanceSegmentationMetrics()
        total_epochs = config["training"]["epochs"]
        val_interval = config["training"]["val_interval"]
        patience = config["training"]["patience"]
        for epoch in range(start_epoch, total_epochs + 1):
            self.logger.info("EPOCH %d/%d", epoch, total_epochs)
            warmup_epochs = config["training"]["warmup_epochs"]
            if warmup_epochs and epoch <= warmup_epochs:
                multiplier = epoch / warmup_epochs
                for group in optimizer.param_groups:
                    group["lr"] = (
                        base_lr
                        * multiplier
                        * (
                            config["optimizer"]["backbone_lr_mult"]
                            if group.get("group_name") == "backbone"
                            else 1.0
                        )
                    )
                self.logger.info("Warmup learning-rate multiplier: %.4f", multiplier)

            frozen = epoch <= config["model"]["freeze_backbone_epochs"]
            if backbone is not None:
                for parameter in backbone.parameters():
                    parameter.requires_grad_(not frozen)

            train_loss, train_time = train_epoch(
                self.model,
                self.device,
                loaders["train"],
                loss_fn,
                optimizer,
                scaler=scaler,
                amp=amp_enabled,
            )
            component_values = getattr(loss_fn, "epoch_components", {})
            if component_values:
                self.logger.info("Training loss components: %s", component_values)
            validate = (
                epoch % val_interval == 0 or epoch == total_epochs
            ) and loaders.get("val") is not None
            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_time": train_time,
                "val_loss": None,
            }
            improved = False
            if validate:
                self.logger.info("Validation starting for epoch %d", epoch)
                if semantic_mode:
                    validation = eval_semantic_epoch(
                        self.model,
                        self.device,
                        loaders["val"],
                        loss_fn,
                        amp=amp_enabled,
                        progress_description="Validation",
                    )
                else:
                    metric_evaluator.reset()
                    validation = eval_epoch(
                        self.model,
                        self.device,
                        loaders["val"],
                        loss_fn,
                        metric_evaluator,
                        amp=amp_enabled,
                        progress_description="Validation",
                    )
                record.update(validation)
                record["val_loss"] = validation["loss"]
                self.logger.info("Validation metrics: %s", validation)
                if validation["loss"] < best_val_loss:
                    best_val_loss = validation["loss"]
                    patience_count = 0
                    improved = True
                else:
                    patience_count += 1
            history.append(record)
            write_history(run_dir / "metrics.csv", history)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_count": patience_count,
                "history": history,
                "rng_state": rng_state(),
                "generator_states": {
                    split: generator.get_state()
                    for split, generator in generators.items()
                },
                "config": config,
                "dataset_identity": str(Path(config["meta"]["data_dir"]).resolve()),
            }
            for path in (checkpoint_dir / "last.pt", run_dir / "last.pt"):
                save_checkpoint(path, checkpoint)
            self.logger.info(
                "Epoch %d train_loss=%.6f val_loss=%s",
                epoch,
                train_loss,
                record["val_loss"] if validate else "N/A",
            )
            if improved:
                for path in (checkpoint_dir / "best.pt", run_dir / "best.pt"):
                    save_checkpoint(path, checkpoint)
                self.logger.info("Best checkpoint updated: %s", checkpoint_dir / "best.pt")
            if validate and not improved and patience_count >= patience:
                self.logger.info("Early stopping at epoch %d", epoch)
                break

        best_path = checkpoint_dir / "best.pt"
        last_path = checkpoint_dir / "last.pt"
        if last_path.is_file() and best_val_loss == float("inf"):
            last_checkpoint = load_checkpoint(last_path, self.device)
            for path in (best_path, run_dir / "best.pt"):
                save_checkpoint(path, last_checkpoint)
        selected_checkpoint = best_path if best_path.is_file() else last_path
        if selected_checkpoint.is_file():
            final_checkpoint = load_checkpoint(selected_checkpoint, self.device)
            self.model.load_state_dict(final_checkpoint["model_state_dict"])
        if loaders.get("test") is not None:
            self.logger.info("Final test evaluation starting.")
            if semantic_mode:
                test_results = eval_semantic_epoch(
                    self.model,
                    self.device,
                    loaders["test"],
                    loss_fn,
                    amp=amp_enabled,
                    progress_description="Test",
                )
            else:
                metric_evaluator.reset()
                test_results = eval_epoch(
                    self.model,
                    self.device,
                    loaders["test"],
                    loss_fn,
                    metric_evaluator,
                    amp=amp_enabled,
                    progress_description="Test",
                )
            self.logger.info("Final test results: %s", test_results)
        self.history = history
        self.config = config
        self.logger.info(
            "Training complete in %.2fs; artifacts saved under %s",
            time.monotonic() - started,
            run_dir,
        )
        return history

    def eval(self, subset: str = "test", **overrides: Any) -> dict[str, float]:
        """Evaluate one supported split without modifying training artifacts."""
        from .eval import InstanceSegmentationMetrics, eval_epoch
        from .loss import (
            AffinityGuidedLoss,
            DiscriminativeEmbeddingLearnedAssignmentLoss,
        )

        if subset not in {"val", "test"}:
            raise ValueError("subset must be either 'val' or 'test'.")
        config = self._load_effective_config(overrides)
        splits, datasets = self._prepare_data(config, training=False)
        if not splits[subset][0]:
            raise ValueError(f"The requested {subset!r} split is empty.")
        loaders, _ = self._make_loaders(config, datasets, training=False)
        if self.model is None:
            self.model = self._build_model(config).to(self.device)
            checkpoint = self._initial_checkpoint
            if checkpoint is None and self._checkpoint_path is not None:
                checkpoint = load_checkpoint(self._checkpoint_path, self.device)
            if checkpoint is not None:
                state = checkpoint.get("model_state_dict")
                if state is None:
                    raise ValueError("Checkpoint has no model_state_dict.")
                self.model.load_state_dict(state)
        self.model.eval()
        amp_enabled = config["training"]["amp"] and self.device.type == "cuda"
        if config["model"]["model_name"] == "semantic_unet":
            results = eval_semantic_epoch(
                self.model,
                self.device,
                loaders[subset],
                SemanticSegmentationLoss(),
                amp=amp_enabled,
                progress_description=subset.capitalize(),
            )
        else:
            dela_mode = config["model"]["model_name"] in {
                "dela_net",
                "discriminative_embedding_learned_assignment",
            }
            results = eval_epoch(
                self.model,
                self.device,
                loaders[subset],
                (
                    DiscriminativeEmbeddingLearnedAssignmentLoss(
                        matched_mask_weight=config["model"]["matched_mask_weight"]
                    )
                    if dela_mode
                    else AffinityGuidedLoss()
                ),
                InstanceSegmentationMetrics(),
                amp=amp_enabled,
                progress_description=subset.capitalize(),
            )
        return results


ModelFacade = InstanSegModel
