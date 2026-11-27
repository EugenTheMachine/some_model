"""Focused tests for the affinity-guided model and its data contract."""

import json
import numpy as np
import pytest
import torch
from PIL import Image

import kaggle_train
from src.dataset.augmentations import Augmentations
from src.dataset.dataset import Segmentation_Dataset, collate_fn
from src.dataset.utils import load_dataset_splits
from src.model.head import AffinityGuidedAggregation, RecurrentSlotGrouping
from src.model.eval import InstanceSegmentationMetrics
from src.model.loss import (
    AffinityGuidedLoss,
    build_affinity_targets,
    build_assignment_targets,
)
from src.model.model import (
    AffinityGuidedMLPNet,
    AffinityGuidedRecurrentNet,
    InstanSegModel,
)
from src.model.semantic import SemanticSegmentationLoss, SemanticSegmentationUNet
from src.model.train import train_epoch
from src.model.utils import load_config


def test_affinity_targets_follow_right_down_diagonal_offsets():
    """Affinity channels match the declared directional neighbor offsets."""
    labels = torch.tensor(
        [[[1, 1, 2], [0, 2, 2]]],
        dtype=torch.int32,
    )

    targets, valid = build_affinity_targets(labels)

    assert targets.shape == (1, 4, 2, 3)
    assert targets[0, 0, 0, 0] == 1
    assert valid[0, 0, 0, 1]
    assert targets[0, 0, 0, 1] == 0
    assert not valid[0, 0, 0, 2]
    assert valid[0, 1, 0, 0]
    assert targets[0, 1, 0, 0] == 0
    assert targets[0, 1, 0, 2] == 1
    assert targets[0, 3, 0, 2] == 1
    assert not valid[0, 2, 1, 1]


def test_affinity_targets_ignore_pairs_touching_ignored_pixels():
    labels = torch.tensor([[[1, -1, 0], [1, 0, 0]]], dtype=torch.int32)

    targets, valid = build_affinity_targets(labels)

    assert not valid[0, 0, 0, 0]
    assert not valid[0, 1, 0, 1]
    assert valid[0, 0, 1, 1]
    assert targets[0, 0, 1, 1] == 0
    assert valid[0, 0, 1, 0]
    assert targets[0, 0, 1, 0] == 0


def test_aggregation_uses_reverse_edge_affinity_at_neighbor():
    """A right edge affinity weights both directed aggregation lookups."""
    features = torch.tensor([[[[2.0, 6.0, 10.0]]]])
    affinity_logits = torch.full((1, 4, 1, 3), -20.0)
    affinity_logits[0, 0, 0, 0] = 20.0

    foreground_probs = torch.ones((1, 1, 1, 3))
    aggregated = AffinityGuidedAggregation()(
        features, affinity_logits, foreground_probs
    )

    assert aggregated[0, 0, 0, 0].item() == pytest.approx(6.0, abs=1e-3)
    assert aggregated[0, 0, 0, 1].item() == pytest.approx(2.0, abs=1e-3)


def test_aggregation_gates_edges_by_both_endpoint_foreground_probs():
    features = torch.tensor([[[[2.0, 6.0, 10.0]]]])
    affinity_logits = torch.full((1, 4, 1, 3), -20.0)
    affinity_logits[0, 0, 0, 0] = 20.0
    foreground_probs = torch.tensor([[[[1.0, 0.0, 1.0]]]])

    aggregated = AffinityGuidedAggregation()(
        features, affinity_logits, foreground_probs
    )

    assert torch.count_nonzero(aggregated) == 0


def test_model_forward_and_assignment_loss_backpropagate_to_affinity_head():
    """Prediction tensors have expected shapes and assignment trains affinities."""
    model = AffinityGuidedMLPNet(
        in_channels=1,
        backbone_channels=(4, 8, 12, 16, 20),
        feature_channels=4,
        num_slots=3,
        mlp_channels=(8, 4),
    )
    images = torch.randn(2, 1, 32, 32)
    labels = torch.zeros(2, 1, 32, 32, dtype=torch.int32)
    labels[:, :, 4:14, 4:14] = 4
    labels[:, :, 18:28, 18:28] = 9

    outputs = model(images)
    loss = AffinityGuidedLoss()(outputs, labels)
    loss.backward()

    assert outputs["foreground_logits"].shape == (2, 1, 32, 32)
    assert outputs["affinity_logits"].shape == (2, 4, 32, 32)
    assert outputs["assignment_logits"].shape == (2, 4, 32, 32)
    affinity_gradient = model.head.affinity_head[-1].weight.grad
    assert affinity_gradient is not None
    assert torch.count_nonzero(affinity_gradient) > 0


def test_recurrent_model_forward_and_loss_backpropagate():
    """Recurrent grouping and slot updates produce dense trainable assignments."""
    model = AffinityGuidedRecurrentNet(
        in_channels=1,
        backbone_channels=(4, 8, 12, 16, 20),
        feature_channels=4,
        num_slots=3,
        recurrent_hidden_channels=4,
        recurrent_iterations=2,
        slot_iterations=2,
    )
    images = torch.randn(2, 1, 32, 32)
    labels = torch.zeros(2, 1, 32, 32, dtype=torch.int32)
    labels[:, :, 4:14, 4:14] = 4
    labels[:, :, 18:28, 18:28] = 9

    outputs = model(images)
    loss = AffinityGuidedLoss()(outputs, labels)
    loss.backward()

    assert outputs["recurrent_features"].shape == (2, 4, 32, 32)
    assert outputs["slot_embeddings"].shape == (2, 3, 4)
    assert outputs["assignment_logits"].shape == (2, 4, 32, 32)
    assert outputs["instance_map"].shape == (2, 32, 32)
    assert model.head.affinity_head[-1].weight.grad is not None


def test_slot_updates_remain_finite_for_dominant_slot_under_amp():
    """Normalize pixel weights before FP16 aggregation to avoid overflow."""
    grouping = RecurrentSlotGrouping(
        pixel_channels=1, num_slots=2, slot_iterations=1
    )
    with torch.no_grad():
        grouping.query.weight.fill_(1)
        grouping.query.bias.zero_()
        grouping.key.weight.fill_(1)
        grouping.key.bias.zero_()
        grouping.value.weight.fill_(1)
        grouping.value.bias.zero_()
        grouping.slot_embeddings[0, 0, 0] = 100
        grouping.slot_embeddings[0, 1, 0] = -100

    updates: list[torch.Tensor] = []
    hook = grouping.slot_gru.register_forward_pre_hook(
        lambda _module, args: updates.append(args[0].detach())
    )
    features = torch.ones(1, 1, 256, 256)
    foreground = torch.ones(1, 1, 256, 256)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            outputs = grouping(features, foreground)
    finally:
        hook.remove()

    assert len(updates) == 1
    assert torch.isfinite(updates[0]).all()
    assert torch.isfinite(outputs["assignment_probs"]).all()


def test_hungarian_matching_is_finite_for_saturated_half_probabilities():
    """Mixed-precision softmax outputs remain safe for matching costs."""
    logits = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[20.0, 20.0], [-20.0, -20.0]],
                [[-20.0, -20.0], [0.0, 20.0]],
            ]
        ]
    )
    probabilities = logits.softmax(dim=1).to(torch.float16)
    labels = torch.tensor([[[1, 1], [0, 2]]], dtype=torch.int32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        targets, matchings = build_assignment_targets(probabilities, labels)

    assert len(matchings) == 1
    assert len(matchings[0][0]) == 2
    assert torch.unique(targets[0][labels[0] > 0]).numel() == 2


def test_dataset_preloads_and_collates_padded_instance_masks():
    """Dataset access returns normalized tensors and contiguous instance IDs."""
    image = np.arange(24 * 24, dtype=np.uint8).reshape(24, 24)
    mask = np.zeros((24, 24), dtype=np.int32)
    mask[5:12, 5:12] = 8
    dataset = Segmentation_Dataset(
        [image],
        [mask],
        metadata=[{"image_modality": "phase-contrast"}],
        size=(32, 32),
        dim_in=1,
        channel_invariant=True,
        random_seed=42,
        mode="test",
    )

    batch_images, batch_masks, lengths = collate_fn([dataset[0]])

    assert batch_images.shape == (1, 1, 32, 32)
    assert batch_masks.shape == (1, 1, 32, 32)
    assert batch_masks.dtype == torch.int32
    assert lengths.dtype == torch.int32
    assert set(torch.unique(batch_masks).tolist()) <= {0, 1}


def test_semantic_dataset_binarizes_every_nonzero_mask_value():
    """Semantic preprocessing maps all nonzero instance IDs to foreground."""
    image = np.zeros((16, 16), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.int32)
    mask[2:6, 2:6] = 8
    mask[8:12, 8:12] = -1
    dataset = Segmentation_Dataset(
        [image],
        [mask],
        common_transforms=False,
        size=(16, 16),
        dim_in=1,
        mode="test",
        binarize_masks=True,
    )

    _, binary_mask = dataset[0]

    assert set(torch.unique(binary_mask).tolist()) <= {0, 1}
    assert torch.all(binary_mask[0, 2:6, 2:6] == 1)
    assert torch.all(binary_mask[0, 8:12, 8:12] == 1)


def test_semantic_unet_emits_binary_foreground_logits_and_backpropagates():
    """Classical U-Net output matches binary masks and supports optimization."""
    model = SemanticSegmentationUNet(in_channels=1, channels=(4, 8, 16))
    images = torch.randn(2, 1, 32, 32)
    masks = torch.zeros(2, 1, 32, 32, dtype=torch.int32)
    masks[:, :, 8:24, 8:24] = 1

    logits = model(images)
    loss = SemanticSegmentationLoss()(logits, masks)
    loss.backward()

    assert logits.shape == (2, 1, 32, 32)
    assert torch.isfinite(loss)
    assert model.output.weight.grad is not None


def test_configuration_rejects_unsupported_optimizer():
    """Unknown optimizer names fail configuration validation."""
    with pytest.raises(ValueError, match="optimizer.type"):
        load_config(overrides={"optimizer_type": "RMSprop"})


def test_training_accumulation_updates_final_partial_group():
    """A final incomplete accumulation group uses its actual group size."""
    model = torch.nn.Conv2d(1, 1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    batches = [
        (torch.ones(1, 1, 1), torch.ones(1, 1, 1))
        for _ in range(5)
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    mean_loss, _ = train_epoch(
        model,
        torch.device("cpu"),
        torch.utils.data.DataLoader(batches, batch_size=1),
        lambda outputs, targets: ((outputs - targets) ** 2).mean(),
        optimizer,
        accumulation_steps=2,
    )

    assert mean_loss == pytest.approx(1.0)
    assert model.weight.item() == pytest.approx(2.0)


def test_object_metrics_are_perfect_for_an_exact_instance_map():
    """Object matching, AP, AJI, PQ, and boundary metrics use instance IDs."""
    labels = torch.zeros(1, 1, 12, 12, dtype=torch.int32)
    labels[:, :, 2:8, 3:9] = 1
    affinity_targets, _ = build_affinity_targets(labels)
    outputs = {
        "instance_map": labels[:, 0],
        "assignment_probs": torch.full((1, 2, 12, 12), 0.01),
        "affinity_logits": (affinity_targets * 2 - 1) * 20,
    }
    outputs["assignment_probs"][:, 0] = 0.99
    outputs["assignment_probs"][:, 0, 2:8, 3:9] = 0.01
    outputs["assignment_probs"][:, 1, 2:8, 3:9] = 0.99
    metrics = InstanceSegmentationMetrics()
    metrics.update(outputs, labels)
    result = metrics.compute()

    for metric in ("precision", "recall", "f1", "ap50", "ap75", "aji", "pq", "boundary_f1"):
        assert result[metric] == pytest.approx(1.0, abs=1e-6)


def test_optional_paired_augmentations_preserve_label_geometry():
    """Perspective, Kornia, and stain processing preserve image tensor layout."""
    image = torch.rand(3, 32, 32)
    labels = torch.zeros(1, 32, 32, dtype=torch.int32)
    labels[:, 8:24, 8:24] = 4
    augmenter = Augmentations(shape=(32, 32), dim_in=3, random_seed=42)

    perspective_image, perspective_labels = augmenter.perspective(
        image, labels, amount=0.2
    )
    kornia_image, _ = augmenter.kornia_base_augmentations(image, labels)
    stain_image, _ = augmenter.normalize_HE_stains(
        image * 255,
        labels,
        amount=0,
        metadata={"image_modality": "Brightfield"},
    )

    assert perspective_image.shape == image.shape
    assert perspective_labels.shape == labels.shape
    assert set(torch.unique(perspective_labels).tolist()) <= {0, 4}
    assert kornia_image.shape == image.shape
    assert stain_image.shape == image.shape
    assert torch.isfinite(stain_image).all()


def test_pth_reader_requires_all_declared_splits(tmp_path, monkeypatch):
    """The serialized dataset reader preloads explicit train/val/test items."""
    image = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.int32)
    sample = {
        "image": image,
        "cell_masks": mask,
        "image_modality": "phase-contrast",
    }
    dataset_path = tmp_path / "cells.pth"
    monkeypatch.setenv("INSTANSEG_DATASET_PATH", str(tmp_path))
    torch.save({"train": [sample], "val": [sample], "test": [sample]}, dataset_path)

    splits = load_dataset_splits(
        {
            "data_dir": str(dataset_path),
            "seed": 42,
            "train_data_ratio": 1.0,
            "source_dataset": ["all"],
        }
    )

    assert all(len(splits[split][0]) == 1 for split in ("train", "val", "test"))
    assert splits["train"][0][0].shape == (8, 8)


def test_facade_trains_saves_and_evaluates(tmp_path, monkeypatch):
    """A short run writes artifacts and exposes the public evaluation API."""
    import src.model.model as model_module

    data_dir = tmp_path / "data"
    image_dir = data_dir / "train" / "images"
    mask_dir = data_dir / "train" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(2):
        image = np.full((32, 32), 40 + index * 20, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[6:20, 6:20] = 3
        Image.fromarray(image).save(image_dir / f"cell_{index}.png")
        Image.fromarray(mask).save(mask_dir / f"cell_{index}.png")
    monkeypatch.setattr(model_module, "PROJECT_ROOT", tmp_path)

    facade = InstanSegModel(
        data_dir=str(data_dir),
        experiment_name="integration",
        imgsz=32,
        batch_size=1,
        epochs=1,
        warmup_epochs=0,
        num_workers=0,
        input_channels=1,
        dim_in=1,
        backbone_channels=[4, 8, 12, 16, 20],
        feature_channels=4,
        num_slots=3,
        mlp_channels=[8, 4],
    )
    history = facade.train()
    resumed_history = facade.train(epochs=2, resume=True)
    result = facade.eval(subset="test")
    artifact_dir = tmp_path / "runs" / "integration"

    assert len(history) == 1
    assert [record["epoch"] for record in resumed_history] == [1, 2]
    assert (artifact_dir / "config.yaml").is_file()
    assert (artifact_dir / "metrics.csv").is_file()
    assert (artifact_dir / "checkpoints" / "last.pt").is_file()
    assert (artifact_dir / "checkpoints" / "best.pt").is_file()
    assert {"loss", "precision", "recall", "accuracy", "f1"} <= result.keys()

    semantic_facade = InstanSegModel(
        data_dir=str(data_dir),
        experiment_name="semantic-integration",
        imgsz=32,
        batch_size=1,
        epochs=1,
        warmup_epochs=0,
        num_workers=0,
        input_channels=1,
        dim_in=1,
        model_name="semantic_unet",
        unet_channels=[4, 8, 16],
    )
    semantic_history = semantic_facade.train()
    semantic_result = semantic_facade.eval(subset="test")

    assert len(semantic_history) == 1
    assert {"loss", "pixel_accuracy", "dice", "iou"} <= semantic_result.keys()


def test_kaggle_staging_copies_only_notebook_and_metadata(tmp_path):
    """Local staging validates notebook code and never copies credentials."""
    notebook_path = tmp_path / "training_kernel.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["print('training')"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "source-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "sample-owner/sample-model",
                "dataset_sources": ["owner/dataset"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "kaggle_token.txt").write_text("not copied", encoding="utf-8")
    upload_dir = tmp_path / "upload"

    staged_metadata = kaggle_train.stage_kernel(
        upload_dir, notebook_path, metadata_path
    )

    assert sorted(path.name for path in upload_dir.iterdir()) == [
        "kernel-metadata.json",
        "training_kernel.ipynb",
    ]
    assert json.loads(staged_metadata.read_text(encoding="utf-8"))["code_file"] == (
        "training_kernel.ipynb"
    )


def test_kaggle_staging_rejects_non_notebook_json(tmp_path):
    """Invalid nbformat inputs fail before any upload is attempted."""
    notebook_path = tmp_path / "training_kernel.ipynb"
    notebook_path.write_text("{}", encoding="utf-8")
    metadata_path = tmp_path / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps({"id": "owner/kernel", "dataset_sources": ["owner/data"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nbformat 4"):
        kaggle_train.stage_kernel(tmp_path / "stage", notebook_path, metadata_path)


def test_kaggle_refuses_existing_kernel(monkeypatch):
    """A configured kernel slug must not be pushed again as a new version."""
    csv_output = "ref,title\nowner/new-kernel,New kernel\n"
    monkeypatch.setattr(
        kaggle_train,
        "run_kaggle_command",
        lambda *_args, **_kwargs: type(
            "Result",
            (),
            {"stdout": csv_output},
        )(),
    )

    with pytest.raises(RuntimeError, match="Refusing to add a version"):
        kaggle_train.ensure_kernel_is_new({}, "owner/new-kernel")


def test_kaggle_allows_absent_kernel(monkeypatch):
    """An unused kernel slug passes the pre-push existence check."""
    monkeypatch.setattr(
        kaggle_train,
        "run_kaggle_command",
        lambda *_args, **_kwargs: type(
            "Result",
            (),
            {"stdout": "ref,title\n"},
        )(),
    )

    kaggle_train.ensure_kernel_is_new({}, "owner/new-kernel")


def test_kaggle_notebook_validation_rejects_invalid_python(tmp_path):
    """Notebook code cells are syntax-checked before staging."""
    notebook_path = tmp_path / "training_kernel.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["def broken(:\n", "    pass\n"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid Python in notebook cell"):
        kaggle_train.validate_notebook(notebook_path)


def test_kaggle_status_404_is_reported_as_missing_session(monkeypatch):
    """A missing session is distinguishable from other Kaggle CLI errors."""
    def raise_not_found(*_args, **_kwargs):
        raise RuntimeError(
            "Kaggle command failed with exit code 1: "
            "404 Client Error: Not Found for GetKernelSessionStatus"
        )

    monkeypatch.setattr(kaggle_train, "run_kaggle_command", raise_not_found)

    assert kaggle_train.get_kernel_status({}, "owner/kernel") is None


def test_kaggle_monitor_retries_404_after_push(monkeypatch):
    """A newly pushed kernel can start after initial status checks return 404."""
    statuses = iter([None, "RUNNING", "COMPLETE"])
    monkeypatch.setattr(
        kaggle_train,
        "get_kernel_status",
        lambda *_args: next(statuses),
    )
    monkeypatch.setattr(kaggle_train.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        kaggle_train, "fetch_and_display_logs", lambda *_args: None
    )

    assert kaggle_train.monitor_kernel(
        {},
        "owner/kernel",
        poll_interval=1,
        timeout=30,
        wait_for_startup=True,
    )


def test_kaggle_monitor_explains_404_without_waiting_for_a_push(capsys, monkeypatch):
    """Monitoring without a push reports absent sessions immediately."""
    monkeypatch.setattr(
        kaggle_train, "get_kernel_status", lambda *_args: None
    )

    assert not kaggle_train.monitor_kernel({}, "owner/kernel")
    assert "Check the owner/kernel slug or push a kernel version first" in (
        capsys.readouterr().err
    )
