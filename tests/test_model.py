"""Focused tests for the affinity-guided model and its data contract."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset.dataset import Segmentation_Dataset, collate_fn
from src.model.head import AffinityGuidedAggregation
from src.model.loss import AffinityGuidedLoss, build_affinity_targets
from src.model.model import AffinityGuidedMLPNet, InstanSegModel
from src.model.utils import load_config


def test_affinity_targets_follow_right_down_diagonal_offsets():
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
    assert targets[0, 1, 0, 2] == 1
    assert targets[0, 3, 0, 2] == 1
    assert not valid[0, 2, 1, 1]


def test_aggregation_uses_reverse_edge_affinity_at_neighbor():
    features = torch.tensor([[[[2.0, 6.0, 10.0]]]])
    affinity_logits = torch.full((1, 4, 1, 3), -20.0)
    affinity_logits[0, 0, 0, 0] = 20.0

    aggregated = AffinityGuidedAggregation()(features, affinity_logits)

    assert aggregated[0, 0, 0, 0].item() == pytest.approx(6.0, abs=1e-3)
    assert aggregated[0, 0, 0, 1].item() == pytest.approx(2.0, abs=1e-3)


def test_model_forward_and_assignment_loss_backpropagate_to_affinity_head():
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


def test_dataset_preloads_and_collates_padded_instance_masks():
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


def test_configuration_rejects_unsupported_optimizer():
    with pytest.raises(ValueError, match="optimizer.type"):
        load_config(overrides={"optimizer_type": "RMSprop"})


def test_facade_trains_saves_and_evaluates(tmp_path, monkeypatch):
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
    result = facade.eval(subset="test")
    artifact_dir = tmp_path / "runs" / "integration"

    assert len(history) == 1
    assert (artifact_dir / "config.yaml").is_file()
    assert (artifact_dir / "metrics.csv").is_file()
    assert (artifact_dir / "checkpoints" / "last.pt").is_file()
    assert (artifact_dir / "checkpoints" / "best.pt").is_file()
    assert {"loss", "precision", "recall", "accuracy", "f1"} <= result.keys()
