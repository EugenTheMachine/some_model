"""Reusable epoch-level training operations."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        images, labels, _ = batch
    elif len(batch) == 2:
        images, labels = batch
    else:
        raise ValueError("Training batch must contain images, labels, and optional lengths.")
    return images, labels


def _loss_tensor(
    loss_fn: Callable[[Any, torch.Tensor], Any],
    outputs: Any,
    labels: torch.Tensor,
) -> torch.Tensor:
    result = loss_fn(outputs, labels)
    if isinstance(result, Mapping):
        if "loss" not in result:
            raise ValueError("A loss mapping must contain a 'loss' tensor.")
        result = result["loss"]
    if not isinstance(result, torch.Tensor) or result.ndim != 0:
        raise TypeError("loss_fn must return a scalar tensor.")
    return result


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: Any,
    loss_fn: Callable[[Any, torch.Tensor], Any],
    optimizer: torch.optim.Optimizer,
    *,
    scaler: Any = None,
    amp: bool = False,
    accumulation_steps: int = 1,
    gradient_clip_norm: float = 20.0,
) -> tuple[float, float]:
    """Train for one epoch and return mean batch loss and elapsed seconds."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive.")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive.")
    if train_loader is None or len(train_loader) == 0:
        raise ValueError("Training loader is empty.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    started = time.monotonic()
    total_batches = len(train_loader)
    component_history: list[dict[str, float]] = []
    for batch_index, batch in enumerate(
        tqdm(train_loader, desc="Training", leave=False), start=1
    ):
        images, labels = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model(images)
            loss = _loss_tensor(loss_fn, outputs, labels)
        components = getattr(loss_fn, "last_components", None)
        if isinstance(components, Mapping):
            component_history.append(
                {key: float(value) for key, value in components.items()}
            )
        losses.append(float(loss.detach()))
        group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total_batches - group_start)
        scaled_loss = loss / group_size
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        update = (
            batch_index % accumulation_steps == 0
            or batch_index == total_batches
        )
        if update:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if component_history and hasattr(loss_fn, "epoch_components"):
        keys = component_history[0]
        loss_fn.epoch_components = {
            key: float(np.mean([components[key] for components in component_history]))
            for key in keys
        }
    return float(np.mean(losses)) if losses else 0.0, time.monotonic() - started
