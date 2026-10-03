import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


History = Dict[str, Any]
DeviceLike = Union[str, torch.device]


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Calculate MAE, RMSE, and R2 for regression predictions."""
    predictions, targets = _validate_prediction_target_tensors(predictions, targets)

    errors = predictions - targets
    absolute_error = torch.abs(errors).mean()
    squared_error = torch.square(errors).mean()
    target_mean = targets.mean()
    total_sum_squares = torch.sum(torch.square(targets - target_mean))
    residual_sum_squares = torch.sum(torch.square(errors))

    if total_sum_squares.item() == 0.0:
        r2 = 1.0 if residual_sum_squares.item() == 0.0 else 0.0
    else:
        r2 = 1.0 - (residual_sum_squares / total_sum_squares).item()

    metrics = {
        'mae': absolute_error.item(),
        'rmse': torch.sqrt(squared_error).item(),
        'r2': r2,
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError('Regression metrics contain NaN or infinite values.')
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    device: Optional[DeviceLike] = None,
    patience: Optional[int] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> History:
    """Train a regression model and return per-epoch training history.

    The model is restored to its best validation-MAE state before returning.
    If ``checkpoint_path`` is provided, the best state and training metadata are
    saved there whenever a new best validation MAE is found. For deterministic
    model initialization, call ``set_seed`` before constructing the model;
    ``seed`` here controls training-time random operations.
    """
    _validate_training_arguments(learning_rate, epochs, patience)
    if seed is not None:
        set_seed(seed)

    resolved_device = torch.device(device or 'cpu')
    model.to(resolved_device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: History = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': [],
        'epochs_completed': 0,
        'best_epoch': None,
        'best_val_mae': None,
        'stopped_early': False,
        'best_state_dict': None,
    }
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_optimizer_state: Optional[Dict[str, Any]] = None
    best_val_mae = float('inf')
    best_epoch: Optional[int] = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = _run_training_epoch(
            model, train_loader, loss_function, optimizer, resolved_device
        )
        validation_loss, validation_metrics = _run_validation_epoch(
            model, val_loader, loss_function, resolved_device
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(validation_loss)
        history['val_mae'].append(validation_metrics['mae'])
        history['val_rmse'].append(validation_metrics['rmse'])
        history['val_r2'].append(validation_metrics['r2'])
        history['epochs_completed'] = epoch

        if validation_metrics['mae'] < best_val_mae:
            best_val_mae = validation_metrics['mae']
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = _copy_state_dict(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                history=history,
                best_state_dict=best_state_dict,
                best_optimizer_state=best_optimizer_state,
                best_epoch=best_epoch,
                best_val_mae=best_val_mae,
                learning_rate=learning_rate,
                epochs=epochs,
                patience=patience,
                device=resolved_device,
                seed=seed,
                checkpoint_metadata=checkpoint_metadata,
            )
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                history['stopped_early'] = True
                break

    if best_state_dict is None or best_epoch is None:
        raise ValueError('Training did not produce a valid best model state.')

    model.load_state_dict(best_state_dict)
    history['best_epoch'] = best_epoch
    history['best_val_mae'] = best_val_mae
    history['best_state_dict'] = best_state_dict
    return history


def _validate_training_arguments(
    learning_rate: float,
    epochs: int,
    patience: Optional[int],
) -> None:
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError('learning_rate must be a finite value greater than zero.')
    if epochs <= 0:
        raise ValueError('epochs must be greater than zero.')
    if patience is not None and patience < 0:
        raise ValueError('patience must be non-negative or None.')


def _run_training_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0

    for features, targets in data_loader:
        features, targets = _prepare_batch(features, targets, device)
        optimizer.zero_grad()
        predictions = model(features)
        _ensure_finite_tensor(predictions, 'model predictions')
        predictions, targets = _validate_prediction_target_tensors(predictions, targets)
        loss = loss_function(predictions, targets)
        _ensure_finite_tensor(loss, 'training loss')
        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError('Training DataLoader is empty.')
    return total_loss / sample_count


def _run_validation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    prediction_batches: List[torch.Tensor] = []
    target_batches: List[torch.Tensor] = []

    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = _prepare_batch(features, targets, device)
            predictions = model(features)
            _ensure_finite_tensor(predictions, 'model predictions')
            predictions, targets = _validate_prediction_target_tensors(predictions, targets)
            loss = loss_function(predictions, targets)
            _ensure_finite_tensor(loss, 'validation loss')

            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            prediction_batches.append(predictions.detach())
            target_batches.append(targets.detach())

    if sample_count == 0:
        raise ValueError('Validation DataLoader is empty.')

    predictions = torch.cat(prediction_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    metrics = calculate_regression_metrics(predictions, targets)
    return total_loss / sample_count, metrics


def _prepare_batch(
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(features, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError('DataLoader batches must contain PyTorch tensors.')
    _ensure_finite_tensor(features, 'input features')
    _ensure_finite_tensor(targets, 'targets')

    features = features.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.float32)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    return features, targets


def _validate_prediction_target_tensors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    if predictions.shape != targets.shape:
        raise ValueError(
            f'Prediction shape {tuple(predictions.shape)} does not match '
            f'target shape {tuple(targets.shape)}.'
        )
    return predictions, targets


def _ensure_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f'{name} contain NaN or infinite values.')


def _copy_state_dict(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state_dict.items()
        if torch.is_tensor(value)
    }


def _save_checkpoint(
    checkpoint_path: Optional[Union[str, Path]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: History,
    best_state_dict: Dict[str, torch.Tensor],
    best_optimizer_state: Dict[str, Any],
    best_epoch: int,
    best_val_mae: float,
    learning_rate: float,
    epochs: int,
    patience: Optional[int],
    device: torch.device,
    seed: Optional[int],
    checkpoint_metadata: Optional[Dict[str, Any]],
) -> None:
    if checkpoint_path is None:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': best_state_dict,
        'optimizer_state_dict': best_optimizer_state,
        'training_config': {
            'learning_rate': learning_rate,
            'epochs': epochs,
            'patience': patience,
            'device': str(device),
            'seed': seed,
            'model_class': model.__class__.__name__,
        },
        'best_epoch': best_epoch,
        'best_val_mae': best_val_mae,
        'checkpoint_metadata': checkpoint_metadata,
        'history': {
            key: value
            for key, value in history.items()
            if key != 'best_state_dict'
        },
    }
    torch.save(checkpoint, path)
