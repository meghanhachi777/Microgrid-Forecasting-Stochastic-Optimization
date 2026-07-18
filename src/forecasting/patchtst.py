"""PatchTST transformer for 168 h -> 24 h wind-speed forecasting,
extracted from WindForecasting.ipynb.

Requires PyTorch and transformers (imported lazily)."""

from __future__ import annotations

import time

import numpy as np

from ..utils.data import inverse_transform
from ..utils.paths import CONTEXT_LENGTH, FORECAST_HORIZON, SEED


def set_seeds(seed: int = SEED) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset_class():
    """Return the WindPatchTSTDataset class (lazy torch import)."""
    import torch
    from torch.utils.data import Dataset

    class WindPatchTSTDataset(Dataset):
        def __init__(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            if X.ndim != 3:
                raise ValueError(
                    f"X must have 3 dimensions, but received {X.shape}")
            if y.ndim == 2:
                y = np.expand_dims(y, axis=-1)
            if y.ndim != 3:
                raise ValueError(
                    f"y must have 2 or 3 dimensions, but received {y.shape}")
            if len(X) != len(y):
                raise ValueError(
                    "X and y must contain the same number of samples.")

            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return {"past_values": self.X[idx],
                    "future_values": self.y[idx]}

    return WindPatchTSTDataset


def build_loaders(X_train, y_train, X_val, y_val, X_test, y_test,
                  batch_size: int = 64):
    import torch
    from torch.utils.data import DataLoader

    DatasetCls = make_dataset_class()
    pin = torch.cuda.is_available()
    train_loader = DataLoader(DatasetCls(X_train, y_train),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(DatasetCls(X_val, y_val),
                            batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)
    test_loader = DataLoader(DatasetCls(X_test, y_test),
                             batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin)
    return train_loader, val_loader, test_loader


def build_model(
    context_length: int = CONTEXT_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON,
    device: str = "cpu",
):
    """PatchTST: patch 24 / stride 12, d_model 64, 3 layers, 4 heads,
    FFN 128, dropout regularization, std scaling, MSE loss."""
    from transformers import PatchTSTConfig, PatchTSTForPrediction

    config = PatchTSTConfig(
        num_input_channels=1,
        context_length=context_length,
        prediction_length=forecast_horizon,
        patch_length=24,
        patch_stride=12,
        d_model=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        ffn_dim=128,
        dropout=0.2,
        attention_dropout=0.1,
        positional_dropout=0.1,
        head_dropout=0.2,
        scaling="std",
        loss="mse",
        channel_attention=False,
        prediction_channel_indices=None,
    )
    return PatchTSTForPrediction(config).to(device)


def _train_one_epoch(model, data_loader, optimizer, device):
    import torch

    model.train()
    total_loss, total_samples = 0.0, 0
    for batch in data_loader:
        past_values = batch["past_values"].to(device)
        future_values = batch["future_values"].to(device)
        optimizer.zero_grad()
        outputs = model(past_values=past_values,
                        future_values=future_values)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss detected: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_size = past_values.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def _evaluate_loss(model, data_loader, device):
    import torch

    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for batch in data_loader:
            past_values = batch["past_values"].to(device)
            future_values = batch["future_values"].to(device)
            outputs = model(past_values=past_values,
                            future_values=future_values)
            batch_size = past_values.size(0)
            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def train(
    model,
    train_loader,
    val_loader,
    device: str = "cpu",
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 12,
    best_model_path: str = "best_patchtst_wind.pt",
    verbose: bool = True,
):
    """AdamW + ReduceLROnPlateau training loop with early stopping;
    restores and returns the best-validation-loss weights."""
    import torch

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)

    train_losses, val_losses = [], []
    best_val_loss, best_epoch = np.inf, 0
    epochs_without_improvement = 0
    training_start = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, device)
        val_loss = _evaluate_loss(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} "
                  f"| Val Loss: {val_loss:.6f} | LR: {current_lr:.7f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_epoch = val_loss, epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}.")
            break

    model.load_state_dict(torch.load(
        best_model_path, map_location=device, weights_only=True))
    model.eval()
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_seconds": time.time() - training_start,
    }


def extract_prediction_tensor(outputs):
    candidate_attributes = [
        "prediction_outputs", "predictions", "prediction_logits"]
    for attribute in candidate_attributes:
        if hasattr(outputs, attribute):
            value = getattr(outputs, attribute)
            if value is not None:
                return value
    raise AttributeError(
        "Could not find a prediction tensor in the PatchTST output. "
        f"Available output fields: {outputs.keys()}")


def predict(model, test_loader, scaler, device: str = "cpu",
            expected_shape: tuple | None = None) -> np.ndarray:
    """Predict on the test loader and inverse-transform to original
    units. Returns an (n, horizon) array."""
    import torch

    predictions_scaled = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            past_values = batch["past_values"].to(device)
            outputs = model(past_values=past_values)
            predictions = extract_prediction_tensor(outputs)
            predictions_scaled.append(
                predictions.detach().cpu().numpy())
    predictions_scaled = np.concatenate(predictions_scaled, axis=0)

    if predictions_scaled.ndim == 3:
        if predictions_scaled.shape[-1] == 1:
            predictions_scaled = predictions_scaled.squeeze(-1)
        elif predictions_scaled.shape[1] == 1:
            predictions_scaled = predictions_scaled.squeeze(1)

    if (expected_shape is not None
            and predictions_scaled.shape != expected_shape):
        raise ValueError(
            "PatchTST predictions have unexpected shape: "
            f"{predictions_scaled.shape} vs {expected_shape}")

    return inverse_transform(scaler, predictions_scaled)
