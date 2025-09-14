import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

@torch.no_grad()  # Disable gradient tracking inside this function (saves memory/compute)
def evaluate(model, loader, criterion, device):
    """
    Puts the model in evaluation mode and computes the average loss
    over all batches in `loader`.

    Args:
        model:     a torch.nn.Module
        loader:    DataLoader for validation/test data
        criterion: loss function (e.g., nn.MSELoss())
        device:    'cuda' or 'cpu' device to run on

    Returns:
        avg_loss (float): mean loss over the dataset
    """
    model.eval()          # turn off dropout / use running stats for batchnorm, etc.
    total_loss, total_n = 0.0, 0

    for xb, yb in loader:
        # Move the current batch to the right device
        xb, yb = xb.to(device), yb.to(device)

        # Forward pass only (no grad because of @torch.no_grad)
        preds = model(xb)
        loss = criterion(preds, yb)

        # Accumulate *sum* of losses so we can compute dataset average
        total_loss += loss.item() * xb.size(0)  # multiply by batch size
        total_n    += xb.size(0)

    return total_loss / total_n  # average loss over all examples


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=200, log_every=20):
    """
    Standard training loop with per-epoch validation.

    Args:
        model:        a torch.nn.Module
        train_loader: DataLoader for training data
        val_loader:   DataLoader for validation data
        criterion:    loss function (e.g., nn.MSELoss())
        optimizer:    torch.optim optimizer (e.g., Adam)
        device:       'cuda' or 'cpu'
        epochs:       number of training epochs
        log_every:    how often to print progress (in epochs)
    """
    for epoch in range(1, epochs + 1):
        model.train()                # enable training behavior (dropout, batchnorm updates, etc.)
        running_loss, total_n = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # 1) Reset gradients from the previous step
            optimizer.zero_grad(set_to_none=True)
            #    set_to_none=True can be a tiny memory/perf win vs. zeroing to 0.0

            # 2) Forward pass: compute predictions and loss
            preds = model(xb)
            loss  = criterion(preds, yb)  # e.g., MSE loss

            # 3) Backward pass: compute gradients w.r.t. parameters
            loss.backward()

            # 4) Update parameters using the optimizer, this will subtract a small bit (lr*grad) of the gradient from each parameter
            optimizer.step()

            # Track running training loss (sum over all samples)
            running_loss += loss.item() * xb.size(0)
            total_n      += xb.size(0)

        # Average train loss this epoch
        train_mse = running_loss / total_n

        # Evaluate on the validation set (no-grad inside)
        val_mse = evaluate(model, val_loader, criterion, device)

        # Print progress
        if (epoch % log_every == 0) or (epoch == 1):
            print(f"epoch {epoch:03d} | train MSE {train_mse:.4f} | val MSE {val_mse:.4f}")
