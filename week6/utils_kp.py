# Import matplotlib for plotting learning rate vs loss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision import datasets, transforms
from pathlib import Path
import timm

def make_model(model_name, num_classes, device, verbose=False):
    """Create and prepare a timm model for transfer learning.

    This function:
    - expects a timm model name and number of output classes
    - will create a pretrained model (weights from ImageNet)
    - freezes all parameters so the backbone is not trained initially
    - later code will unfreeze the classifier/head only so you can warm-up the head

    Args:
        model_name (str): name of timm model (e.g. 'convnext_tiny')
        num_classes (int): number of output classes for the classifier head
        device (torch.device): device to move the model to
        verbose (bool): print classifier info when True

    Returns:
        torch.nn.Module: the created model

    Example Usage:
        model = make_model('convnext_tiny', 10, device)
    """
    #pre trained =True or you get random weights
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    if verbose: print("Classifier layer:", model.get_classifier())

    # Freeze backbone; train classifier head first
    for name, p in model.named_parameters():
        p.requires_grad = False

    # Unfreeze classifier / head only (name depends on model family; get via get_classifier())
    clf_name = model.get_classifier()

    #make sure the last classifier layer is trainable
    for name, p in clf_name.named_parameters():
        p.requires_grad = True

    model = model.to(device)
    return model

# For a sweep of learning rates, reinitialize the model for each rate, train for a **few batches**, and save (learning rate, loss). <br>
# When done plot the loss versus learning rate, then choose the largest learning rate on the descending slope—just before the loss starts rising.
class LearningRateFinder:
    """
    Utility class to help find an optimal learning rate for training neural networks.
    This class runs a series of short training loops over a range of learning rates,
    records the average loss for each, and provides a method to plot the results.
    The goal is to help select a learning rate that leads to rapid loss decrease
    without instability.
    Args:
        model_fn (callable): A function that returns a new instance of the model to be trained.
        criterion (callable): The loss function to use for training.
        device (torch.device or str): The device on which to run the model and data.
    Attributes:
        model_fn (callable): Function to instantiate the model.
        criterion (callable): Loss function.
        device (torch.device or str): Device for computation.
        results (list): Stores tuples of (learning rate, average loss).
    Methods:
        run(lr_list, train_loader, num_batches=5):
            Runs training for each learning rate in lr_list for a few batches,
            records the average loss, and stores the results.
        plot():
            Plots the recorded average losses against the learning rates on a log scale.
    
    Example usage:
        lr_candidates = np.logspace(-6, -1, num=8)
        finder = LearningRateFinder(make_model, criterion, device)
        results = finder.run(lr_candidates, train_loader, num_batches=5)
        finder.plot()

        best_lr, best_loss = min(results, key=lambda t: t[1])
        best_lr, best_loss
    
    """

    def __init__(self, model_name, num_classes, criterion, device):
        self.model_name = model_name
        self.num_classes = num_classes  
        self.criterion = criterion
        self.device = device
        self.results = []

    def _reinit(self):
        return make_model(self.model_name, self.num_classes, self.device)

    def run(self, lr_list, train_loader, num_batches=5):
        self.results.clear()
        for lr in lr_list:
            model = self._reinit()
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
#            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model.train()
            it = iter(train_loader); losses = []
            for _ in range(num_batches):
                xb, yb = next(it)
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = self.criterion(logits, yb)
                loss.backward(); optimizer.step()
                losses.append(loss.item())
            avg_loss = float(np.mean(losses)) if losses else float('nan')
            self.results.append((float(lr), avg_loss))
        return self.results

    def plot(self):
        if not self.results:
            print("No results. Run .run() first."); return
        lrs, losses = zip(*self.results)
        plt.figure()
        plt.plot(lrs, losses, marker='o')
        plt.xscale('log')
        plt.xlabel("Learning Rate"); plt.ylabel("Average Loss (few batches)")
        plt.title("LR Finder — Loss vs Learning Rate"); plt.show()
