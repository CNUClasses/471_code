# Import matplotlib for plotting learning rate vs loss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision import datasets, transforms
from pathlib import Path

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

    def __init__(self, model_fn, criterion, device):
        self.model_fn = model_fn
        self.criterion = criterion
        self.device = device
        self.results = []

    def _reinit(self):
        return self.model_fn().to(self.device)

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
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(train_loader); xb, yb = next(it)
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
