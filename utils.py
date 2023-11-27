import torch.optim as optim
from torch.optim import lr_scheduler
import math


class WarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_factor, eta_min=0, last_epoch=-1):
        """
        Learning rate scheduler with warmup and cosine annealing.

        Args:
        - optimizer: Optimizer to adjust the learning rate for.
        - warmup_epochs: Number of epochs for warmup phase.
        - max_epochs: Maximum number of epochs.
        - warmup_factor: Warmup factor to scale the learning rate during warmup.
        - eta_min: Minimum learning rate after cosine annealing (default: 0).
        - last_epoch: The index of the last epoch (default: -1).

        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_factor = warmup_factor
        self.eta_min = eta_min

        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Get the learning rates for each parameter group based on the current epoch.

        Returns:
        - List of learning rates for each parameter group.

        """
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * self.warmup_factor * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / self.max_epochs)) / 2
                    for base_lr in self.base_lrs]