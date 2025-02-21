import torch
import numpy as np

class CLRPlusScheduler:
    def __init__(self, optimizer, base_lr, max_lr, step_size, adjust_on="loss", adjustment_factor=0.1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.iteration = 0
        self.adjust_on = adjust_on
        self.adjustment_factor = adjustment_factor
        self.prev_loss = None

    def step(self, loss=None, gradients=None):
        # Loss trend adjustment
        if self.adjust_on == "loss" and loss is not None:
            if self.prev_loss is not None and loss > self.prev_loss:
                self.max_lr *= (1 - self.adjustment_factor)  # Shrink learning rate range
                self.base_lr *= (1 - self.adjustment_factor)
            self.prev_loss = loss
        
        # Gradient magnitude adjustment
        if self.adjust_on == "gradients" and gradients is not None:
            grad_norm = gradients.norm().item()
            self.max_lr *= 1 / (1 + self.adjustment_factor * grad_norm)

        # Calculate cyclical learning rate
        cycle = np.floor(1 + self.iteration / (2 * self.step_size))
        x = np.abs(self.iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.iteration += 1
