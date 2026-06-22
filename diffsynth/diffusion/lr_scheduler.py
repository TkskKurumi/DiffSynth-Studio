import math
import torch


class CosineWarmupRestartLR(torch.optim.lr_scheduler.LambdaLR):
    """
    Learning rate scheduler with linear warmup + cosine annealing + periodic restarts.

    During warmup (step <= warmup_steps):
        lr = lr_max * step / warmup_steps

    After warmup:
        lr = lr_max * (cos(pi * ((step - warmup_steps) % restart_steps) / restart_steps) + 1) / 2
    """

    def __init__(self, optimizer, warmup_steps: int, restart_steps: int, last_epoch: int = -1):
        self.warmup_steps = max(int(warmup_steps), 1)
        self.restart_steps = max(int(restart_steps), 1)

        def lr_lambda(current_step):
            if current_step <= self.warmup_steps:
                return current_step / self.warmup_steps
            else:
                tmp = (current_step - self.warmup_steps) % self.restart_steps
                tmp = tmp / self.restart_steps * math.pi
                tmp = (math.cos(tmp) + 1) / 2
                return tmp

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def get_scheduler(
    optimizer,
    scheduler_type: str = "constant",
    warmup_steps: int = 0,
    restart_steps: int = 1000,
):
    """
    Factory function to create a learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer.
        scheduler_type: One of "constant" or "cosine-warmup-restart".
        warmup_steps: Number of linear warmup steps (used by cosine-warmup-restart).
        restart_steps: Number of steps per cosine cycle (used by cosine-warmup-restart).

    Returns:
        A torch.optim.lr_scheduler instance.
    """
    if scheduler_type == "cosine-warmup-restart":
        return CosineWarmupRestartLR(
            optimizer,
            warmup_steps=warmup_steps,
            restart_steps=restart_steps,
        )
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer)
    else:
        raise ValueError(f"Unknown lr scheduler type: {scheduler_type!r}. Supported: constant, cosine-warmup-restart.")
