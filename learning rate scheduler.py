import math

class GPTLearningRateScheduler:
    def __init__(self, max_lr: float, min_lr: float, warm_up_iters: int, plateau_iters: int, cooldown_iters: int, start_lr: float = 0.0):
        self.start_lr: float = start_lr
        self.min_lr: float = min_lr
        self.max_lr: float = max_lr
        self.warm_up_iters: int = warm_up_iters
        self.plateau_iters: int = plateau_iters
        self.cooldown_iters: int = cooldown_iters
        self.total_iters: int = warm_up_iters + plateau_iters + cooldown_iters

    def linear_warm_up(self, current_iter: int):
        """Linearly increase LR from start_lr to max_lr."""
        lr_increment = (self.max_lr - self.start_lr) / self.warm_up_iters
        return self.start_lr + lr_increment * current_iter

    def plateau(self, current_iter: int):
        """Keep LR constant at max_lr."""
        return self.max_lr

    def linear_cooldown(self, current_iter: int):
        """Linearly decrease LR from max_lr to min_lr."""
        elapsed_cooldown_iters = current_iter - self.warm_up_iters - self.plateau_iters
        lr_decrement = (self.max_lr - self.min_lr) / self.cooldown_iters
        return self.max_lr - lr_decrement * elapsed_cooldown_iters

    def get_lr(self, current_iter: int):
        if current_iter <= self.warm_up_iters:
            lr = self.linear_warm_up(current_iter)
        elif current_iter <= self.warm_up_iters + self.plateau_iters:
            lr = self.plateau(current_iter)
        elif current_iter <= self.total_iters:
            lr = self.linear_cooldown(current_iter)
        else:
            lr = self.min_lr  # Keep LR at min_lr after cooldown
        return lr
