"""Cosine LR schedule with linear warmup."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_schedule_with_warmup(
    optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """LR ramps linearly to base for warmup_steps, then cosine down to min_lr_ratio * base."""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        progress = min(progress, 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return LambdaLR(optimizer, lr_lambda)
