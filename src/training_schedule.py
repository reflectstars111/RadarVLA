"""Stateless epoch-indexed learning rates; resume is determined by pinned protocol."""
import math


def learning_rate_for_update(update, total_updates, base_lr, schedule='flat_cosine',
                             flat_updates=0, min_ratio=.1):
    """One-based optimizer updates, including the final partial accumulation."""
    if (type(update) is not int or type(total_updates) is not int
            or not 1 <= update <= total_updates or not math.isfinite(base_lr) or base_lr <= 0
            or schedule not in ('constant', 'flat_cosine', 'cosine')
            or type(flat_updates) is not int or flat_updates < 0
            or not math.isfinite(min_ratio) or not 0 < min_ratio <= 1):
        raise ValueError('Invalid optimizer-update learning rate schedule')
    if schedule == 'constant':
        return base_lr
    if total_updates < 2 or (schedule == 'flat_cosine' and flat_updates >= total_updates):
        raise ValueError('Cosine schedule needs at least two updates and a decay interval')
    if schedule == 'cosine' or flat_updates == 0:
        progress = (update - 1) / (total_updates - 1)
    else:
        progress = max(0., (update - flat_updates) / (total_updates - flat_updates))
    return base_lr * (min_ratio + .5 * (1-min_ratio) * (1+math.cos(math.pi*progress)))


def learning_rate_for_epoch(epoch, epochs, base_lr, schedule='constant', decay_start_epoch=5, min_ratio=.1):
    if (type(epoch) is not int or type(epochs) is not int or not 1 <= epoch <= epochs
            or not math.isfinite(base_lr) or base_lr <= 0):
        raise ValueError('Invalid epoch budget or base learning rate')
    if (schedule not in ('constant', 'flat_cosine') or type(decay_start_epoch) is not int
            or decay_start_epoch < 0 or not math.isfinite(min_ratio) or not 0 < min_ratio <= 1):
        raise ValueError('Invalid learning rate schedule')
    if schedule == 'constant':
        return base_lr
    if decay_start_epoch >= epochs:
        raise ValueError('Cosine decay must start before the final epoch')
    progress = max(0., (epoch-decay_start_epoch)/(epochs-decay_start_epoch))
    return base_lr * (min_ratio + .5*(1-min_ratio)*(1+math.cos(math.pi*progress)))
