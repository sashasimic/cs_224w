"""Training loop shared by the torch models (gat, hetero).

masked_loss(pred, target, mask, spread_pairs=None, spread_weight=0.0)
    L1 on valid hours, plus `spread_weight` x L1 on (pred_a - pred_b) vs (target_a - target_b) over the
    interconnected pairs given as an index array (2, P) into the batch's zone dimension.

fit(model, batches_train, batches_val, cfg) -> model
    Adam, early stopping on validation loss, `cfg["epochs"]`, `cfg["lr"]`, `cfg["patience"]`; pretrain
    then fine-tune is two calls with different batches. Checkpoints go to runs/<run_id>/.
"""
import torch


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                spread_pairs: torch.Tensor | None = None, spread_weight: float = 0.0) -> torch.Tensor:
    # Index before arithmetic: NaN * zero is still NaN, including in backward.
    valid = mask.bool() & torch.isfinite(target)
    loss = (pred[valid] - target[valid]).abs().sum() / valid.sum().clamp(min=1)
    if spread_pairs is not None and spread_weight > 0:
        a, b = spread_pairs
        m = valid[a] & valid[b]
        sp = ((pred[a][m] - pred[b][m]) - (target[a][m] - target[b][m])).abs()
        loss = loss + spread_weight * sp.sum() / m.sum().clamp(min=1)
    return loss


def fit(model: torch.nn.Module, batches_train, batches_val, cfg: dict) -> torch.nn.Module:
    raise NotImplementedError("shared loop; owner B writes it with checkpoint 3, C reuses it")
