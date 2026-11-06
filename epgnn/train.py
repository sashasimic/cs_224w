"""Training loop shared by the torch models (gru_mlp, gat, hetero).

Data layout: a "day batch" holds every zone of B delivery days, so tensors are (B, Z, ...) and the spread
term and the per-day graph (for the GNNs) can be formed inside the batch:
    hist   (B, Z, W, n_hist)   z-scored history windows
    xday   (B, Z, H, n_day)    z-scored day-D inputs per hour slot
    target (B, Z, H)           residual target (epgnn.data.scaling)
    mask   (B, Z, H)           True where the hour exists, has a price and a finite target

A model is any nn.Module with forward(hist, xday, extra) -> (B, Z, H). `extra` is a dict the model may
ignore (the GNNs put edge_index / edge features there).
"""
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from epgnn.paths import ROOT


def device(name: str | None = None) -> torch.device:
    """`name` from the config ("cpu", "mps", "cuda") overrides the automatic choice."""
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                spread_pairs: torch.Tensor | None = None, spread_weight: float = 0.0) -> torch.Tensor:
    """L1 on valid hours, plus `spread_weight` x L1 on the difference between the two zones of each
    interconnected pair (spread_pairs: (2, P) zone indices), so the model is also judged on spreads."""
    m = mask.float()
    err = (pred - target).abs()
    loss = (err * m).sum() / m.sum().clamp(min=1)
    if spread_pairs is not None and spread_weight > 0:
        a, b = spread_pairs
        mm = m[:, a] * m[:, b]
        sp = ((pred[:, a] - pred[:, b]) - (target[:, a] - target[:, b])).abs()
        loss = loss + spread_weight * (sp * mm).sum() / mm.sum().clamp(min=1)
    return loss


class DayBatches:
    """Index-based batching of day-major tensors (first axis = days). Shuffles days each epoch."""

    def __init__(self, tensors: dict[str, torch.Tensor], batch_days: int, shuffle: bool, seed: int = 0) -> None:
        self.t = tensors
        self.n = next(iter(tensors.values())).shape[0]
        self.b = batch_days
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = self.rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        for i in range(0, self.n, self.b):
            idx = torch.as_tensor(order[i:i + self.b])
            yield {k: v[idx] for k, v in self.t.items()}

    def __len__(self) -> int:
        return (self.n + self.b - 1) // self.b


def _epoch(model: nn.Module, batches: DayBatches, dev: torch.device, extra: dict, cfg: dict,
           opt: torch.optim.Optimizer | None) -> float:
    train = opt is not None
    model.train(train)
    pairs = extra.get("spread_pairs")
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for b in batches:
            b = {k: v.to(dev) for k, v in b.items()}
            pred = model(b["hist"], b["xday"], extra)
            loss = masked_loss(pred, b["target"], b["mask"], pairs, cfg.get("spread_weight", 0.0))
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                opt.step()
            total += float(loss.detach()) * len(b["hist"])
            count += len(b["hist"])
    return total / max(count, 1)


def fit(model: nn.Module, train_b: DayBatches, val_b: DayBatches | None, cfg: dict, extra: dict,
        run_dir: Path | None = None) -> dict:
    """Adam + early stopping on the validation loss. Returns {"epochs": n, "best_val": v, "history": [...]}.
    Restores the best weights into `model`; saves them to run_dir/model.pt if run_dir is given."""
    dev = device(cfg.get("device"))
    print(f"  training on {dev}, {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{len(train_b)} batches of {train_b.b} days per epoch", flush=True)
    model.to(dev)
    extra = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in extra.items()}
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 0.0))
    best, best_state, bad, hist = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    for ep in range(cfg.get("epochs", 50)):
        t0 = time.time()
        tr = _epoch(model, train_b, dev, extra, cfg, opt)
        va = _epoch(model, val_b, dev, extra, cfg, None) if val_b is not None else tr
        hist.append({"epoch": ep + 1, "train": tr, "val": va, "seconds": round(time.time() - t0, 1)})
        print(f"  epoch {ep + 1:3d}  train {tr:.4f}  val {va:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if va < best - 1e-5:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.get("patience", 8):
                break
    model.load_state_dict(best_state)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, run_dir / "model.pt")
    return {"epochs": len(hist), "best_val": best, "history": hist}


def predict(model: nn.Module, batches: DayBatches, extra: dict, dev_name: str | None = None) -> np.ndarray:
    dev = device(dev_name)
    model.to(dev).eval()
    extra = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in extra.items()}
    out = []
    with torch.no_grad():
        for b in batches:
            out.append(model(b["hist"].to(dev), b["xday"].to(dev), extra).cpu().numpy())
    return np.concatenate(out)


def run_dir(name: str) -> Path:
    return ROOT / "runs" / name
