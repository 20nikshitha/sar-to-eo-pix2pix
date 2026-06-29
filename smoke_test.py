"""
Small helpers used across train / eval / infer.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Seed everything we can for reproducibility. Note: full bit-exact
    reproducibility on GPU also needs deterministic cuDNN, which slows training,
    so we leave benchmark on and accept tiny run-to-run variation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for saving/visualisation."""
    return (t.clamp(-1, 1) + 1.0) / 2.0


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) in [-1,1] -> (H, W, C) uint8 in [0,255]."""
    img = denormalize(t.detach()).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255.0 + 0.5).astype(np.uint8)


def save_checkpoint(path, generator, discriminator, optim_g, optim_d,
                    epoch, model_cfg):
    """We store the model_cfg inside the checkpoint so infer.py can rebuild the
    exact generator architecture with no external config file -> robust + offline."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "epoch": epoch,
        "model_cfg": model_cfg,
        "generator": generator.state_dict(),
    }
    if discriminator is not None:
        state["discriminator"] = discriminator.state_dict()
    if optim_g is not None:
        state["optim_g"] = optim_g.state_dict()
    if optim_d is not None:
        state["optim_d"] = optim_d.state_dict()
    torch.save(state, path)


def plot_loss_curves(csv_path, out_path, adversarial=True):
    """Read the per-epoch loss CSV and save the curve plot. For the adversarial
    run we plot G and D losses; the train-vs-val comparison uses the L1 term,
    which is the only loss directly comparable between the two phases."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else float("nan"))
                         for k, v in r.items()})
    if not rows:
        return
    ep = [r["epoch"] for r in rows]

    if adversarial:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(ep, [r["train_G_total"] for r in rows], label="G total")
        ax[0].plot(ep, [r["train_G_GAN"] for r in rows], label="G GAN")
        ax[0].plot(ep, [r["train_G_L1"] for r in rows], label="G L1")
        ax[0].plot(ep, [r["train_D"] for r in rows], label="D")
        ax[0].set_title("Training losses (generator & discriminator)")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()

        ax[1].plot(ep, [r["train_G_L1"] for r in rows], label="train L1")
        ax[1].plot(ep, [r["val_L1"] for r in rows], label="val L1")
        ax[1].set_title("Train vs validation L1")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("L1"); ax[1].legend()
    else:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ep, [r["train_G_L1"] for r in rows], label="train L1")
        ax.plot(ep, [r["val_L1"] for r in rows], label="val L1")
        ax.set_title("Train vs validation L1 (L1-only run)")
        ax.set_xlabel("epoch"); ax.set_ylabel("L1"); ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
