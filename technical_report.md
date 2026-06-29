"""
Training entry point.

    python train.py --config config.yaml

What it does each epoch:
  - trains the generator (and discriminator, unless use_gan: false)
  - logs G total / G GAN / G L1 / D losses (train) and L1 (val)
  - appends those to outputs/<exp>/losses.csv
  - re-renders outputs/<exp>/loss_curve.png
  - keeps the best checkpoint by validation L1, plus periodic snapshots

The two ablation switches live in the config:
  use_gan: true|false           -> L1+adversarial  vs  L1-only
  lambda_perceptual: 0.0|>0     -> with/without perceptual loss
This lets a single codebase produce the ablation cleanly by changing one line.
"""

import argparse
import csv
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PairedSAROpticalDataset
from src.losses import GANLoss, VGGPerceptualLoss
from src.models import build_generator, build_discriminator
from src.utils import set_seed, save_checkpoint, plot_loss_curves


def build_model_cfg(cfg):
    m = cfg["model"]
    return {
        "input_nc": cfg["data"]["sar_channels"],
        "output_nc": 3,
        "ngf": m["ngf"],
        "ndf": m["ndf"],
        "num_downs": m["num_downs"],
        "n_layers_d": m.get("n_layers_d", 3),
        "norm": m["norm"],
        "use_dropout": m.get("use_dropout", True),
    }


@torch.no_grad()
def validate(generator, loader, device):
    """Validation loss = mean L1 between generated and real EO. We use L1 (not the
    GAN loss) because it's a stable, comparable number across epochs and is what
    tells us about overfitting."""
    generator.eval()
    l1 = nn.L1Loss()
    total, n = 0.0, 0
    for batch in loader:
        sar = batch["sar"].to(device)
        eo = batch["eo"].to(device)
        fake = generator(sar)
        total += l1(fake, eo).item() * sar.size(0)
        n += sar.size(0)
    generator.train()
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {device}")

    exp_dir = os.path.join(cfg["output_dir"], cfg["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)
    # snapshot the exact config used, next to the outputs
    with open(os.path.join(exp_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ---- data ----
    d = cfg["data"]
    train_ds = PairedSAROpticalDataset(d["root"], "train", d["image_size"],
                                       augment=d.get("augment", True))
    val_ds = PairedSAROpticalDataset(d["root"], "val", d["image_size"],
                                     augment=False)
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=d["batch_size"], shuffle=True,
                              num_workers=d["num_workers"], drop_last=True,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=d["batch_size"], shuffle=False,
                            num_workers=d["num_workers"],
                            pin_memory=(device == "cuda"))

    # ---- models ----
    model_cfg = build_model_cfg(cfg)
    G = build_generator(model_cfg).to(device)
    use_gan = cfg["train"]["use_gan"]
    D = build_discriminator(model_cfg).to(device) if use_gan else None

    # ---- losses ----
    t = cfg["train"]
    criterion_l1 = nn.L1Loss()
    criterion_gan = GANLoss(gan_mode=t.get("gan_mode", "lsgan")).to(device) if use_gan else None
    lambda_l1 = t["lambda_l1"]
    lambda_perc = t.get("lambda_perceptual", 0.0)
    perceptual = VGGPerceptualLoss(device=device).to(device) if lambda_perc > 0 else None
    if perceptual:
        print(f"[train] perceptual loss ON (lambda={lambda_perc})")
    if not use_gan:
        print("[train] adversarial loss OFF -> L1(+perceptual) only ablation")

    # ---- optimisers ----
    opt_g = torch.optim.Adam(G.parameters(), lr=t["lr"], betas=(t["beta1"], t["beta2"]))
    opt_d = (torch.optim.Adam(D.parameters(), lr=t["lr"], betas=(t["beta1"], t["beta2"]))
             if use_gan else None)

    # ---- loss log ----
    csv_path = os.path.join(exp_dir, "losses.csv")
    fields = ["epoch", "train_G_total", "train_G_GAN", "train_G_L1", "train_D", "val_L1"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best_val = float("inf")
    for epoch in range(1, t["epochs"] + 1):
        G.train()
        run = {"G_total": 0.0, "G_GAN": 0.0, "G_L1": 0.0, "D": 0.0}
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{t['epochs']}")
        for batch in pbar:
            sar = batch["sar"].to(device)
            eo = batch["eo"].to(device)
            fake = G(sar)

            # ---------- update D ----------
            d_loss_val = 0.0
            if use_gan:
                opt_d.zero_grad(set_to_none=True)
                # real pair
                pred_real = D(torch.cat([sar, eo], dim=1))
                loss_d_real = criterion_gan(pred_real, True)
                # fake pair (detach so D update doesn't backprop into G)
                pred_fake = D(torch.cat([sar, fake.detach()], dim=1))
                loss_d_fake = criterion_gan(pred_fake, False)
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                loss_d.backward()
                opt_d.step()
                d_loss_val = loss_d.item()

            # ---------- update G ----------
            opt_g.zero_grad(set_to_none=True)
            loss_g_gan = torch.tensor(0.0, device=device)
            if use_gan:
                pred_fake_for_g = D(torch.cat([sar, fake], dim=1))
                loss_g_gan = criterion_gan(pred_fake_for_g, True)  # fool D
            loss_g_l1 = criterion_l1(fake, eo) * lambda_l1
            loss_g = loss_g_gan + loss_g_l1
            if perceptual is not None:
                loss_g = loss_g + perceptual(fake, eo) * lambda_perc
            loss_g.backward()
            opt_g.step()

            bs = sar.size(0)
            seen += bs
            run["G_total"] += loss_g.item() * bs
            run["G_GAN"] += float(loss_g_gan) * bs
            run["G_L1"] += loss_g_l1.item() * bs
            run["D"] += d_loss_val * bs
            pbar.set_postfix(G=f"{loss_g.item():.3f}", D=f"{d_loss_val:.3f}")

        # epoch averages
        for k in run:
            run[k] /= max(seen, 1)
        val_l1 = validate(G, val_loader, device)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{run['G_total']:.6f}",
                                    f"{run['G_GAN']:.6f}", f"{run['G_L1']:.6f}",
                                    f"{run['D']:.6f}", f"{val_l1:.6f}"])
        plot_loss_curves(csv_path, os.path.join(exp_dir, "loss_curve.png"),
                         adversarial=use_gan)
        print(f"[epoch {epoch}] G_total={run['G_total']:.4f} "
              f"G_L1={run['G_L1']:.4f} D={run['D']:.4f} val_L1={val_l1:.4f}")

        # checkpoints
        if val_l1 < best_val:
            best_val = val_l1
            save_checkpoint(os.path.join(exp_dir, "best.pth"), G, D, opt_g, opt_d,
                            epoch, model_cfg)
            print(f"[epoch {epoch}] new best val_L1={val_l1:.4f} -> saved best.pth")
        if epoch % t.get("save_every", 10) == 0:
            save_checkpoint(os.path.join(exp_dir, f"epoch_{epoch:03d}.pth"),
                            G, D, opt_g, opt_d, epoch, model_cfg)

    save_checkpoint(os.path.join(exp_dir, "last.pth"), G, D, opt_g, opt_d,
                    t["epochs"], model_cfg)
    print(f"[train] done. best val_L1={best_val:.4f}. outputs in {exp_dir}")


if __name__ == "__main__":
    main()
