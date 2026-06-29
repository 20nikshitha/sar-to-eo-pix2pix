# ===========================================================================
# Config for SAR -> EO Pix2Pix.
# Everything that affects a run lives here so experiments are reproducible.
# The two ablation switches are train.use_gan and train.lambda_perceptual.
# ===========================================================================

experiment_name: pix2pix_sen12_baseline
seed: 42
output_dir: outputs

data:
  # Folder produced by scripts/prepare_dataset.py (expects train/ and val/,
  # each with sar/ and eo/ subfolders, matched by filename).
  root: data/sen12
  image_size: 256
  sar_channels: 1          # VV only. Use 2 if you prepared VV+VH (SEN12MS).
  batch_size: 8            # 8 fits a 16 GB T4 at 256x256; raise if you have room.
  num_workers: 2
  augment: true            # paired horizontal/vertical flips only

model:
  ngf: 64                  # generator base channels
  ndf: 64                  # discriminator base channels
  num_downs: 8             # U-Net depth: 256 = 2**8
  n_layers_d: 3            # PatchGAN -> ~70x70 receptive field
  norm: batch              # 'batch' (default) or 'instance'
  use_dropout: true        # dropout in the U-Net bottleneck (Pix2Pix style)

train:
  epochs: 100
  lr: 0.0002
  beta1: 0.5               # Adam betas as in the Pix2Pix paper
  beta2: 0.999
  lambda_l1: 100.0         # weight on the L1 reconstruction term
  lambda_perceptual: 0.0   # >0 turns ON VGG perceptual loss (ablation knob #2)
  use_gan: true            # false -> L1(+perceptual) only (ablation knob #1)
  gan_mode: lsgan          # 'lsgan' (default, stabler) or 'vanilla'
  save_every: 10           # also always keeps best.pth (by val L1) and last.pth
  val_every: 1
