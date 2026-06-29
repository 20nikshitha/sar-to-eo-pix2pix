"""
Dataset / dataloader.

Expects the layout produced by scripts/prepare_dataset.py:

    data/sen12/
        train/
            sar/  <name>.png   (single channel, 8-bit, dB-scaled, min-max [0,255])
            eo/   <name>.png   (RGB, 8-bit)
        val/
            sar/ ...
            eo/  ...

SAR and EO are matched BY FILENAME. The SAR representation here is deliberately
identical to the inference I/O contract (single-channel, dB-scaled, min-max to
[0,255]) so that train-time and inference-time inputs have the same statistics.
That consistency matters more than the exact normalisation choice.

Both modalities are scaled to [-1, 1] (because the generator ends in Tanh).
"""

import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


class PairedSAROpticalDataset(Dataset):
    def __init__(self, root, split, image_size=256, augment=False):
        self.sar_dir = os.path.join(root, split, "sar")
        self.eo_dir = os.path.join(root, split, "eo")
        if not (os.path.isdir(self.sar_dir) and os.path.isdir(self.eo_dir)):
            raise FileNotFoundError(
                f"Expected '{self.sar_dir}' and '{self.eo_dir}'. "
                f"Run scripts/prepare_dataset.py first."
            )
        sar_names = set(_list_images(self.sar_dir))
        eo_names = set(_list_images(self.eo_dir))
        # only keep pairs that exist on both sides
        self.names = sorted(sar_names & eo_names)
        missing = (sar_names ^ eo_names)
        if missing:
            print(f"[dataset] {split}: ignoring {len(missing)} unpaired files")
        if not self.names:
            raise RuntimeError(f"No matched SAR/EO pairs found in {root}/{split}")

        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def _load(self, path, mode):
        img = Image.open(path).convert(mode)
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        return np.asarray(img, dtype=np.float32)

    def __getitem__(self, idx):
        name = self.names[idx]
        sar = self._load(os.path.join(self.sar_dir, name), "L")     # (H, W)
        eo = self._load(os.path.join(self.eo_dir, name), "RGB")     # (H, W, 3)

        # paired augmentation: the SAME flip is applied to both images so they
        # stay co-registered. Only flips — rotations/crops would break the
        # SAR/optical geometric correspondence or introduce padding artefacts.
        if self.augment:
            if random.random() < 0.5:        # horizontal
                sar = sar[:, ::-1]
                eo = eo[:, ::-1, :]
            if random.random() < 0.5:        # vertical
                sar = sar[::-1, :]
                eo = eo[::-1, :, :]

        sar = np.ascontiguousarray(sar)
        eo = np.ascontiguousarray(eo)

        # to [-1, 1]
        sar_t = torch.from_numpy(sar).unsqueeze(0) / 127.5 - 1.0      # (1, H, W)
        eo_t = torch.from_numpy(eo).permute(2, 0, 1) / 127.5 - 1.0    # (3, H, W)

        return {"sar": sar_t, "eo": eo_t, "name": name}
