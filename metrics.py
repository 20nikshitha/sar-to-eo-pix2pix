"""
Loss functions.

Three pieces:
  - GANLoss: adversarial term. Supports vanilla (BCE) and LSGAN (MSE). LSGAN is
    a bit more stable so it's the default in the config.
  - L1 loss is just nn.L1Loss in the training loop (the heavy-weighted term,
    lambda_L1=100 as in Pix2Pix). L1 is what gives low-frequency correctness;
    the GAN term sharpens high-frequency texture.
  - VGGPerceptualLoss: OPTIONAL. Only built if lambda_perceptual > 0. It compares
    deep VGG features instead of raw pixels, which lines up better with how the
    human eye (and LPIPS) judges similarity. This is the second ablation knob.

Note on the perceptual loss and the offline-inference rule: VGG weights download
once at TRAIN time. That's fine — the spec only requires *inference* to be
offline, and infer.py never touches VGG.
"""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, gan_mode="lsgan", target_real=1.0, target_fake=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(float(target_real)))
        self.register_buffer("fake_label", torch.tensor(float(target_fake)))
        self.gan_mode = gan_mode
        if gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f"gan_mode '{gan_mode}' not supported")

    def _target(self, prediction, target_is_real):
        t = self.real_label if target_is_real else self.fake_label
        return t.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        return self.loss(prediction, self._target(prediction, target_is_real))


class VGGPerceptualLoss(nn.Module):
    """L1 distance between VGG16 feature maps. Inputs are expected in [-1, 1]
    (the generator's output range); we convert to [0,1] and apply ImageNet
    normalisation before feeding VGG."""

    def __init__(self, layers=(3, 8, 15, 22), device="cpu"):
        super().__init__()
        import torchvision  # local import so torchvision.models is only needed if used
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        vgg = torchvision.models.vgg16(weights=weights).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.layers = set(layers)
        self.criterion = nn.L1Loss()
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _prep(self, x):
        x = (x + 1.0) / 2.0          # [-1,1] -> [0,1]
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        f, r = self._prep(fake), self._prep(real)
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            f, r = layer(f), layer(r)
            if i in self.layers:
                loss = loss + self.criterion(f, r)
        return loss
