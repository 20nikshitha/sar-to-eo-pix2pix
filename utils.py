"""
Model definitions for SAR -> EO translation.

I went with the standard Pix2Pix pairing:
  - Generator: U-Net (encoder-decoder with skip connections). The skips matter a
    lot here because a lot of the *structure* (roads, field boundaries, coastlines)
    is already present in the SAR image and should pass straight through; the
    network mostly has to invent colour/texture, not geometry.
  - Discriminator: PatchGAN. It classifies overlapping 70x70 patches as real/fake
    instead of judging the whole image at once. For texture realism (which is what
    we care about in EO imagery) a patch-level critic works better and is cheaper.

The implementation follows Isola et al. (2017) "Image-to-Image Translation with
Conditional Adversarial Networks" and the reference CycleGAN-and-pix2pix repo.
I kept it close to the reference on purpose so the behaviour is well understood.
"""

import functools

import torch
import torch.nn as nn


def get_norm_layer(norm_type: str):
    """BatchNorm vs InstanceNorm. We default to batch; instance is here in case
    we ever drop to batch_size=1 on a smaller GPU."""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    raise ValueError(f"Unknown norm type: {norm_type}")


def init_weights(net: nn.Module, gain: float = 0.02):
    """Normal init, as in the Pix2Pix paper. Helps GAN training stability."""
    def _init(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(_init)
    return net


# --------------------------------------------------------------------------- #
# Generator: U-Net built recursively from skip-connection blocks
# --------------------------------------------------------------------------- #
class UnetSkipConnectionBlock(nn.Module):
    """One down/up level of the U-Net. Blocks are nested so the innermost block
    is built first and wrapped outward."""

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2,
                             padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]  # Tanh -> output in [-1, 1]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                                        stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4,
                                        stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout:
                model = model + [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        # skip connection: concatenate input with the processed output
        return torch.cat([x, self.model(x)], dim=1)


class UnetGenerator(nn.Module):
    """U-Net generator. For 256x256 inputs use num_downs=8 (256 = 2**8)."""

    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=True):
        super().__init__()
        # innermost
        block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                        submodule=None, norm_layer=norm_layer,
                                        innermost=True)
        # intermediate ngf*8 blocks (with dropout, as in the paper)
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                            submodule=block, norm_layer=norm_layer,
                                            use_dropout=use_dropout)
        # gradually reduce filters going outward
        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None,
                                        submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None,
                                        submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
                                        submodule=block, norm_layer=norm_layer)
        # outermost
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,
                                             submodule=block, outermost=True,
                                             norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------------------------------- #
# Discriminator: PatchGAN
# --------------------------------------------------------------------------- #
class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator. With n_layers=3 the receptive field is ~70x70.
    Input is the SAR image concatenated with the EO image (conditional GAN),
    so input_nc = sar_channels + 3."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        kw, padw = 4, 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                          stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                      stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # final 1-channel prediction map (one value per patch)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                               stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------------------------------- #
# Factory helpers
# --------------------------------------------------------------------------- #
def build_generator(cfg):
    norm_layer = get_norm_layer(cfg.get("norm", "batch"))
    net = UnetGenerator(
        input_nc=cfg["input_nc"],
        output_nc=cfg["output_nc"],
        num_downs=cfg.get("num_downs", 8),
        ngf=cfg.get("ngf", 64),
        norm_layer=norm_layer,
        use_dropout=cfg.get("use_dropout", True),
    )
    return init_weights(net)


def build_discriminator(cfg):
    norm_layer = get_norm_layer(cfg.get("norm", "batch"))
    net = NLayerDiscriminator(
        input_nc=cfg["input_nc"] + cfg["output_nc"],  # conditional: SAR + EO
        ndf=cfg.get("ndf", 64),
        n_layers=cfg.get("n_layers_d", 3),
        norm_layer=norm_layer,
    )
    return init_weights(net)
