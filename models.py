"""
Metric helpers used by eval.py.

  - SSIM / PSNR: pixel-level, via scikit-image. Computed per image pair and
    averaged.
  - LPIPS: perceptual, via the `lpips` package (Zhang et al. 2018). Lower = more
    perceptually similar. Inputs in [-1, 1].
  - FID: distribution-level, via `pytorch-fid`. Computed once over the whole set
    of predictions vs the whole set of ground truth (not per-image).

LPIPS and FID download small backbone weights the first time they run; that's
fine for evaluation. Inference (infer.py) does NOT use any of these.
"""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def load_uint8(path, mode):
    return np.asarray(Image.open(path).convert(mode), dtype=np.uint8)


def ssim_psnr(pred_path, gt_path):
    pred = load_uint8(pred_path, "RGB")
    gt = load_uint8(gt_path, "RGB")
    if pred.shape != gt.shape:
        gt = np.asarray(Image.fromarray(gt).resize(pred.shape[1::-1]), dtype=np.uint8)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=255)
    psnr = peak_signal_noise_ratio(gt, pred, data_range=255)
    return ssim, psnr


def build_lpips(device="cpu", net="alex"):
    import lpips
    return lpips.LPIPS(net=net).to(device).eval()


def lpips_score(lpips_model, pred_path, gt_path, device="cpu"):
    import torch
    pred = load_uint8(pred_path, "RGB").astype(np.float32)
    gt = load_uint8(gt_path, "RGB").astype(np.float32)

    def to_tensor(a):
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0  # [-1,1]
        return t.to(device)

    with torch.no_grad():
        d = lpips_model(to_tensor(pred), to_tensor(gt))
    return float(d.item())


def fid_score(pred_dir, gt_dir, device="cpu", batch_size=50, dims=2048):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    return float(
        calculate_fid_given_paths([gt_dir, pred_dir], batch_size=batch_size,
                                  device=device, dims=dims)
    )
