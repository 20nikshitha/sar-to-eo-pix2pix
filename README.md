"""
Evaluation — computes all four metrics required by Section 4.

    python eval.py --pred_dir <generated_eo> --gt_dir <ground_truth_eo>

Reports:
    Primary (perceptual, drive ranking):  LPIPS (lower) , FID (lower)
    Secondary (pixel-level):              SSIM (higher) , PSNR (higher)

Workflow: generate predictions first with infer.py, then point this script at the
prediction directory and the matching ground-truth directory (matched by
filename). SSIM/PSNR/LPIPS are per-pair averages; FID is computed once over the
whole set. Results are printed and also written to <pred_dir>/metrics.json.

Why we keep generation (infer.py) and metrics (eval.py) separate: the held-out
GalaxEye evaluation only runs infer.py, so it must stand alone. eval.py is purely
for our own reporting and can use heavier metric libraries.
"""

import argparse
import json
import os

from tqdm import tqdm

from src.metrics import ssim_psnr, build_lpips, lpips_score, fid_score

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def matched_names(pred_dir, gt_dir):
    def stems(d):
        return {os.path.splitext(f)[0]: f
                for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)}
    p, g = stems(pred_dir), stems(gt_dir)
    common = sorted(set(p) & set(g))
    if not common:
        raise RuntimeError("No filename-matched pairs between pred_dir and gt_dir.")
    skipped = (set(p) ^ set(g))
    if skipped:
        print(f"[eval] {len(skipped)} files had no match and were skipped")
    return [(p[s], g[s]) for s in common]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    parser.add_argument("--lpips_net", default="alex")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device = {device}")

    pairs = matched_names(args.pred_dir, args.gt_dir)
    print(f"[eval] {len(pairs)} matched pairs")

    lpips_model = build_lpips(device=device, net=args.lpips_net)

    ssim_vals, psnr_vals, lpips_vals = [], [], []
    for pred_f, gt_f in tqdm(pairs, desc="ssim/psnr/lpips"):
        pp = os.path.join(args.pred_dir, pred_f)
        gp = os.path.join(args.gt_dir, gt_f)
        s, p = ssim_psnr(pp, gp)
        ssim_vals.append(s)
        psnr_vals.append(p)
        lpips_vals.append(lpips_score(lpips_model, pp, gp, device=device))

    def mean(xs):
        return float(sum(xs) / len(xs))

    print("[eval] computing FID over the full set ...")
    fid = fid_score(args.pred_dir, args.gt_dir, device=device)

    results = {
        "num_pairs": len(pairs),
        "LPIPS": round(mean(lpips_vals), 4),
        "FID": round(fid, 3),
        "SSIM": round(mean(ssim_vals), 4),
        "PSNR": round(mean(psnr_vals), 3),
    }

    print("\n================ RESULTS ================")
    print(f"  LPIPS (down) : {results['LPIPS']}")
    print(f"  FID   (down) : {results['FID']}")
    print(f"  SSIM  (up)   : {results['SSIM']}")
    print(f"  PSNR  (up)   : {results['PSNR']} dB")
    print("=========================================\n")

    out = os.path.join(args.pred_dir, "metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] saved {out}")


if __name__ == "__main__":
    main()
