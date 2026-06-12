"""
Module 2 — Quality Evaluation
Pipeline step: Encoded videos -> PSNR / SSIM scores.
Compares sampled frames of each encode against the source.
  PSNR (dB): pixel fidelity, higher is better (~>35 dB is good).
  SSIM (0-1): structural similarity, closer to 1 is better.
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def compute_psnr_ssim(orig_path: str, enc_path: str,
                      n_samples: int = 20, step: int = 5) -> tuple[float, float]:
    """Sample n_samples frames (step frames apart) and return mean PSNR/SSIM."""
    cap_o = cv2.VideoCapture(orig_path)
    cap_e = cv2.VideoCapture(enc_path)
    psnr_vals, ssim_vals = [], []

    for _ in range(n_samples):
        ok1, f1 = cap_o.read()
        ok2, f2 = cap_e.read()
        if not (ok1 and ok2):
            break
        # Fair cross-resolution comparison: upscale the encode back to the
        # SOURCE resolution and measure there (per-title encoding practice).
        # Low-res rungs then correctly show their upscaling blur instead of
        # being graded on an easier, downscaled reference.
        h, w = f1.shape[:2]
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(f2, (w, h)), cv2.COLOR_BGR2GRAY)
        psnr_vals.append(_psnr(g1, g2, data_range=255))
        ssim_vals.append(_ssim(g1, g2, data_range=255))
        for _ in range(step - 1):
            cap_o.read()
            cap_e.read()

    cap_o.release()
    cap_e.release()
    if not psnr_vals:
        return 0.0, 0.0
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))
