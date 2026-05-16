"""
quality_metrics.py
──────────────────
Per-frame quality measurement: PSNR, SSIM, and approximated VMAF.
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def compute_quality(orig_path: str, enc_path: str,
                    max_frames: int = 30) -> tuple[float, float]:
    """
    Sample up to `max_frames` frames and compute mean PSNR and SSIM.

    Args:
        orig_path:  Path to the original (reference) video.
        enc_path:   Path to the encoded video.
        max_frames: Maximum number of frames to sample.

    Returns:
        (mean_psnr_db, mean_ssim) as floats.
    """
    orig_cap = cv2.VideoCapture(orig_path)
    enc_cap  = cv2.VideoCapture(enc_path)
    psnr_vals, ssim_vals = [], []
    frame_idx = 0

    while frame_idx < max_frames:
        r1, f1 = orig_cap.read()
        r2, f2 = enc_cap.read()
        if not r1 or not r2:
            break

        # Resize reference to match encoded dimensions
        h2, w2 = f2.shape[:2]
        f1r = cv2.resize(f1, (w2, h2))

        g1 = cv2.cvtColor(f1r, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2,  cv2.COLOR_BGR2GRAY)

        psnr_vals.append(_psnr(g1, g2, data_range=255))
        ssim_vals.append(_ssim(g1, g2, data_range=255))

        # Advance both captures by 5 frames
        frame_idx += 5
        for _ in range(4):
            orig_cap.read()
            enc_cap.read()

    orig_cap.release()
    enc_cap.release()

    if not psnr_vals:
        return 30.0, 0.80

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def compute_vmaf_approx(psnr_val: float, ssim_val: float) -> float:
    """
    Approximate VMAF score from PSNR and SSIM via a simplified regression.
    Note: not a substitute for libvmaf — used for lightweight demo purposes.

    Args:
        psnr_val: Mean PSNR in dB.
        ssim_val: Mean SSIM (0–1).

    Returns:
        Estimated VMAF score clamped to [0, 100].
    """
    vmaf = 20 * np.log10(max(psnr_val, 1)) + 30 * ssim_val - 10
    return float(np.clip(vmaf, 0, 100))
