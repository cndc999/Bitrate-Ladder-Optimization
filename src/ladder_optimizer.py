"""
ladder_optimizer.py
───────────────────
Convex-hull based Pareto frontier selection for bitrate ladder optimization.
"""

import numpy as np
from scipy.spatial import ConvexHull


# ── Constants ─────────────────────────────────────────────────────────────────

BITRATE_POOL = [150, 250, 400, 600, 900, 1200, 1800, 2500, 3500, 5000, 7000, 10000]

# Bitrate tối đa hợp lý cho từng resolution (kbps)
# Dưới ngưỡng này thì encode không có ý nghĩa chất lượng
RESOLUTION_BITRATE_CAP: dict[int, int] = {
    240:  500,
    360:  900,
    480:  1800,
    720:  4000,
    1080: 8000,
    1440: 12000,
    2160: 20000,  # 4K
}

# Bitrate tối thiểu hợp lý cho từng resolution (kbps)
# Dưới ngưỡng này chất lượng quá thấp, không có ý nghĩa
RESOLUTION_BITRATE_MIN: dict[int, int] = {
    240:  100,
    360:  200,
    480:  400,
    720:  800,
    1080: 2000,
    1440: 4000,
    2160: 8000,
}

STANDARD_LADDER = [
    {"label": "240p",  "height": 240,  "bitrate": 200},
    {"label": "360p",  "height": 360,  "bitrate": 500},
    {"label": "480p",  "height": 480,  "bitrate": 1000},
    {"label": "720p",  "height": 720,  "bitrate": 2500},
    {"label": "1080p", "height": 1080, "bitrate": 5000},
]


# ── Pool filter ───────────────────────────────────────────────────────────────

def filter_pool_by_resolution(height: int) -> list[int]:
    """
    Trả về danh sách bitrate trong BITRATE_POOL phù hợp với resolution đã chọn.
    Loại bỏ các bitrate quá cao (lãng phí) hoặc quá thấp (chất lượng vô nghĩa).

    Args:
        height: Chiều cao resolution (px), ví dụ 240, 720, 1080.

    Returns:
        Danh sách bitrate (kbps) đã lọc, luôn có ít nhất 3 phần tử.
    """
    cap = RESOLUTION_BITRATE_CAP.get(height, max(BITRATE_POOL))
    low = RESOLUTION_BITRATE_MIN.get(height, 0)
    filtered = [b for b in BITRATE_POOL if low <= b <= cap]
    # Đảm bảo luôn có ít nhất 3 điểm để convex hull hoạt động
    if len(filtered) < 3:
        filtered = sorted(BITRATE_POOL, key=lambda b: abs(b - cap))[:3]
        filtered = sorted(filtered)
    return filtered


# ── Core algorithm ─────────────────────────────────────────────────────────────

def build_optimized_ladder(quality_data: list[dict], n_rungs: int = 5,
                           metric: str = "psnr") -> list[dict]:
    """
    Select N rungs from the upper convex hull (Pareto frontier) of the
    bitrate–quality curve.

    Algorithm:
        1. Plot all encode points in (bitrate, quality) space.
        2. Compute the convex hull; keep only the upper frontier
           (monotonically increasing quality with bitrate).
        3. Sample N evenly-spaced points along that frontier.
        4. Map each chosen point back to the nearest entry in quality_data.

    Args:
        quality_data: List of dicts, each with keys:
                      "actual_bitrate", "psnr", "ssim", "vmaf_approx", ...
        n_rungs:      Number of ladder rungs to select (3–7 recommended).
        metric:       Quality metric to optimise on ("psnr"|"ssim"|"vmaf_approx").

    Returns:
        Subset of quality_data representing the optimal ladder rungs.
    """
    if len(quality_data) < 3:
        return quality_data

    pts = np.array([[d["actual_bitrate"], d[metric]] for d in quality_data])

    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        hull_pts = hull_pts[np.argsort(hull_pts[:, 0])]

        # Keep only the upper hull: strictly non-decreasing quality
        pareto = [hull_pts[0]]
        for p in hull_pts[1:]:
            if p[1] >= pareto[-1][1]:
                pareto.append(p)
        pareto = np.array(pareto)

        # Evenly-spaced sample along the frontier
        indices = np.linspace(0, len(pareto) - 1,
                              min(n_rungs, len(pareto)), dtype=int)
        chosen_bitrates = pareto[indices, 0]

        # Map back to original data points (nearest actual_bitrate)
        result = []
        for br in chosen_bitrates:
            closest = min(quality_data, key=lambda d: abs(d["actual_bitrate"] - br))
            if closest not in result:
                result.append(closest)
        return result

    except Exception:
        # Fallback: evenly spaced from the full pool
        step = max(1, len(quality_data) // n_rungs)
        return quality_data[::step][:n_rungs]


def bandwidth_savings(optimized: list[dict], standard: list[dict] = STANDARD_LADDER) -> float:
    """
    Estimate percentage bandwidth saved vs the standard ladder.

    Returns:
        Savings as a percentage (positive = optimized uses less bandwidth).
    """
    total_opt = sum(r["actual_bitrate"] for r in optimized)
    total_std = sum(r["bitrate"] for r in standard)
    if total_std == 0:
        return 0.0
    return (1 - total_opt / total_std) * 100
