"""
Module 3 — Ladder Design
Pipeline step: Quality curve -> the BEST set of bitrates for THIS video.

Two ladders are compared:
  Standard ladder : one fixed bitrate set used for every video.
  Optimized ladder: rungs picked from the Pareto frontier (upper convex
  hull) of the measured bitrate-quality curve — the per-title encoding
  idea popularised by Netflix. A point is on the frontier when no other
  point is both cheaper (lower bitrate) AND better (higher quality).
"""

# Classic fixed ladder (kbps) — mapped to the nearest encoded candidate
STANDARD_BITRATES = [200, 500, 1000, 2500, 5000]


def standard_ladder(results: list[dict]) -> list[dict]:
    """Map each standard bitrate to the closest encode (to get PSNR/SSIM)."""
    rungs = []
    for sb in STANDARD_BITRATES:
        closest = min(results, key=lambda r: abs(r["target_bitrate"] - sb))
        if closest not in rungs:
            rungs.append(closest)
    return sorted(rungs, key=lambda r: r["actual_bitrate"])


def _upper_convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper convex hull via monotone chain (pure Python, no scipy)."""
    pts = sorted(points)
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    # Keep only the non-decreasing part = the true Pareto frontier
    pareto = [hull[0]]
    for p in hull[1:]:
        if p[1] >= pareto[-1][1]:
            pareto.append(p)
    return pareto


def optimized_ladder(results: list[dict], n_rungs: int = 5,
                     metric: str = "psnr") -> list[dict]:
    """
    Select n_rungs rungs from the Pareto frontier of (actual bitrate, quality):
      1. Compute the upper convex hull of all measured points.
      2. Take n_rungs evenly spaced points along the frontier.
      3. Map each back to the nearest encoded result.
    """
    if len(results) <= n_rungs:
        return sorted(results, key=lambda r: r["actual_bitrate"])

    pts = [(r["actual_bitrate"], r[metric]) for r in results]
    pareto = _upper_convex_hull(pts)

    n = min(n_rungs, len(pareto))
    idxs = [round(i * (len(pareto) - 1) / (n - 1)) for i in range(n)]
    chosen = [pareto[i][0] for i in idxs]

    rungs = []
    for br in chosen:
        closest = min(results, key=lambda r: abs(r["actual_bitrate"] - br))
        if closest not in rungs:
            rungs.append(closest)
    return sorted(rungs, key=lambda r: r["actual_bitrate"])


def bandwidth_savings(optimized: list[dict], standard: list[dict]) -> float:
    """% bandwidth saved: total rung bitrate of optimized vs standard ladder."""
    total_opt = sum(r["actual_bitrate"] for r in optimized)
    total_std = sum(r["actual_bitrate"] for r in standard)
    return (1 - total_opt / total_std) * 100 if total_std else 0.0
