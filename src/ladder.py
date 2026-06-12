"""
Module 3 — Ladder Design
Pipeline step: Quality curve -> the BEST set of bitrates for THIS video.

Two ladders are compared:
  Standard ladder : the classic fixed resolution/bitrate table used for
                    every video (240p@400 ... 1080p@5000). It is actually
                    encoded and measured, not approximated.
  Optimized ladder: one rung per resolution, picked from the Pareto
                    frontier (upper convex hull) of the measured
                    bitrate-quality curve — per-title encoding.

Comparison is per matched resolution: a rung in the optimized ladder is
compared only against the standard rung of the SAME resolution.
"""

# The classic fixed ladder: (height, label, bitrate kbps)
STANDARD_LADDER_SPEC = [
    (240,  "240p",  400),
    (360,  "360p",  800),
    (480,  "480p",  1500),
    (720,  "720p",  3000),
    (1080, "1080p", 5000),
]


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
    pareto = [hull[0]]
    for p in hull[1:]:
        if p[1] >= pareto[-1][1]:
            pareto.append(p)
    return pareto


def optimized_ladder(results: list[dict], n_rungs: int = 5,
                     metric: str = "psnr") -> list[dict]:
    """
    Select up to n_rungs rungs from the Pareto frontier of
    (actual bitrate, quality), keeping AT MOST ONE rung per resolution
    (the best-quality frontier point of that resolution).
    """
    pts = [(r["actual_bitrate"], r[metric]) for r in results]
    pareto = _upper_convex_hull(pts) if len(results) > 2 else pts

    n = min(n_rungs, len(pareto))
    idxs = [round(i * (len(pareto) - 1) / max(n - 1, 1)) for i in range(n)]
    chosen_bitrates = [pareto[i][0] for i in idxs]

    picked: dict[str, dict] = {}          # resolution -> best rung
    for br in chosen_bitrates:
        r = min(results, key=lambda d: abs(d["actual_bitrate"] - br))
        res = r["resolution"]
        if res not in picked or r[metric] > picked[res][metric]:
            picked[res] = r
    return sorted(picked.values(), key=lambda r: r["actual_bitrate"])


def matched_pairs(optimized: list[dict], standard: list[dict]) -> list[tuple[dict, dict]]:
    """Pair each optimized rung with the standard rung of the same resolution."""
    std_by_res = {r["resolution"]: r for r in standard}
    return [(o, std_by_res[o["resolution"]])
            for o in optimized if o["resolution"] in std_by_res]


def bandwidth_savings(optimized: list[dict], standard: list[dict]) -> float:
    """% bandwidth saved, computed ONLY over matched resolutions."""
    pairs = matched_pairs(optimized, standard)
    if not pairs:
        return 0.0
    tot_opt = sum(o["actual_bitrate"] for o, _ in pairs)
    tot_std = sum(s["actual_bitrate"] for _, s in pairs)
    return (1 - tot_opt / tot_std) * 100 if tot_std else 0.0
