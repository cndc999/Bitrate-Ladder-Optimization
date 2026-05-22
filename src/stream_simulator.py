"""
stream_simulator.py
───────────────────
Buffer-based ABR (Adaptive Bitrate) streaming simulation.
Models how a player selects ladder rungs as bandwidth fluctuates over time.
"""

from __future__ import annotations
import numpy as np


# ── Bandwidth presets ──────────────────────────────────────────────────────────

BANDWIDTH_PROFILES: dict[str, list[float]] = {
    "Stable": [5000] * 60,
    "Moderate Fluctuation":    [5000, 4500, 4000, 3000, 3500, 4000, 5000, 4500, 3000, 2500] * 6,
    "Weak & Intermittent": [2000, 1000, 500, 200, 1500, 3000, 500, 200, 1000, 2000] * 6,
}


# ── Core simulator ─────────────────────────────────────────────────────────────

def simulate_streaming(ladder_rungs: list[dict],
                       bandwidth_profile: list[float],
                       headroom: float = 0.80,
                       buffer_max: float = 30.0) -> list[dict]:
    """
    Simulate a buffer-based ABR algorithm over a bandwidth timeline.

    At each segment the player picks the highest-quality rung whose bitrate
    fits within `headroom * available_bandwidth`.  Buffer health is tracked;
    falling below 0.5 s counts as a rebuffering event.

    Args:
        ladder_rungs:      List of rung dicts (must contain "actual_bitrate",
                           "psnr", "ssim", "label").
        bandwidth_profile: Sequence of available bandwidth values (kbps),
                           one per simulated segment.
        headroom:          Safety margin: only use this fraction of bandwidth
                           (default 0.80 → 80 %).
        buffer_max:        Maximum buffer size in seconds (default 30 s).

    Returns:
        List of per-segment event dicts with keys:
            time, bandwidth, selected_bitrate, quality_label,
            psnr, ssim, buffer_health, rebuffering.
    """
    events: list[dict] = []
    buffer: float = 0.0

    sorted_rungs = sorted(ladder_rungs, key=lambda r: r["actual_bitrate"])

    for t, bw in enumerate(bandwidth_profile):
        # Best rung that fits within headroom budget
        available = [r for r in sorted_rungs
                     if r["actual_bitrate"] <= bw * headroom]
        chosen = available[-1] if available else sorted_rungs[0]

        # Buffer dynamics: gain 1 s per segment, spend bitrate ratio
        buffer += bw / max(chosen["actual_bitrate"], 1) - 1
        buffer = float(np.clip(buffer, 0, buffer_max))

        events.append({
            "time":             t,
            "bandwidth":        bw,
            "selected_bitrate": chosen["actual_bitrate"],
            "quality_label":    chosen.get("label", "?"),
            "psnr":             chosen["psnr"],
            "ssim":             chosen["ssim"],
            "buffer_health":    buffer,
            "rebuffering":      buffer < 0.5,
        })

    return events


def streaming_kpis(events: list[dict]) -> dict:
    """
    Summarise a simulation run into headline KPIs.

    Returns:
        Dict with avg_bitrate, avg_psnr, avg_ssim,
        rebuffer_count, rebuffer_rate.
    """
    if not events:
        return {}
    return {
        "avg_bitrate":    float(np.mean([e["selected_bitrate"] for e in events])),
        "avg_psnr":       float(np.mean([e["psnr"]             for e in events])),
        "avg_ssim":       float(np.mean([e["ssim"]             for e in events])),
        "rebuffer_count": int(sum(e["rebuffering"]             for e in events)),
        "rebuffer_rate":  float(np.mean([e["rebuffering"]      for e in events])) * 100,
    }
