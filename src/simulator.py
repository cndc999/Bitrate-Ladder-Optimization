"""
Module 5 — Streaming Simulation
Pipeline step: Designed ladder -> proof it works under real networks.
A buffer-based ABR player picks, for every segment, the highest rung
whose bitrate fits within 80% of the available bandwidth. Buffer level
is tracked; dropping below 0.5 s counts as a rebuffering (stall) event.
"""

import numpy as np

# Bandwidth traces (kbps), one value per segment
BANDWIDTH_PROFILES: dict[str, list[float]] = {
    "3-phase: good -> crash -> recover (5/1/3 Mbps)":
        [5000] * 10 + [1000] * 10 + [3000] * 10,
    "Stable network": [5000] * 60,
    "Moderate fluctuation":
        [5000, 4500, 4000, 3000, 3500, 4000, 5000, 4500, 3000, 2500] * 6,
    "Weak & intermittent":
        [2000, 1000, 500, 200, 1500, 3000, 500, 200, 1000, 2000] * 6,
}


def simulate_streaming(ladder: list[dict], bandwidth: list[float],
                       headroom: float = 0.8, buffer_max: float = 30.0) -> list[dict]:
    """Run the ABR simulation for one ladder over one bandwidth trace."""
    rungs = sorted(ladder, key=lambda r: r["actual_bitrate"])
    buffer, events = 0.0, []

    for t, bw in enumerate(bandwidth):
        fit = [r for r in rungs if r["actual_bitrate"] <= bw * headroom]
        chosen = fit[-1] if fit else rungs[0]
        # Buffer dynamics: download faster than playback -> buffer grows
        buffer += bw / max(chosen["actual_bitrate"], 1) - 1
        buffer = float(np.clip(buffer, 0, buffer_max))
        events.append({
            "time":        t,
            "bandwidth":   bw,
            "bitrate":     chosen["actual_bitrate"],
            "label":       chosen["label"],
            "psnr":        chosen["psnr"],
            "buffer":      buffer,
            "rebuffering": buffer < 0.5,
        })
    return events


def streaming_kpis(events: list[dict]) -> dict:
    """Headline KPIs: average bitrate, average PSNR, stall count."""
    return {
        "avg_bitrate":    float(np.mean([e["bitrate"] for e in events])),
        "avg_psnr":       float(np.mean([e["psnr"]    for e in events])),
        "rebuffer_count": int(sum(e["rebuffering"]    for e in events)),
    }
