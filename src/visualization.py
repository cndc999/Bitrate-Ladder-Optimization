"""
Module 4 — Visualization
Pipeline step: measured data -> Bitrate-Quality Curves & supporting charts.
All matplotlib figure builders, separated from the UI.
"""

import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "font.size": 10, "figure.facecolor": "white",
})

BLUE, PURPLE, GREEN, AMBER, GRAY = "#2563eb", "#7c3aed", "#16a34a", "#d97706", "#94a3b8"


def plot_target_vs_actual(results: list[dict]):
    """Bar chart: target vs actual (ffprobe) bitrate for every encode."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = range(len(results))
    ax.bar([i - 0.2 for i in x], [r["target_bitrate"] for r in results],
           0.4, label="Target", color=GRAY)
    ax.bar([i + 0.2 for i in x], [r["actual_bitrate"] for r in results],
           0.4, label="Actual (ffprobe)", color=BLUE)
    ax.set_xticks(list(x),
                  [f'{r["target_bitrate"]}k\n{r["resolution"]}' for r in results])
    ax.set_ylabel("Bitrate (kbps)")
    ax.set_title("Target vs actual bitrate per encode")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_quality_curves(results: list[dict], optimized: list[dict]):
    """The core output charts: PSNR vs bitrate and SSIM vs bitrate."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, name in [(axes[0], "psnr", "PSNR (dB)"),
                          (axes[1], "ssim", "SSIM")]:
        ax.plot([r["actual_bitrate"] for r in results],
                [r[key] for r in results],
                "o-", color=BLUE, label="All encodes")
        ax.scatter([r["actual_bitrate"] for r in optimized],
                   [r[key] for r in optimized],
                   marker="*", s=240, color=GREEN, zorder=3,
                   label="Selected ladder rungs")
        ax.set_xlabel("Actual bitrate (kbps)")
        ax.set_title(f"{name} vs bitrate")
    axes[0].legend()
    fig.tight_layout()
    return fig


def plot_bitrate_filesize(results: list[dict]):
    """Bitrate vs file size — higher bitrate, larger file."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot([r["actual_bitrate"] for r in results],
            [r["size_kb"] for r in results], "o-", color=PURPLE)
    for r in results:
        ax.annotate(r["resolution"], (r["actual_bitrate"], r["size_kb"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xlabel("Actual bitrate (kbps)")
    ax.set_ylabel("File size (KB)")
    ax.set_title("Bitrate vs file size")
    fig.tight_layout()
    return fig


def plot_ladder_comparison(pairs: list[tuple[dict, dict]]):
    """Grouped bars per MATCHED resolution: standard vs optimized bitrate."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    n = len(pairs)
    ax.bar([i - 0.2 for i in range(n)],
           [s["actual_bitrate"] for _, s in pairs], 0.4,
           label="Standard ladder", color=GRAY)
    ax.bar([i + 0.2 for i in range(n)],
           [o["actual_bitrate"] for o, _ in pairs], 0.4,
           label="Optimized ladder", color=GREEN)
    ax.set_xticks(range(n), [o["resolution"] for o, _ in pairs])
    ax.set_ylabel("Bitrate (kbps)")
    ax.set_title("Standard vs optimized — matched resolutions only")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_simulation(ev_opt: list[dict], ev_std: list[dict], bw: list[float]):
    """Two rows: selected bitrate over time (ABR switching) and buffer health."""
    t = [e["time"] for e in ev_opt]
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)

    axA.fill_between(range(len(bw)), bw, color=GRAY, alpha=0.25, label="Bandwidth")
    axA.step(t, [e["bitrate"] for e in ev_opt], where="post",
             color=GREEN, lw=2, label="Optimized ladder")
    axA.step(t, [e["bitrate"] for e in ev_std], where="post",
             color=AMBER, lw=2, ls="--", label="Standard ladder")
    axA.set_ylabel("kbps")
    axA.set_title("Selected bitrate over time (ABR switching)")
    axA.legend(loc="upper right")

    axB.fill_between(t, [e["buffer"] for e in ev_opt], color=GREEN, alpha=0.25)
    axB.plot(t, [e["buffer"] for e in ev_opt], color=GREEN, label="Buffer (optimized)")
    axB.plot(t, [e["buffer"] for e in ev_std], color=AMBER, ls="--",
             label="Buffer (standard)")
    axB.axhline(0.5, color="red", ls=":", label="Rebuffering threshold")
    axB.set_xlabel("Segment #")
    axB.set_ylabel("Buffer (s)")
    axB.set_title("Buffer health")
    axB.legend(loc="upper right")
    fig.tight_layout()
    return fig
