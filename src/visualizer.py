"""
visualizer.py
─────────────
Plotly figure builders — pure data-in / figure-out, no Streamlit imports.
All functions return a plotly.graph_objects.Figure ready for st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared theme ───────────────────────────────────────────────────────────────

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0f172a",
)
_GRID = dict(gridcolor="#1e293b")
_COLORS = {
    "psnr":        "#38bdf8",
    "ssim":        "#a78bfa",
    "vmaf_approx": "#f472b6",
    "optimal":     "#4ade80",
    "standard":    "#64748b",
    "bandwidth":   "#64748b",
    "opt_line":    "#4ade80",
    "std_line":    "#f59e0b",
    "buffer":      "#4ade80",
}


# ── 1. Bitrate-Quality Curves ──────────────────────────────────────────────────

def plot_quality_curves(quality_data: list[dict],
                        optimized: list[dict]) -> go.Figure:
    """
    Three-panel plot: PSNR / SSIM / VMAF vs actual bitrate.
    Optimal ladder rungs are overlaid as green stars.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("PSNR (dB)", "SSIM", "VMAF (approx)"),
    )
    metrics = [
        ("psnr",        "PSNR",  _COLORS["psnr"],        1),
        ("ssim",        "SSIM",  _COLORS["ssim"],        2),
        ("vmaf_approx", "VMAF≈", _COLORS["vmaf_approx"], 3),
    ]
    for m_key, m_name, m_color, col_n in metrics:
        xs = [d["actual_bitrate"] for d in quality_data]
        ys = [d[m_key]            for d in quality_data]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=f"{m_name} (all)",
            line=dict(color=m_color, width=2), marker=dict(size=6),
            showlegend=(col_n == 1),
            hovertemplate=f"<b>{m_name}</b>: %{{y:.2f}}<br>Bitrate: %{{x:.0f}} kbps<extra></extra>",
        ), row=1, col=col_n)

        ox = [d["actual_bitrate"] for d in optimized]
        oy = [d[m_key]            for d in optimized]
        fig.add_trace(go.Scatter(
            x=ox, y=oy, mode="markers", name="Optimal rungs",
            marker=dict(color=_COLORS["optimal"], size=12, symbol="star",
                        line=dict(color="white", width=1.5)),
            showlegend=(col_n == 1),
            hovertemplate="<b>★ Optimal Rung</b><br>Bitrate: %{x:.0f} kbps<extra></extra>",
        ), row=1, col=col_n)

    fig.update_layout(**_DARK, height=400,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.22),
                      margin=dict(t=40, b=40))
    fig.update_xaxes(title_text="Bitrate (kbps)", **_GRID)
    fig.update_yaxes(**_GRID)
    return fig


# ── 2. Efficiency Heatmap ──────────────────────────────────────────────────────

def plot_efficiency_heatmap(quality_data: list[dict]) -> go.Figure:
    """
    Bar chart of PSNR-per-Mbps efficiency at each target bitrate.
    """
    df = pd.DataFrame(quality_data)
    df["efficiency"] = df["psnr"] / (df["actual_bitrate"] / 1000).clip(lower=0.01)
    fig = px.bar(
        df, x="target_bitrate", y="efficiency",
        color="efficiency",
        color_continuous_scale=["#1e3a5f", "#38bdf8", "#4ade80"],
        labels={"target_bitrate": "Target Bitrate (kbps)", "efficiency": "PSNR / Mbps"},
        **_DARK,
    )
    fig.update_layout(height=280, margin=dict(t=20, b=40),
                      coloraxis_colorbar=dict(title="Efficiency"))
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


# ── 3. Ladder Comparison (bar + scatter) ──────────────────────────────────────

def plot_ladder_comparison(quality_data: list[dict],
                           optimized: list[dict],
                           standard_ladder: list[dict]) -> tuple[go.Figure, go.Figure]:
    """
    Returns two figures:
        fig_bar     — grouped bar: standard vs optimised bitrates per rung label
        fig_scatter — PSNR scatter with optimal stars annotated
    """
    # Grouped bar
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Standard Ladder",
        x=[r["label"]   for r in standard_ladder],
        y=[r["bitrate"] for r in standard_ladder],
        marker_color=_COLORS["standard"], opacity=0.7,
    ))
    fig_bar.add_trace(go.Bar(
        name="Optimized Ladder",
        x=[r["label"]          for r in optimized],
        y=[r["actual_bitrate"] for r in optimized],
        marker_color=_COLORS["optimal"], opacity=0.9,
    ))
    fig_bar.update_layout(
        **_DARK, barmode="group", height=320,
        margin=dict(t=20, b=40), yaxis_title="Bitrate (kbps)",
        legend=dict(orientation="h", y=-0.22),
    )
    fig_bar.update_xaxes(**_GRID)
    fig_bar.update_yaxes(**_GRID)

    # PSNR scatter
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=[d["actual_bitrate"] for d in quality_data],
        y=[d["psnr"]           for d in quality_data],
        mode="markers", name="All encoded",
        marker=dict(
            color=[d["ssim"] for d in quality_data],
            colorscale="Viridis", size=8,
            colorbar=dict(title="SSIM"), showscale=True,
        ),
        hovertemplate="Bitrate: %{x:.0f} kbps<br>PSNR: %{y:.2f} dB<extra></extra>",
    ))
    fig_scatter.add_trace(go.Scatter(
        x=[d["actual_bitrate"] for d in optimized],
        y=[d["psnr"]           for d in optimized],
        mode="markers+text", name="★ Optimal",
        marker=dict(color=_COLORS["vmaf_approx"], size=16, symbol="star"),
        text=[d["label"] for d in optimized],
        textposition="top center",
        textfont=dict(color="white", size=10),
    ))
    fig_scatter.update_layout(
        **_DARK, height=350, margin=dict(t=20, b=40),
        xaxis_title="Actual Bitrate (kbps)", yaxis_title="PSNR (dB)",
    )
    fig_scatter.update_xaxes(**_GRID)
    fig_scatter.update_yaxes(**_GRID)

    return fig_bar, fig_scatter


# ── 4. Streaming Simulation ────────────────────────────────────────────────────

def plot_streaming_simulation(events_opt: list[dict],
                              events_std: list[dict],
                              bw_profile: list[float]) -> go.Figure:
    """
    Three-row subplot:
        Row 1 — Bandwidth vs selected bitrate (optimized & standard)
        Row 2 — PSNR timeline (optimized vs standard)
        Row 3 — Buffer health (optimized)
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=(
            "Bandwidth vs Selected Bitrate",
            "PSNR Over Time",
            "Buffer Health (s)",
        ),
    )
    t = list(range(len(bw_profile)))

    # Row 1
    fig.add_trace(go.Scatter(
        x=t, y=bw_profile, name="Bandwidth",
        line=dict(color=_COLORS["bandwidth"], width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(100,116,139,0.10)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[e["time"] for e in events_opt],
        y=[e["selected_bitrate"] for e in events_opt],
        name="Optimized Ladder",
        line=dict(color=_COLORS["opt_line"], width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[e["time"] for e in events_std],
        y=[e["selected_bitrate"] for e in events_std],
        name="Standard Ladder",
        line=dict(color=_COLORS["std_line"], width=2, dash="dash"),
    ), row=1, col=1)

    # Row 2
    fig.add_trace(go.Scatter(
        x=[e["time"] for e in events_opt],
        y=[e["psnr"]            for e in events_opt],
        name="PSNR (Opt)", showlegend=False,
        line=dict(color=_COLORS["psnr"], width=2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[e["time"] for e in events_std],
        y=[e["psnr"]            for e in events_std],
        name="PSNR (Std)", showlegend=False,
        line=dict(color=_COLORS["vmaf_approx"], width=2, dash="dash"),
    ), row=2, col=1)

    # Row 3
    fig.add_trace(go.Scatter(
        x=[e["time"]          for e in events_opt],
        y=[e["buffer_health"] for e in events_opt],
        name="Buffer (Opt)", showlegend=False,
        fill="tozeroy", fillcolor="rgba(74,222,128,0.15)",
        line=dict(color=_COLORS["buffer"], width=2),
    ), row=3, col=1)

    fig.update_layout(
        **_DARK, height=560,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", y=-0.07),
    )
    fig.update_yaxes(**_GRID)
    fig.update_xaxes(title_text="Segment #", **_GRID, row=3, col=1)
    return fig
