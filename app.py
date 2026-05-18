"""
app.py
──────
Main Streamlit UI — layout, widgets, and orchestration only.
All business logic lives in src/.
"""

import os
import time
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import (
    encode_with_ffmpeg,
    get_actual_bitrate,
    compute_quality,
    compute_vmaf_approx,
    build_optimized_ladder,
    BITRATE_POOL,
    STANDARD_LADDER,
    simulate_streaming,
    make_synthetic_video,
    extract_frame,
    compute_diff_map,
    plot_quality_curves,
    plot_efficiency_heatmap,
    plot_ladder_comparison,
    plot_streaming_simulation,
)
from src.encoder import resolution_for_bitrate
from src.stream_simulator import BANDWIDTH_PROFILES, streaming_kpis
from src.ladder_optimizer import bandwidth_savings

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bitrate Ladder Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background & global text ── */
.stApp { background: #f8fafc; color: #0f172a; }

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #cbd5e1;
    box-shadow: 2px 0 12px rgba(0,0,0,0.03);
}
div[data-testid="stSidebar"] * { color: #0f172a !important; }

/* ── Main title ── */
.main-title {
    font-family: 'Inter', sans-serif;
    font-weight: 800; font-size: 2.4rem;
    color: #1e3a8a; /* High contrast dark blue */
    letter-spacing: -1px; margin-bottom: 0;
}
.subtitle {
    font-family: 'Fira Code', monospace; font-size: 0.75rem;
    color: #475569; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 2rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    transition: border-color 0.2s;
}
.metric-card:hover {
    border-color: #2563eb;
}
.metric-label {
    font-family: 'Fira Code', monospace; font-size: 0.7rem;
    color: #475569; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 6px;
    font-weight: 600;
}
.metric-value {
    font-family: 'Inter', sans-serif;
    font-weight: 800; font-size: 1.8rem; color: #2563eb;
}
.metric-unit { font-size: 0.8rem; color: #64748b; font-weight: 500; }

/* ── Section headers ── */
.section-header {
    font-family: 'Fira Code', monospace; font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase; color: #1e40af;
    border-bottom: 2px solid #cbd5e1;
    padding-bottom: 8px; margin: 1.8rem 0 1rem;
}

/* ── Info boxes ── */
.info-box {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #2563eb;
    padding: 1rem; border-radius: 0 8px 8px 0;
    font-size: 0.9rem; color: #0f172a; margin: 0.5rem 0;
    line-height: 1.6;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* ── Rung chips ── */
.rung-chip {
    display: inline-block;
    background: #f1f5f9; border: 1px solid #cbd5e1;
    border-radius: 4px; padding: 6px 12px; margin: 4px;
    font-family: 'Fira Code', monospace;
    font-size: 0.75rem; color: #0f172a;
    font-weight: 500;
}
.optimal-chip {
    background: #ecfdf5; border-color: #6ee7b7; color: #065f46;
}

/* ── Run button ── */
.stButton > button {
    background: #2563eb; /* Solid blue */
    color: white; border: none; border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 700; font-size: 0.95rem;
    letter-spacing: 0.5px; padding: 0.75rem 2rem;
    width: 100%; transition: background-color 0.2s;
}
.stButton > button:hover {
    background: #1d4ed8; /* Darker blue on hover */
    color: white;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: #2563eb;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    color: #475569 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1e40af !important;
}

/* ── Selectbox / slider labels ── */
label { color: #0f172a !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _metric_card(label: str, value: str, unit: str = "") -> str:
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.6rem;">BLO</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Bitrate Ladder Optimizer</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Video Input</div>', unsafe_allow_html=True)
    input_mode = st.radio("Video Source", ["Synthetic", "Upload File"], label_visibility="collapsed")
    uploaded = None
    if input_mode == "Upload File":
        uploaded = st.file_uploader("Upload video", type=["mp4", "mkv", "avi", "mov"],
                                    label_visibility="collapsed")

    st.markdown('<div class="section-header">Encoding Config</div>', unsafe_allow_html=True)
    codec_label = st.selectbox("Codec", ["H.264 (libx264)", "H.265 (libx265)"])
    codec_lib   = "libx264" if "264" in codec_label else "libx265"
    n_rungs     = st.slider("Number of rungs", 3, 7, 5)

    st.markdown('<div class="section-header">Quality Metric</div>', unsafe_allow_html=True)
    metric = st.selectbox("Primary metric", ["psnr", "ssmi", "vmaf_approx"])

    st.markdown('<div class="section-header">Streaming Sim</div>', unsafe_allow_html=True)
    bw_scenario = st.selectbox("Bandwidth Scenario",
                                list(BANDWIDTH_PROFILES.keys()) + ["Custom"])
    custom_bw_input = ""
    if bw_scenario == "Custom":
        custom_bw_input = st.text_input("Bandwidth (kbps, comma-separated)",
                                        "5000,3000,1500,4000,2000,5000")

    st.markdown("---")
    run_btn = st.button("RUN ANALYSIS", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">Bitrate Ladder Optimization</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Data Compression & Encoding · Project 2502e · HUST</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="info-box">A <b>Bitrate Ladder</b> is a set of video versions at various bitrates, allowing HLS/DASH to select the appropriate quality based on viewer bandwidth.</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="info-box"><b>Convex Hull Optimization</b> selects points on the Pareto frontier of the bitrate-quality curve, eliminating inefficient rungs.</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="info-box"><b>ABR Streaming Simulation</b> mimics adaptive bitrate algorithms under real-time bandwidth fluctuations.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE BITRATE EXPLORER (always visible)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Interactive Bitrate Explorer</div>', unsafe_allow_html=True)

sl1, sl2, sl3 = st.columns([2, 2, 1])
with sl1:
    sel_bitrate = st.slider("Target Bitrate (kbps)", 150, 12000, 2500, step=50,
                             format="%d kbps")
with sl2:
    sel_res = st.select_slider("Resolution",
                                options=["240p", "360p", "480p", "720p", "1080p", "1440p", "4K"],
                                value="720p")

res_map    = {"240p": 240, "360p": 360, "480p": 480, "720p": 720,
              "1080p": 1080, "1440p": 1440, "4K": 2160}
h_px       = res_map[sel_res]
bpp        = sel_bitrate / (h_px * (h_px * 16 / 9) * 30)
est_psnr   = float(np.clip(10 * np.log10(1 / max(bpp * 0.01, 1e-8)) * 0.3 + 25, 20, 48))
est_ssim   = float(np.clip(0.65 + 0.3 * (sel_bitrate / 12000) ** 0.4, 0.5, 0.99))

with sl3:
    st.markdown(_metric_card("Est. PSNR", f"{est_psnr:.1f}", "dB"), unsafe_allow_html=True)

mc1, mc2, mc3, mc4 = st.columns(4)
for col, lbl, val, unit in [
    (mc1, "Target Bitrate", f"{sel_bitrate:,}", "kbps"),
    (mc2, "Resolution",     sel_res,             ""),
    (mc3, "Est. PSNR",      f"{est_psnr:.1f}",   "dB"),
    (mc4, "Est. SSIM",      f"{est_ssim:.3f}",   ""),
]:
    with col:
        st.markdown(_metric_card(lbl, val, unit), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS (triggered by RUN button)
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    st.markdown("---")

    # Use a persistent temp dir stored in session_state
    if "tmpdir_obj" in st.session_state:
        st.session_state.tmpdir_obj.cleanup()
    tmpdir_obj = tempfile.TemporaryDirectory()
    st.session_state.tmpdir_obj = tmpdir_obj
    tmpdir = tmpdir_obj.name

    # ── Step 1: Source video ──────────────────────────────────────────────────
    progress = st.progress(0, text="Preparing source video...")
    src_path = os.path.join(tmpdir, "source.mp4")

    if input_mode == "Upload File" and uploaded:
        with open(src_path, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Successfully loaded: {uploaded.name}")
    else:
        with st.spinner("Generating synthetic test video (1280x720, 3s)..."):
            make_synthetic_video(src_path)
        st.success("Synthetic video ready (1280x720, 30 fps, 3 s)")

    # ── Step 2: Multi-bitrate encoding ───────────────────────────────────────
    progress.progress(10, text="Encoding multiple bitrates...")
    quality_data = []
    enc_status   = st.empty()
    enc_bar      = st.progress(0)

    # Store source frame in memory once
    src_frame = extract_frame(src_path, 0)

    for idx, br in enumerate(BITRATE_POOL):
        h, label = resolution_for_bitrate(br)
        out_path = os.path.join(tmpdir, f"enc_{br}k.mp4")

        enc_status.markdown(
            f'<div class="info-box">Encoding <b>{br} kbps @ {label}</b> [{idx+1}/{len(BITRATE_POOL)}]...</div>',
            unsafe_allow_html=True,
        )
        success = encode_with_ffmpeg(src_path, br, h, out_path, codec=codec_lib)
        enc_bar.progress((idx + 1) / len(BITRATE_POOL))

        if success and os.path.exists(out_path):
            p, s      = compute_quality(src_path, out_path)
            actual_br = get_actual_bitrate(out_path) or br
            enc_frame = extract_frame(out_path, 0)
            quality_data.append({
                "target_bitrate": br,
                "actual_bitrate": actual_br,
                "height":         h,
                "label":          label,
                "psnr":           p,
                "ssim":           s,
                "vmaf_approx":    compute_vmaf_approx(p, s),
                "frame":          enc_frame,   # stored in memory
            })

    enc_status.empty()
    enc_bar.empty()

    # ── Step 3: Ladder optimisation ───────────────────────────────────────────
    progress.progress(70, text="Computing Pareto frontier...")
    optimized = build_optimized_ladder(quality_data, n_rungs=n_rungs, metric=metric)

    # ── Step 4: Streaming simulation ──────────────────────────────────────────
    progress.progress(85, text="Running streaming simulation...")

    if bw_scenario == "Custom":
        try:
            seq        = [float(x) for x in custom_bw_input.split(",")]
            bw_profile = (seq * 20)[:60]
        except Exception:
            bw_profile = list(BANDWIDTH_PROFILES.values())[0]
    else:
        bw_profile = BANDWIDTH_PROFILES.get(bw_scenario, list(BANDWIDTH_PROFILES.values())[0])

    events_opt = simulate_streaming(optimized,    bw_profile)
    events_std = simulate_streaming(quality_data, bw_profile)

    progress.progress(100, text="Completed!")
    time.sleep(0.4)
    progress.empty()

    # ── Save everything to session_state ─────────────────────────────────────
    st.session_state.results = {
        "quality_data": quality_data,
        "optimized":    optimized,
        "events_opt":   events_opt,
        "events_std":   events_std,
        "bw_profile":   bw_profile,
        "bw_scenario":  bw_scenario,
        "metric":       metric,
        "src_frame":    src_frame,
    }

# ─────────────────────────────────────────────────────────────────────────────
# RESULT TABS — rendered from session_state (persists across widget interactions)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.results:
    st.markdown("---")
    r            = st.session_state.results
    quality_data = r["quality_data"]
    optimized    = r["optimized"]
    events_opt   = r["events_opt"]
    events_std   = r["events_std"]
    bw_profile   = r["bw_profile"]
    bw_scenario  = r["bw_scenario"]
    metric       = r["metric"]
    src_frame    = r["src_frame"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Bitrate-Quality Curves",
        "Ladder Comparison",
        "Streaming Simulation",
        "Frame Analysis",
        "Data Table",
    ])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">BITRATE - QUALITY CURVES</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_quality_curves(quality_data, optimized), use_container_width=True)

        st.markdown('<div class="section-header">EFFICIENCY MAP (PSNR / Mbps)</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_efficiency_heatmap(quality_data), use_container_width=True)

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">STANDARD VS OPTIMIZED LADDER</div>', unsafe_allow_html=True)

        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown("**Standard ABR Ladder**")
            for r_item in STANDARD_LADDER:
                st.markdown(f'<span class="rung-chip">{r_item["label"]} · {r_item["bitrate"]} kbps</span>',
                            unsafe_allow_html=True)
            st.markdown(f"<br>**Total:** {sum(r_item['bitrate'] for r_item in STANDARD_LADDER):,} kbps",
                        unsafe_allow_html=True)

        with lc2:
            st.markdown(f"**Optimized Ladder** ({metric.upper()})")
            for r_item in optimized:
                st.markdown(
                    f'<span class="rung-chip optimal-chip">'
                    f'{r_item["label"]} · {r_item["actual_bitrate"]:.0f} kbps · '
                    f'{metric.upper()}={r_item[metric]:.2f}</span>',
                    unsafe_allow_html=True,
                )
            savings   = bandwidth_savings(optimized, STANDARD_LADDER)
            total_opt = sum(r_item["actual_bitrate"] for r_item in optimized)
            st.markdown(f"<br>**Total:** {total_opt:,.0f} kbps", unsafe_allow_html=True)
            if savings > 0:
                st.markdown(
                    f'<div class="info-box">Saved <b>{savings:.1f}%</b> bandwidth '
                    f'compared to the Standard Ladder.</div>',
                    unsafe_allow_html=True,
                )

        fig_bar, fig_scatter = plot_ladder_comparison(quality_data, optimized, STANDARD_LADDER)
        st.plotly_chart(fig_bar,     use_container_width=True)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">ABR STREAMING SIMULATION</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-box">Scenario: <b>{bw_scenario}</b> · Segments: {len(bw_profile)}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_streaming_simulation(events_opt, events_std, bw_profile),
            use_container_width=True,
        )

        kpis_opt = streaming_kpis(events_opt)
        kpis_std = streaming_kpis(events_std)
        kc1, kc2, kc3, kc4 = st.columns(4)
        for col, lbl, val, unit in [
            (kc1, "Rebuffer Events (Opt)", str(kpis_opt["rebuffer_count"]),                            "events"),
            (kc2, "Avg PSNR Opt vs Std",   f"{kpis_opt['avg_psnr']:.1f} vs {kpis_std['avg_psnr']:.1f}", "dB"),
            (kc3, "Avg Bitrate (Opt)",      f"{kpis_opt['avg_bitrate']:.0f}",                         "kbps"),
            (kc4, "Quality Gain",           f"+{kpis_opt['avg_psnr'] - kpis_std['avg_psnr']:.1f}",    "dB"),
        ]:
            with col:
                st.markdown(_metric_card(lbl, val, unit), unsafe_allow_html=True)

    # ── Tab 4 ─────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">PER-RUNG QUALITY ANALYSIS</div>', unsafe_allow_html=True)

        rung_labels = [f"{d['label']} @ {d['actual_bitrate']:.0f} kbps" for d in quality_data]
        sel_rung    = st.selectbox("Select rung", rung_labels)
        d_sel       = quality_data[rung_labels.index(sel_rung)]

        qc1, qc2, qc3, qc4 = st.columns(4)
        for col, lbl, val, unit in [
            (qc1, "Actual Bitrate", f"{d_sel['actual_bitrate']:.0f}", "kbps"),
            (qc2, "PSNR",          f"{d_sel['psnr']:.2f}",           "dB"),
            (qc3, "SSIM",          f"{d_sel['ssim']:.4f}",           ""),
            (qc4, "VMAF Approx",   f"{d_sel['vmaf_approx']:.1f}",    "/100"),
        ]:
            with col:
                st.markdown(_metric_card(lbl, val, unit), unsafe_allow_html=True)

        st.markdown('<div class="section-header">FRAME PREVIEW</div>', unsafe_allow_html=True)
        frame_enc = d_sel.get("frame")

        if src_frame is not None and frame_enc is not None:
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown("**Original**")
                st.image(cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB),
                         caption="Source (lossless)", use_container_width=True)
            with fc2:
                st.markdown(f"**Encoded @ {d_sel['actual_bitrate']:.0f} kbps**")
                st.image(cv2.cvtColor(frame_enc, cv2.COLOR_BGR2RGB),
                         caption=f"{d_sel['label']} · PSNR={d_sel['psnr']:.1f} dB · SSIM={d_sel['ssim']:.3f}",
                         use_container_width=True)

            st.markdown('<div class="section-header">DIFFERENCE MAP (x5 amplified)</div>', unsafe_allow_html=True)
            st.image(compute_diff_map(src_frame, frame_enc, amplify=5.0),
                     caption="Brighter areas = higher information loss",
                     use_container_width=True)

    # ── Tab 5 ─────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">RAW QUALITY DATA</div>', unsafe_allow_html=True)
        df = pd.DataFrame([{
            "Target Bitrate (kbps)": d["target_bitrate"],
            "Actual Bitrate (kbps)": round(d["actual_bitrate"], 1),
            "Resolution":            d["label"],
            "PSNR (dB)":             round(d["psnr"], 2),
            "SSIM":                  round(d["ssim"], 4),
            "VMAF Approx":           round(d["vmaf_approx"], 1),
            "Efficiency (dB/Mbps)":  round(d["psnr"] / max(d["actual_bitrate"] / 1000, 0.01), 2),
            "In Ladder":             "Yes" if d in optimized else "",
        } for d in quality_data])

        st.dataframe(df, use_container_width=True, height=450,
                     column_config={
                         "PSNR (dB)": st.column_config.NumberColumn(format="%.2f dB"),
                         "SSIM":      st.column_config.ProgressColumn(min_value=0, max_value=1),
                     })
        st.download_button(
            "Export CSV",
            df.to_csv(index=False).encode("utf-8"),
            "bitrate_ladder_results.csv",
            "text/csv",
        )

else:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:3rem; opacity:0.45;">
        <div style="font-family:'Fira Code',monospace; font-size:0.8rem;
                    letter-spacing:3px; color:#475569; margin-top:1rem; font-weight:600;">
            CONFIGURE SETTINGS IN SIDEBAR → CLICK RUN ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Fira Code',monospace;
            font-size:0.7rem; color:#64748b; letter-spacing:1.5px; font-weight:500;">
    PROJECT 2502e · BITRATE LADDER OPTIMIZATION ·
    NGUYEN LE QUANG ANH 202414611 · NGUYEN DANG ANH DUNG 202414619
</div>
""", unsafe_allow_html=True)