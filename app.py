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
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }

.main-title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.6rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1px; margin-bottom: 0;
}
.subtitle {
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    color: #64748b; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 1.2rem 1.5rem; text-align: center;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-label {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    color: #64748b; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px;
}
.metric-value { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.8rem; color: #38bdf8; }
.metric-unit  { font-size: 0.75rem; color: #94a3b8; }

.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 4px; text-transform: uppercase; color: #38bdf8;
    border-bottom: 1px solid #1e3a5f; padding-bottom: 8px; margin: 1.5rem 0 1rem;
}
.rung-chip {
    display: inline-block; background: #1e3a5f; border: 1px solid #38bdf8;
    border-radius: 6px; padding: 4px 12px; margin: 4px;
    font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #38bdf8;
}
.optimal-chip { background: #14532d; border-color: #4ade80; color: #4ade80; }
.info-box {
    background: #0f1e35; border-left: 3px solid #38bdf8;
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: #94a3b8; margin: 0.5rem 0;
}
div[data-testid="stSidebar"] { background: #060b14; border-right: 1px solid #1e293b; }
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white;
    border: none; border-radius: 8px; font-family: 'Syne', sans-serif;
    font-weight: 600; letter-spacing: 1px; padding: 0.6rem 2rem; width: 100%;
}
.stButton > button:hover { opacity: 0.85; }
.stProgress > div > div { background: linear-gradient(90deg, #38bdf8, #818cf8); }
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
    st.markdown('<div class="main-title" style="font-size:1.4rem;">⚡ BLO</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Bitrate Ladder Optimizer</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">📂 Video Input</div>', unsafe_allow_html=True)
    input_mode = st.radio("Nguồn video", ["🎲 Synthetic (tự tạo)", "📁 Upload file"], label_visibility="collapsed")
    uploaded = None
    if input_mode == "📁 Upload file":
        uploaded = st.file_uploader("Upload video", type=["mp4", "mkv", "avi", "mov"],
                                    label_visibility="collapsed")

    st.markdown('<div class="section-header">⚙️ Encoding Config</div>', unsafe_allow_html=True)
    codec_label = st.selectbox("Codec", ["H.264 (libx264)", "H.265 (libx265)"])
    codec_lib   = "libx264" if "264" in codec_label else "libx265"
    n_rungs     = st.slider("Số rungs trong ladder", 3, 7, 5)

    st.markdown('<div class="section-header">📊 Quality Metric</div>', unsafe_allow_html=True)
    metric = st.selectbox("Primary metric", ["psnr", "ssim", "vmaf_approx"])

    st.markdown('<div class="section-header">📡 Streaming Sim</div>', unsafe_allow_html=True)
    bw_scenario = st.selectbox("Kịch bản băng thông",
                                list(BANDWIDTH_PROFILES.keys()) + ["Tùy chỉnh"])
    custom_bw_input = ""
    if bw_scenario == "Tùy chỉnh":
        custom_bw_input = st.text_input("Bandwidth (kbps, cách nhau dấu phẩy)",
                                        "5000,3000,1500,4000,2000,5000")

    st.markdown("---")
    run_btn = st.button("🚀 RUN ANALYSIS", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">Bitrate Ladder Optimization</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Nén & Mã hóa Dữ liệu · Project 2502e · HUST</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="info-box"><b>Bitrate Ladder</b> là tập hợp các phiên bản video ở nhiều bitrate khác nhau, giúp HLS/DASH chọn chất lượng phù hợp với băng thông người xem.</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="info-box"><b>Convex Hull Optimization</b> chọn các điểm nằm trên Pareto frontier của đường cong bitrate-quality, loại bỏ các rungs không hiệu quả.</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="info-box"><b>ABR Streaming Simulation</b> mô phỏng thuật toán adaptive bitrate (HLS-like) khi băng thông thay đổi theo thời gian thực.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE BITRATE EXPLORER (always visible)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🎚️ Interactive Bitrate Explorer</div>', unsafe_allow_html=True)

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
# MAIN ANALYSIS (triggered by RUN button)
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    st.markdown("---")
    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Step 1: Source video ──────────────────────────────────────────────
        progress   = st.progress(0, text="⏳ Chuẩn bị video nguồn...")
        src_path   = os.path.join(tmpdir, "source.mp4")

        if input_mode == "📁 Upload file" and uploaded:
            with open(src_path, "wb") as f:
                f.write(uploaded.read())
            st.success(f"✅ Đã load: {uploaded.name}")
        else:
            with st.spinner("🎨 Tạo synthetic test video (1280×720, 3s)..."):
                make_synthetic_video(src_path)
            st.success("✅ Synthetic video ready (1280×720, 30 fps, 3 s)")

        # ── Step 2: Multi-bitrate encoding ───────────────────────────────────
        progress.progress(10, text="🔧 Encoding multiple bitrates...")
        quality_data   = []
        enc_status     = st.empty()
        enc_bar        = st.progress(0)

        for idx, br in enumerate(BITRATE_POOL):
            h, label   = resolution_for_bitrate(br)
            out_path   = os.path.join(tmpdir, f"enc_{br}k.mp4")

            enc_status.markdown(
                f'<div class="info-box">🎬 Encoding <b>{br} kbps @ {label}</b> [{idx+1}/{len(BITRATE_POOL)}]...</div>',
                unsafe_allow_html=True,
            )
            success = encode_with_ffmpeg(src_path, br, h, out_path, codec=codec_lib)
            enc_bar.progress((idx + 1) / len(BITRATE_POOL))

            if success and os.path.exists(out_path):
                p, s       = compute_quality(src_path, out_path)
                actual_br  = get_actual_bitrate(out_path) or br
                quality_data.append({
                    "target_bitrate": br,
                    "actual_bitrate": actual_br,
                    "height":         h,
                    "label":          label,
                    "psnr":           p,
                    "ssim":           s,
                    "vmaf_approx":    compute_vmaf_approx(p, s),
                    "encoded_path":   out_path,
                })

        enc_status.empty()
        enc_bar.empty()

        # ── Step 3: Ladder optimisation ───────────────────────────────────────
        progress.progress(70, text="📊 Computing Pareto frontier...")
        optimized = build_optimized_ladder(quality_data, n_rungs=n_rungs, metric=metric)

        # ── Step 4: Streaming simulation ──────────────────────────────────────
        progress.progress(85, text="📡 Running streaming simulation...")

        if bw_scenario == "Tùy chỉnh":
            try:
                seq        = [float(x) for x in custom_bw_input.split(",")]
                bw_profile = (seq * 20)[:60]
            except Exception:
                bw_profile = BANDWIDTH_PROFILES["Biến động vừa"]
        else:
            bw_profile = BANDWIDTH_PROFILES.get(bw_scenario, BANDWIDTH_PROFILES["Biến động vừa"])

        events_opt = simulate_streaming(optimized,    bw_profile)
        events_std = simulate_streaming(quality_data, bw_profile)

        progress.progress(100, text="✅ Hoàn tất!")
        time.sleep(0.4)
        progress.empty()

        # ─────────────────────────────────────────────────────────────────────
        # RESULT TABS
        # ─────────────────────────────────────────────────────────────────────

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Bitrate-Quality Curves",
            "🏗️ Ladder Comparison",
            "📡 Streaming Simulation",
            "🔬 Frame Analysis",
            "📋 Data Table",
        ])

        # ── Tab 1 ─────────────────────────────────────────────────────────────
        with tab1:
            st.markdown('<div class="section-header">BITRATE – QUALITY CURVES</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_quality_curves(quality_data, optimized), use_container_width=True)

            st.markdown('<div class="section-header">EFFICIENCY MAP (PSNR / Mbps)</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_efficiency_heatmap(quality_data), use_container_width=True)

        # ── Tab 2 ─────────────────────────────────────────────────────────────
        with tab2:
            st.markdown('<div class="section-header">STANDARD vs OPTIMIZED LADDER</div>', unsafe_allow_html=True)

            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown("**📺 Standard ABR Ladder**")
                for r in STANDARD_LADDER:
                    st.markdown(f'<span class="rung-chip">{r["label"]} · {r["bitrate"]} kbps</span>',
                                unsafe_allow_html=True)
                st.markdown(f"<br>**Tổng:** {sum(r['bitrate'] for r in STANDARD_LADDER):,} kbps",
                            unsafe_allow_html=True)

            with lc2:
                st.markdown(f"**⭐ Optimized Ladder** ({metric.upper()})")
                for r in optimized:
                    st.markdown(
                        f'<span class="rung-chip optimal-chip">'
                        f'{r["label"]} · {r["actual_bitrate"]:.0f} kbps · '
                        f'{metric.upper()}={r[metric]:.2f}</span>',
                        unsafe_allow_html=True,
                    )
                savings = bandwidth_savings(optimized, STANDARD_LADDER)
                total_opt = sum(r["actual_bitrate"] for r in optimized)
                st.markdown(f"<br>**Tổng:** {total_opt:,.0f} kbps", unsafe_allow_html=True)
                if savings > 0:
                    st.markdown(
                        f'<div class="info-box">💡 Tiết kiệm <b>{savings:.1f}%</b> băng thông '
                        f'so với Standard Ladder.</div>',
                        unsafe_allow_html=True,
                    )

            fig_bar, fig_scatter = plot_ladder_comparison(quality_data, optimized, STANDARD_LADDER)
            st.plotly_chart(fig_bar,     use_container_width=True)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Tab 3 ─────────────────────────────────────────────────────────────
        with tab3:
            st.markdown('<div class="section-header">ABR STREAMING SIMULATION</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="info-box">Kịch bản: <b>{bw_scenario}</b> · '
                f'Segments: {len(bw_profile)}</div>',
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
                (kc1, "Rebuffer Events (Opt)", str(kpis_opt["rebuffer_count"]),                      "events"),
                (kc2, "Avg PSNR Opt vs Std",   f"{kpis_opt['avg_psnr']:.1f} vs {kpis_std['avg_psnr']:.1f}", "dB"),
                (kc3, "Avg Bitrate (Opt)",      f"{kpis_opt['avg_bitrate']:.0f}",                   "kbps"),
                (kc4, "Quality Gain",           f"+{kpis_opt['avg_psnr'] - kpis_std['avg_psnr']:.1f}", "dB"),
            ]:
                with col:
                    st.markdown(_metric_card(lbl, val, unit), unsafe_allow_html=True)

        # ── Tab 4 ─────────────────────────────────────────────────────────────
        with tab4:
            st.markdown('<div class="section-header">PER-RUNG QUALITY ANALYSIS</div>', unsafe_allow_html=True)

            rung_labels = [f"{d['label']} @ {d['actual_bitrate']:.0f} kbps" for d in quality_data]
            sel_rung    = st.selectbox("Chọn rung", rung_labels)
            d_sel       = quality_data[rung_labels.index(sel_rung)]

            qc1, qc2, qc3, qc4 = st.columns(4)
            for col, lbl, val, unit in [
                (qc1, "Actual Bitrate", f"{d_sel['actual_bitrate']:.0f}", "kbps"),
                (qc2, "PSNR",          f"{d_sel['psnr']:.2f}",           "dB"),
                (qc3, "SSIM",          f"{d_sel['ssim']:.4f}",           ""),
                (qc4, "VMAF≈",         f"{d_sel['vmaf_approx']:.1f}",    "/100"),
            ]:
                with col:
                    st.markdown(_metric_card(lbl, val, unit), unsafe_allow_html=True)

            st.markdown('<div class="section-header">FRAME PREVIEW</div>', unsafe_allow_html=True)
            frame_src = extract_frame(src_path, 0)
            frame_enc = extract_frame(d_sel["encoded_path"], 0)

            if frame_src is not None and frame_enc is not None:
                fc1, fc2 = st.columns(2)
                with fc1:
                    st.markdown("**Original**")
                    st.image(cv2.cvtColor(frame_src, cv2.COLOR_BGR2RGB),
                             caption="Source (lossless)", use_column_width=True)
                with fc2:
                    st.markdown(f"**Encoded @ {d_sel['actual_bitrate']:.0f} kbps**")
                    st.image(cv2.cvtColor(frame_enc, cv2.COLOR_BGR2RGB),
                             caption=f"{d_sel['label']} · PSNR={d_sel['psnr']:.1f} dB · SSIM={d_sel['ssim']:.3f}",
                             use_column_width=True)

                st.markdown('<div class="section-header">DIFFERENCE MAP (×5 amplified)</div>', unsafe_allow_html=True)
                st.image(compute_diff_map(frame_src, frame_enc, amplify=5.0),
                         caption="Vùng sáng = mất mát thông tin nhiều hơn",
                         use_column_width=True)

        # ── Tab 5 ─────────────────────────────────────────────────────────────
        with tab5:
            st.markdown('<div class="section-header">RAW QUALITY DATA</div>', unsafe_allow_html=True)
            df = pd.DataFrame([{
                "Target Bitrate (kbps)":  d["target_bitrate"],
                "Actual Bitrate (kbps)":  round(d["actual_bitrate"], 1),
                "Resolution":             d["label"],
                "PSNR (dB)":              round(d["psnr"], 2),
                "SSIM":                   round(d["ssim"], 4),
                "VMAF≈":                  round(d["vmaf_approx"], 1),
                "Efficiency (dB/Mbps)":   round(d["psnr"] / max(d["actual_bitrate"] / 1000, 0.01), 2),
                "★ In Ladder":            "✅" if d in optimized else "",
            } for d in quality_data])

            st.dataframe(df, use_container_width=True, height=450,
                         column_config={
                             "PSNR (dB)": st.column_config.NumberColumn(format="%.2f dB"),
                             "SSIM":      st.column_config.ProgressColumn(min_value=0, max_value=1),
                         })
            st.download_button(
                "⬇️ Export CSV",
                df.to_csv(index=False).encode("utf-8"),
                "bitrate_ladder_results.csv",
                "text/csv",
            )

else:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:3rem; opacity:0.45;">
        <div style="font-size:3rem;">🎬</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.8rem;
                    letter-spacing:3px; color:#64748b; margin-top:1rem;">
            CONFIGURE SETTINGS IN SIDEBAR → CLICK RUN ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace;
            font-size:0.65rem; color:#334155; letter-spacing:2px;">
    PROJECT 2502e · BITRATE LADDER OPTIMIZATION ·
    NGUYEN LE QUANG ANH 202414611 · NGUYEN DANG ANH DUNG 202414619
</div>
""", unsafe_allow_html=True)
