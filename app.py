"""
2502e — Bitrate Ladder Optimization
Multimedia & Compression course project

Pipeline (strict order):
  Source video -> Encode at multiple bitrates -> PSNR/SSIM quality
  -> Bitrate-Quality Curves -> Ladder Design -> Streaming Simulation

Final output: the BEST set of bitrates for the uploaded video,
balancing quality against bandwidth.

Authors: Nguyen Le Quang Anh (202414611), Nguyen Dang Anh Dung (202414619)
Run:  streamlit run app.py   (FFmpeg must be installed)
"""

import os
import tempfile

import streamlit as st

from src.encoder import (CANDIDATE_BITRATES, encode_video, probe_actual_bitrate,
                         file_size_kb, resolution_for_bitrate)
from src.ladder import (STANDARD_LADDER_SPEC, optimized_ladder,
                        matched_pairs, bandwidth_savings)
from src.simulator import BANDWIDTH_PROFILES, simulate_streaming, streaming_kpis

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="2502e · Bitrate Ladder Optimization",
                   page_icon="", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.rung-chip {display:inline-block; padding:8px 18px; margin:4px 6px 4px 0;
            border-radius:999px; background:#dcfce7; color:#14532d;
            font-weight:700; font-size:1.05rem; border:1px solid #86efac;}
.credit {color:#64748b; font-size:0.95rem;}
</style>""", unsafe_allow_html=True)

st.title("Bitrate Ladder Optimization")
st.markdown('<p class="credit">Project <b>2502e</b> · Multimedia &amp; Compression'
            ' &nbsp;|&nbsp; Nguyen Le Quang Anh — 202414611 &nbsp;·&nbsp;'
            ' Nguyen Dang Anh Dung — 202414619</p>', unsafe_allow_html=True)
st.markdown("**Pipeline:** Source video → Encode at multiple bitrates → "
            "PSNR/SSIM → Bitrate–Quality Curves → Ladder Design → "
            "Streaming Simulation")

# ── Sidebar: source video & settings ──────────────────────────────────────────
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Source video",
                                    type=["mp4", "mov", "avi", "mkv"])
max_seconds = st.sidebar.slider("Seconds of video to encode (sample)", 3, 15, 5)
n_rungs = st.sidebar.slider("Rungs in the optimized ladder", 3, 6, 5)
run = st.sidebar.button("▶ Run analysis", type="primary",
                        disabled=uploaded is None)
st.sidebar.caption("Requires FFmpeg installed and available on PATH.")

if uploaded is None:
    st.info("⬅ Upload a source video in the sidebar, then press **Run analysis**.")

# ── Pipeline steps 1–2: encode all bitrates, measure PSNR/SSIM ────────────────
if run:
    from src.quality import compute_psnr_ssim   # lazy import: loads cv2/skimage

    workdir = tempfile.mkdtemp()
    src_path = os.path.join(workdir, uploaded.name)
    with open(src_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Jobs: 8 candidate encodes (for the quality curve) + the 5 rungs of the
    # classic standard ladder (so its PSNR/SSIM are measured, not approximated)
    jobs = [(br, *resolution_for_bitrate(br), "cand") for br in CANDIDATE_BITRATES]
    jobs += [(br, h, lbl, "std") for (h, lbl, br) in
             [(h, l, b) for (h, l, b) in STANDARD_LADDER_SPEC]]

    results, std_results, cache = [], [], {}
    progress = st.progress(0.0, text="Encoding...")
    for i, (br, height, label, kind) in enumerate(jobs):
        progress.progress(i / len(jobs), text=f"Encoding {br} kbps ({label})...")
        key = (height, br)
        if key in cache:
            entry = cache[key]
        else:
            out_path = os.path.join(workdir, f"enc_{height}p_{br}k.mp4")
            ok, err = encode_video(src_path, br, height, out_path, max_seconds)
            if not ok:
                st.error(f"Encoding at {br} kbps failed:\n\n```\n{err}\n```")
                st.stop()
            psnr, ssim = compute_psnr_ssim(src_path, out_path)
            entry = {
                "target_bitrate": br,
                "actual_bitrate": probe_actual_bitrate(out_path),
                "label":          label,
                "resolution":     f"{height}p",
                "size_kb":        file_size_kb(out_path),
                "psnr":           psnr,
                "ssim":           ssim,
                "path":           out_path,
            }
            cache[key] = entry
        (results if kind == "cand" else std_results).append(entry)
    progress.progress(1.0, text="Done!")
    st.session_state["results"] = results
    st.session_state["std_results"] = std_results

if "results" not in st.session_state:
    st.stop()
results = st.session_state["results"]
std_results = st.session_state["std_results"]

import pandas as pd   # lazy: only loaded once results exist
from src.visualization import (plot_target_vs_actual, plot_quality_curves,
                               plot_bitrate_filesize, plot_ladder_comparison,
                               plot_simulation)

# ── FINAL OUTPUT: the best bitrate set for this video ─────────────────────────
std = std_results
opt = optimized_ladder(results, n_rungs=n_rungs)
pairs = matched_pairs(opt, std)
st.success("✅ **Final output — the best bitrate set for this video** "
           "(balances quality vs bandwidth):")
st.markdown("".join(
    f'<span class="rung-chip">{r["resolution"]} @ {r["actual_bitrate"]:.0f} kbps</span>'
    for r in opt), unsafe_allow_html=True)
st.caption(f"Selected from the Pareto frontier of the measured quality curve · "
           f"saves {bandwidth_savings(opt, std):.1f}% bandwidth vs the standard ladder.")

# ── Tabs follow the pipeline strictly ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Encode Multiple Bitrates",
    "Quality (PSNR / SSIM)",
    "Bitrate–Quality Curves",
    "Ladder Design",
    "Streaming Simulation",
])

# ═══ TAB 1: ENCODE MULTIPLE BITRATES ══════════════════════════════════════════
with tab1:
    st.subheader("What bitrate was each video actually encoded at?")
    df = pd.DataFrame(results)
    st.dataframe(pd.DataFrame({
        "Target bitrate (kbps)": df["target_bitrate"],
        "ACTUAL bitrate (kbps)": df["actual_bitrate"].round(0),
        "Deviation (%)": ((df["actual_bitrate"] - df["target_bitrate"])
                          / df["target_bitrate"] * 100).round(1),
        "Resolution":     df["resolution"],
        "File size (KB)": df["size_kb"].round(0),
    }), use_container_width=True, hide_index=True)
    st.pyplot(plot_target_vs_actual(results))

    st.subheader("Side-by-side comparison of two encodes")
    labels = [f'{r["target_bitrate"]} kbps → actual {r["actual_bitrate"]:.0f} kbps '
              f'({r["resolution"]}, {r["size_kb"]:.0f} KB)' for r in results]
    colA, colB = st.columns(2)
    for col, default_idx, side in [(colA, 0, "A"), (colB, len(results) - 1, "B")]:
        with col:
            idx = st.selectbox(f"Encode {side}", range(len(results)),
                               index=default_idx,
                               format_func=lambda i: labels[i], key=f"vid_{side}")
            r = results[idx]
            if os.path.exists(r["path"]):
                with open(r["path"], "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.markdown(f'**Actual bitrate: `{r["actual_bitrate"]:.0f}` kbps** · '
                            f'{r["resolution"]} · {r["size_kb"]:.0f} KB')
                st.download_button(f'Download {r["target_bitrate"]}k encode',
                                   video_bytes,
                                   file_name=f'encoded_{r["target_bitrate"]}k.mp4',
                                   key=f"dl_{side}")
            else:
                st.warning("Encoded file no longer exists — run the analysis again.")

# ═══ TAB 2: QUALITY (PSNR / SSIM) ═════════════════════════════════════════════
with tab2:
    st.subheader("Per-encode quality vs the source video")
    st.markdown("- **PSNR (dB)** — pixel fidelity; higher is better, ~35 dB+ is good.\n"
                "- **SSIM (0–1)** — structural similarity; closer to 1 is better.")
    st.dataframe(pd.DataFrame({
        "Encode": [f'{r["target_bitrate"]}k ({r["resolution"]})' for r in results],
        "Actual bitrate (kbps)": df["actual_bitrate"].round(0),
        "PSNR (dB)": df["psnr"].round(2),
        "SSIM":      df["ssim"].round(4),
    }), use_container_width=True, hide_index=True)
    best = max(results, key=lambda r: r["psnr"])
    st.metric("Best PSNR measured",
              f'{best["psnr"]:.2f} dB',
              f'at {best["actual_bitrate"]:.0f} kbps ({best["resolution"]})',
              delta_color="off")

# ═══ TAB 3: BITRATE–QUALITY CURVES ════════════════════════════════════════════
with tab3:
    st.subheader("Quality curves — where does quality saturate?")
    st.pyplot(plot_quality_curves(results, opt))
    st.caption("Past the knee of the curve, extra bitrate buys almost no extra "
               "quality — that is exactly where ladder rungs stop being worth it. "
               "Green stars = rungs selected for the optimized ladder.")
    st.pyplot(plot_bitrate_filesize(results))

# ═══ TAB 4: LADDER DESIGN ═════════════════════════════════════════════════════
with tab4:
    st.subheader("The designed ladder — the project's final output")
    st.markdown("".join(
        f'<span class="rung-chip">{r["resolution"]} @ {r["actual_bitrate"]:.0f} kbps</span>'
        for r in opt), unsafe_allow_html=True)
    st.metric("Bandwidth saved vs standard ladder (same resolutions)",
              f"{bandwidth_savings(opt, std):.1f}%")

    st.markdown("**Head-to-head at matched resolutions** — each optimized rung "
                "is compared only against the standard rung of the same resolution:")
    st.dataframe(pd.DataFrame([{
        "Resolution":        o["resolution"],
        "Standard (kbps)":   round(s["actual_bitrate"]),
        "Optimized (kbps)":  round(o["actual_bitrate"]),
        "Bitrate change (%)": round((o["actual_bitrate"] / s["actual_bitrate"] - 1) * 100, 1),
        "Standard PSNR (dB)": round(s["psnr"], 2),
        "Optimized PSNR (dB)": round(o["psnr"], 2),
    } for o, s in pairs]), use_container_width=True, hide_index=True)
    st.pyplot(plot_ladder_comparison(pairs))

    with st.expander("Full standard ladder (all 5 rungs, measured)"):
        st.table(pd.DataFrame([{"Rung": r["label"],
                                "Target (kbps)": r["target_bitrate"],
                                "Actual (kbps)": round(r["actual_bitrate"]),
                                "PSNR (dB)": round(r["psnr"], 2),
                                "SSIM": round(r["ssim"], 4)} for r in std]))
    st.caption("Method: rungs are picked from the upper convex hull (Pareto "
               "frontier) of the measured bitrate–quality curve, so every rung "
               "is the best quality available at its cost — the per-title "
               "encoding approach.")

# ═══ TAB 5: STREAMING SIMULATION ══════════════════════════════════════════════
with tab5:
    st.subheader("Does the designed ladder survive a real network?")
    profile_name = st.selectbox("Bandwidth trace", list(BANDWIDTH_PROFILES))
    bw = BANDWIDTH_PROFILES[profile_name]

    ev_std = simulate_streaming(std, bw)
    ev_opt = simulate_streaming(opt, bw)
    k_std, k_opt = streaming_kpis(ev_std), streaming_kpis(ev_opt)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg bitrate (Opt vs Std)", f'{k_opt["avg_bitrate"]:.0f} kbps',
              f'{k_opt["avg_bitrate"] - k_std["avg_bitrate"]:+.0f} kbps')
    c2.metric("Avg PSNR (Opt vs Std)", f'{k_opt["avg_psnr"]:.2f} dB',
              f'{k_opt["avg_psnr"] - k_std["avg_psnr"]:+.2f} dB')
    c3.metric("Rebuffering events (Opt vs Std)", k_opt["rebuffer_count"],
              k_opt["rebuffer_count"] - k_std["rebuffer_count"],
              delta_color="inverse")

    st.pyplot(plot_simulation(ev_opt, ev_std, bw))
    st.caption("The player adapts: when bandwidth crashes, it steps down to a "
               "lower rung instead of stalling; when bandwidth recovers, it "
               "climbs back up.")
